"""ViT5 generation and architecture-specific Qwen3.5 / Gemma4 text adapters."""
from pathlib import Path
import torch
import transformers

from sum_models.src.configuration import MODELS
from sum_models.src.core import dumps
from sum_models.src.features import Overlength
from sum_models.prompts.templates import messages
from sum_models.src.spans import parse_prediction


class GenerativeRuntime:
    def __init__(self, config, task, revision, training=False, demos=()):
        self.c, self.task, self.training, self.demos = config, task, training, demos
        spec = MODELS[config["model"]]
        self.family = spec["family"]
        self.seq2seq = self.family == "seq2seq"
        self.processor = None
        dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16 if config["device"].startswith("cuda") else torch.float32
        kwargs = dict(revision=revision, dtype=torch.float32 if training and not config["quantize_4bit"] else dtype,
                      attn_implementation=config["attention"])
        if config["quantize_4bit"]:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
            kwargs["device_map"] = {"": config["device"]}
        if self.seq2seq:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(spec["id"], revision=revision)
            cls = transformers.AutoModelForSeq2SeqLM
        else:
            self.processor = transformers.AutoProcessor.from_pretrained(spec["id"], revision=revision)
            self.tokenizer = self.processor.tokenizer
            class_name = "Qwen3_5ForConditionalGeneration" if self.family == "qwen" else "Gemma4ForConditionalGeneration"
            cls = getattr(transformers, class_name, None)
            if cls is None:
                raise RuntimeError(f"Installed Transformers does not expose {class_name}; use the LLM environment")
        self.model = cls.from_pretrained(spec["id"], **kwargs)
        if not config["quantize_4bit"]:
            # Keep FP32 master weights for full fine-tuning; autocast controls compute dtype.
            self.model = self.model.float() if training else self.model
            self.model.to(config["device"])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if config["method"] == "qlora":
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model,
                use_gradient_checkpointing=training and config["gradient_checkpointing"],
                gradient_checkpointing_kwargs={"use_reentrant": False})
            suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
            targets = [name for name, module in self.model.named_modules()
                       if name.rsplit(".", 1)[-1] in suffixes and
                       not any(part in name for part in ("visual", "vision", "audio")) and
                       hasattr(module, "weight") and module.weight.ndim == 2]
            if not targets:
                raise RuntimeError("No text projection modules found for Qwen3.5 LoRA")
            self.model = get_peft_model(self.model, LoraConfig(task_type="CAUSAL_LM",
                r=config["lora_r"], lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"], target_modules=targets, bias="none"))
            self.lora_targets = targets
        elif training and config["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if training:
            self.model.config.use_cache = False

    def prepare(self, row, training=False):
        msg = messages(row, self.task, self.demos)
        if self.seq2seq:
            prompt = "\n".join(m["role"] + ": " + m["content"] for m in msg) + "\nassistant:"
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            # Render with the model's own processor; text-only input needs no media tensors.
            prompt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                                        enable_thinking=False)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        maximum = self.c["max_length"] if training else self.c["inference_max_length"]
        budget = self.c["max_new_tokens"]
        if len(ids) + (0 if self.seq2seq else budget) > maximum:
            raise Overlength(f"Input {len(ids)} + output reserve {budget} exceeds limit {maximum}")
        item = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if training:
            answer = self.tokenizer.encode(dumps(row["target"]), add_special_tokens=self.seq2seq)
            if not self.seq2seq:
                answer.append(self.tokenizer.eos_token_id)
            if len(answer) > budget:
                raise Overlength(f"Target length {len(answer)} exceeds output budget {budget}")
            if self.seq2seq:
                item["labels"] = answer
            else:
                item = {"input_ids": ids + answer, "attention_mask": [1] * (len(ids) + len(answer)),
                        "labels": [-100] * len(ids) + answer}
        return item

    def tensors(self, feature):
        return {k: torch.tensor([v], device=self.c["device"]) for k, v in feature.items()}

    def loss(self, row, feature):
        return self.model(**self.tensors(feature), use_cache=False).loss

    def predict(self, row, feature=None):
        feature = feature if feature is not None else self.prepare(row)
        tensors = self.tensors(feature)
        generated = self.model.generate(**tensors, max_new_tokens=self.c["max_new_tokens"],
            do_sample=False, use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
        ids = generated[0] if self.seq2seq else generated[0, len(feature["input_ids"]):]
        raw = self.tokenizer.decode(ids, skip_special_tokens=False)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        if len(ids) >= self.c["max_new_tokens"] and int(ids[-1]) not in eos:
            exc = ValueError("Generation reached output budget without an end token")
            exc.raw_output = raw
            raise exc
        if self.family == "gemma" and hasattr(self.processor, "parse_response"):
            parsed = self.processor.parse_response(raw, prefix=tensors["input_ids"])
            # parse_response returns a structured assistant message; only text content is accepted.
            content = parsed.get("content", []) if isinstance(parsed, dict) else []
            if isinstance(content, str):
                clean = content
            else:
                clean = "".join(x.get("text", "") for x in content if x.get("type") == "text")
        else:
            clean = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        try:
            output = parse_prediction(clean, self.task, row)
        except ValueError as exc:
            exc.raw_output = raw
            raise
        return output, raw

    def save_weights(self, folder):
        folder = Path(folder)
        self.model.save_pretrained(folder)
        (self.processor or self.tokenizer).save_pretrained(folder)

    def load_weights(self, folder):
        if self.c["method"] == "qlora":
            from peft import set_peft_model_state_dict
            from peft.utils.save_and_load import load_peft_weights
            state = load_peft_weights(str(folder), device="cpu")
            result = set_peft_model_state_dict(self.model, state)
            if result.unexpected_keys:
                raise ValueError("Unexpected adapter keys")
        else:
            # Reinstantiate to correctly handle tied embeddings and sharded safetensors.
            cls = type(self.model)
            self.model = cls.from_pretrained(folder, dtype=torch.float32,
                                             attn_implementation=self.c["attention"]).to(self.c["device"])
            if self.training and self.c["gradient_checkpointing"]:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
