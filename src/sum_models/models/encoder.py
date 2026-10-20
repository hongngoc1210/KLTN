"""Fine-tune the whole encoder, with pooling over full window bags."""
from pathlib import Path
import torch
from torch import nn

from sum_models.src.core import DIAGNOSES, LABELS
from sum_models.src.configuration import MODELS, location
from sum_models.src.features import WindowEncoder
from sum_models.src.spans import decode_bio


class EncoderNetwork(nn.Module):
    def __init__(self, backbone, task):
        super().__init__()
        self.backbone, self.task = backbone, task
        size = backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(size * (2 if task == "diagnosis" else 1),
                                      len(LABELS) if task == "classification" else len(DIAGNOSES) if task == "diagnosis" else 3))

    def forward(self, feature):
        pooled, focused, values, indices = [], [], [], []
        device = next(self.parameters()).device
        for bag in feature["bags"]:
            ids = torch.tensor([bag["input_ids"]], device=device)
            hidden = self.backbone(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0]
            if self.task in {"classification", "diagnosis"}:
                pooled.append(hidden[bag["content"]].mean(0))
                if bag["focus"]:
                    focused.append(hidden[bag["focus"]])
            else:
                values.append(hidden[bag["positions"]])
                indices.extend(bag["indices"])
        if self.task in {"classification", "diagnosis"}:
            vector = torch.stack(pooled).mean(0)
            if self.task == "diagnosis":
                vector = torch.cat([vector, torch.cat(focused).mean(0)])
            return self.head(vector)
        hidden = torch.cat(values)
        ix = torch.tensor(indices, device=device)
        summed = hidden.new_zeros((feature["size"], hidden.shape[-1])).index_add(0, ix, hidden)
        counts = torch.bincount(ix, minlength=feature["size"]).clamp_min(1).unsqueeze(1)
        # One supervised loss per original token, after combining all context/answer windows.
        return self.head(summed / counts)


class EncoderRuntime:
    def __init__(self, config, task, revision, training=False):
        from transformers import AutoModel, AutoTokenizer
        self.c, self.task, self.training = config, task, training
        spec = MODELS[config["model"]]
        self.tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=revision, use_fast=not spec.get("segment", False))
        if not spec.get("segment") and not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for exact raw offset alignment")
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<hallu_span>", "</hallu_span>"]})
        backbone = AutoModel.from_pretrained(spec["id"], revision=revision,
                                             attn_implementation=config["attention"])
        backbone.resize_token_embeddings(len(self.tokenizer))
        if training and config["gradient_checkpointing"]:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = EncoderNetwork(backbone, task).to(config["device"])
        segmenter = None
        if spec.get("segment"):
            import py_vncorenlp
            folder = location(config, "vncorenlp_dir")
            if not folder.exists():
                raise FileNotFoundError("Install VnCoreNLP into " + str(folder) + "; see notebook setup")
            segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(folder))
        self.features = WindowEncoder(self.tokenizer, config, task, segmenter)

    def prepare(self, row, training=False):
        return self.features.prepare(row, include_targets=training)

    def loss(self, row, feature):
        logits = self.model(feature)
        if self.task == "classification":
            target = torch.tensor([LABELS.index(row["target"]["label"])], device=logits.device)
            return nn.functional.cross_entropy(logits.unsqueeze(0).float(), target)
        if self.task == "diagnosis":
            target = torch.tensor([float(x in row["target"]["diagnosis_labels"]) for x in DIAGNOSES], device=logits.device)
            return nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
        labels = torch.tensor(feature["labels"], device=logits.device)
        return nn.functional.cross_entropy(logits.float(), labels)

    def predict(self, row, feature=None):
        feature = feature if feature is not None else self.prepare(row)
        logits = self.model(feature).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite encoder logits")
        if self.task == "classification":
            return {"label": LABELS[int(logits.argmax())]}, None
        if self.task == "diagnosis":
            probs = logits.sigmoid()
            labels = [x for x, p in zip(DIAGNOSES, probs.tolist()) if p >= self.c["diagnosis_threshold"]]
            return {"diagnosis_labels": sorted(labels or [DIAGNOSES[int(probs.argmax())]])}, None
        spans = decode_bio(logits.argmax(-1).tolist(), feature["offsets"],
                           row["response"] if self.task == "localization" else row["context"])
        return ({"spans": [dict(s, evidence=[]) for s in spans]} if self.task == "localization" else {"evidence": spans}), None

    def save_weights(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in self.model.state_dict().items()}, folder / "weights.pt")
        self.tokenizer.save_pretrained(folder / "tokenizer")

    def load_weights(self, folder):
        self.model.load_state_dict(torch.load(Path(folder) / "weights.pt", map_location="cpu", weights_only=True))
