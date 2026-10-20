"""Window bags: no per-window duplication of response-level labels."""
from functools import lru_cache
from sum_models.src.spans import bio_labels


class Overlength(ValueError):
    pass


def windows(size, capacity, overlap):
    if capacity < 1:
        raise Overlength("No room for document tokens")
    if size == 0:
        return [(0, 0)]
    step = max(1, int(capacity * (1 - overlap)))
    result = []
    for a in range(0, size, step):
        result.append((a, min(a + capacity, size)))
        if a + capacity >= size:
            break
    return result


class WindowEncoder:
    def __init__(self, tokenizer, config, task, segmenter=None):
        self.tokenizer, self.c, self.task, self.segmenter = tokenizer, config, task, segmenter
        self.begin = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        self.end = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        if self.begin is None or self.end is None:
            raise ValueError("Encoder requires explicit boundary tokens")
        self.open_marker = tokenizer.convert_tokens_to_ids("<hallu_span>")
        self.close_marker = tokenizer.convert_tokens_to_ids("</hallu_span>")

    @lru_cache(maxsize=2048)
    def tokenize(self, text):
        if self.segmenter:
            segmented = " ".join(self.segmenter.word_segment(text)) if text.strip() else ""
            ids = self.tokenizer.encode(segmented, add_special_tokens=False)
            return tuple(ids), None
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return tuple(encoded["input_ids"]), tuple(tuple(x) for x in encoded["offset_mapping"])

    def prepare(self, row, include_targets=True):
        qids, _ = self.tokenize(row["question"])
        if len(qids) > self.c["question_tokens"]:
            raise Overlength(f"Question has {len(qids)} tokens, budget {self.c['question_tokens']}; no silent truncation")
        cids, coffsets = self.tokenize(row["context"])
        if self.task in {"diagnosis", "evidence"}:
            span = row["span"]
            pieces = [row["response"][:span["start"]], row["response"][span["start"]:span["end"]], row["response"][span["end"]:]]
            ids = [list(self.tokenize(s)[0]) for s in pieces]
            aids = ids[0] + [self.open_marker] + ids[1] + [self.close_marker] + ids[2]
            focus = set(range(len(ids[0]) + 1, len(ids[0]) + 1 + len(ids[1])))
            if not focus:
                raise Overlength("Target span has no model tokens")
            aoffsets = None
        else:
            aids, aoffsets = self.tokenize(row["response"])
            focus = set()
        answer_capacity = self.c["answer_window_tokens"]
        context_capacity = self.c["max_length"] - len(qids) - answer_capacity - 4
        # [CLS] question [SEP] answer-window [SEP] context-window [SEP]
        bags = []
        for aa, ab in windows(len(aids), answer_capacity, self.c["overlap"]):
            for ca, cb in windows(len(cids), context_capacity, self.c["overlap"]):
                ids = [self.begin] + list(qids) + [self.end] + list(aids[aa:ab]) + [self.end] + list(cids[ca:cb]) + [self.end]
                answer_start = len(qids) + 2
                context_start = answer_start + ab - aa + 1
                positions = (list(range(answer_start, answer_start + ab - aa)) if self.task == "localization" else
                             list(range(context_start, context_start + cb - ca)) if self.task == "evidence" else [])
                indices = (list(range(aa, ab)) if self.task == "localization" else
                           list(range(ca, cb)) if self.task == "evidence" else [])
                bags.append(dict(input_ids=ids, positions=positions, indices=indices,
                                 focus=[answer_start + j - aa for j in range(aa, ab) if j in focus],
                                 content=[i for i in range(1, len(ids) - 1)
                                          if i not in {len(qids) + 1, context_start - 1}]))
        result = {"bags": bags}
        if self.task in {"localization", "evidence"}:
            offsets = aoffsets if self.task == "localization" else coffsets
            if offsets is None:
                raise ValueError("Tagging requires a tokenizer with raw offsets")
            result["offsets"] = offsets
            result["size"] = len(offsets)
            if include_targets and "target" in row:
                spans = row["target"]["spans"] if self.task == "localization" else row["target"]["evidence"]
                result["labels"] = bio_labels(offsets, spans)
        return result
