"""Character offsets, canonical overlap unions and strict prediction validation."""
from sum_models.src.core import DIAGNOSES, LABELS, strict_json


def validate_span(span, text):
    if not isinstance(span, dict) or set(span) != {"start", "end", "quote"}:
        raise ValueError("A span requires exactly start, end, quote")
    a, b = span["start"], span["end"]
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
        raise ValueError("Invalid [start,end) character offsets")
    if not isinstance(span["quote"], str) or text[a:b] != span["quote"]:
        raise ValueError("Quote does not match original text at offsets")
    return span


def canonical_spans(spans, text):
    intervals = sorted({(validate_span(s, text)["start"], s["end"]) for s in spans})
    result = []
    for a, b in intervals:
        if result and a < result[-1][1]:  # touching intervals retain their boundary
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return [dict(start=a, end=b, quote=text[a:b]) for a, b in result]


def overlap(a, b):
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def iou(a, b):
    n = overlap(a, b)
    return n / (a["end"] - a["start"] + b["end"] - b["start"] - n)


def characters(spans):
    return set(i for s in spans for i in range(s["start"], s["end"]))


def bio_labels(offsets, spans):
    """Overlap alignment, including subwords intersecting a partial-word annotation."""
    result, previous = [], None
    for a, b in offsets:
        matches = [i for i, s in enumerate(spans) if max(a, s["start"]) < min(b, s["end"])]
        current = matches[0] if matches else None
        result.append(0 if current is None else 2 if current == previous else 1)
        previous = current
    return result


def decode_bio(labels, offsets, text):
    result = []
    active = False
    for label, (a, b) in zip(labels, offsets):
        if b <= a:
            continue
        if label == 1 or (label == 2 and not active):
            result.append([a, b])
            active = True
        elif label == 2:
            result[-1][1] = b
        elif label == 0:
            active = False
        else:
            raise ValueError("Invalid BIO label")
    return [dict(start=a, end=b, quote=text[a:b]) for a, b in result]


def validate_prediction(value, task, row):
    if not isinstance(value, dict):
        raise ValueError("Prediction must be a JSON object")
    if task == "classification":
        if set(value) != {"label"} or value["label"] not in LABELS:
            raise ValueError("Unknown response label")
    elif task == "diagnosis":
        labels = value.get("diagnosis_labels")
        if (set(value) != {"diagnosis_labels"} or not isinstance(labels, list) or
                not labels or any(not isinstance(x, str) for x in labels) or
                len(set(labels)) != len(labels) or not set(labels) <= set(DIAGNOSES)):
            raise ValueError("Expected a nonempty unique list of diagnosis labels")
        value = {"diagnosis_labels": sorted(labels)}
    elif task == "evidence":
        if set(value) != {"evidence"} or not isinstance(value["evidence"], list):
            raise ValueError("Expected evidence list")
        value = {"evidence": canonical_spans(value["evidence"], row["context"])}
    elif task == "localization":
        if set(value) != {"spans"} or not isinstance(value["spans"], list):
            raise ValueError("Expected spans list")
        for item in value["spans"]:
            if not isinstance(item, dict) or set(item) != {"start", "end", "quote", "evidence"}:
                raise ValueError("Each hallucination span requires linked evidence")
            validate_span({k: item[k] for k in ("start", "end", "quote")}, row["response"])
            if not isinstance(item["evidence"], list):
                raise ValueError("Expected evidence list")
            for e in item["evidence"]:
                validate_span(e, row["context"])
        # Apply exactly the same overlapping-region convention as reference preparation.
        merged = canonical_spans([{k: s[k] for k in ("start", "end", "quote")}
                                  for s in value["spans"]], row["response"])
        value = {"spans": [dict(s, evidence=canonical_spans(
            [e for t in value["spans"] if overlap(s, t) for e in t["evidence"]], row["context"]))
                           for s in merged]}
    else:
        raise ValueError("Unknown task " + task)
    return value


def parse_prediction(text, task, row):
    return validate_prediction(strict_json(text), task, row)
