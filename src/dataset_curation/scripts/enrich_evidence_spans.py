# -*- coding: utf-8 -*-
"""Enrich intrinsic evidence using DeepSeek/Qwen source generations.

Source records are matched by (chunk_id, normalized question). For intrinsic
components, evidence is expanded to the full sentence in content that contains
the source span. Existing human evidence is used as a fallback locator.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = DATA / "hallu_qa_response.json"
TARGET = DATA / "hallu_qa_response_evidence_legacy.json"
DEEPSEEK = DATA / "03_deepseek_v4_flash_0731.json"
QWEN = DATA / "qwen3_8_output.json"
REPORT = ROOT / "reports" / "evidence_unresolved.json"


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def save(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def norm(value):
    value = unicodedata.normalize("NFC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", value)


def compact(value):
    return re.sub(r"\s+", "", norm(value))


def find_loose(content, phrase):
    """Find phrase in content while tolerating arbitrary OCR whitespace."""
    if not isinstance(content, str) or not isinstance(phrase, str):
        return None
    chars = list(re.sub(r"\s+", "", phrase.strip()))
    if not chars:
        return None
    pattern = r"\s*".join(re.escape(char) for char in chars)
    return re.search(pattern, content, flags=re.IGNORECASE)


def find_relaxed(content, phrase):
    """Find a source span, allowing a repeated list head to be omitted.

    Example: source ``khớp cổ chân`` matches ``..., khớp cổ tay, cổ chân``.
    The longest unique token window is preferred to limit false matches.
    """
    exact = find_loose(content, phrase)
    if exact:
        return exact
    if not isinstance(phrase, str):
        return None
    tokens = phrase.split()
    minimum = max(2, (len(tokens) + 1) // 2)
    for size in range(len(tokens) - 1, minimum - 1, -1):
        # Prefer the suffix because Vietnamese coordinated lists commonly omit
        # a repeated head noun: "khớp cổ tay, cổ chân".
        for start in range(len(tokens) - size, -1, -1):
            candidate = " ".join(tokens[start:start + size])
            chars = list(re.sub(r"\s+", "", candidate))
            pattern = r"\s*".join(re.escape(char) for char in chars)
            matches = list(re.finditer(pattern, content, flags=re.IGNORECASE))
            if len(matches) == 1:
                return matches[0]
    return None


def full_sentence(content, phrase):
    match = find_relaxed(content, phrase)
    if not match:
        return None

    # Sentence boundaries require punctuation followed by whitespace and a
    # letter, quote, or bullet. This avoids splitting inside formatted numbers.
    boundaries = list(re.finditer(
        r"[.!?](?:[\"”’']*)\s+(?=(?:\d+\.\s+)?[^\W\d_\s\"“])|\n+",
        content,
        flags=re.UNICODE,
    ))
    start = 0
    end = len(content)
    for boundary in boundaries:
        if boundary.end() <= match.start():
            start = boundary.end()
        elif boundary.start() >= match.end():
            end = boundary.start() + 1 if content[boundary.start()] in ".!?" else boundary.start()
            break
    sentence = content[start:end].strip(" \t\r\n-–—•")
    return sentence or match.group(0).strip()


def build_source_index():
    result = {}
    for path, generation_key in ((DEEPSEEK, "deepseek_generation"), (QWEN, "qwen_generation")):
        for row in load(path):
            generated = (row.get(generation_key) or {}).get("result") or {}
            if generated.get("status") != "ok":
                continue
            key = (row.get("chunk_id"), norm(row.get("question")))
            if key in result:
                raise ValueError(f"Duplicate source key: {key}")
            result[key] = generated
    return result


def components(generated):
    label = generated.get("target_label")
    if label == "mixed_hallucination":
        return generated.get("mixed_components") or []
    if label == "intrinsic_hallucination":
        kind = "intrinsic"
    elif label == "extrinsic_hallucination":
        kind = "extrinsic"
    else:
        return []
    return [{
        "type": kind,
        "hallucinated_span": generated.get("hallucinated_span"),
        "source_span": generated.get("source_span"),
    }]


def match_component(item, source_components):
    intrinsic = [x for x in source_components if x.get("type") == "intrinsic"]
    span = norm(item.get("span"))
    exact = [x for x in intrinsic if norm(x.get("hallucinated_span")) == span]
    if len(exact) == 1:
        return exact[0]
    contained = [
        x for x in intrinsic
        if span and norm(x.get("hallucinated_span"))
        and (span in norm(x.get("hallucinated_span")) or norm(x.get("hallucinated_span")) in span)
    ]
    if len(contained) == 1:
        return contained[0]
    if len(intrinsic) == 1:
        return intrinsic[0]
    return None


def make_item(component, content, previous=None):
    kind = component.get("type")
    span = component.get("hallucinated_span")
    if kind == "extrinsic":
        return {
            "span": span,
            "label": "extrinsic_hallucination",
            "evidence": None,
        }, None

    source_span = component.get("source_span")
    evidence = full_sentence(content, source_span)
    used_fallback = False
    if evidence is None and previous:
        evidence = full_sentence(content, previous.get("evidence"))
        used_fallback = evidence is not None
    if evidence is None and previous:
        evidence = previous.get("evidence")

    item = {
        "span": span if isinstance(span, str) and span.strip() else previous.get("span") if previous else span,
        "label": "intrinsic_hallucination",
        "evidence": evidence,
    }
    unresolved = None
    if not isinstance(source_span, str) or not source_span.strip():
        unresolved = "missing_source_span"
    elif full_sentence(content, source_span) is None and not used_fallback:
        unresolved = "source_and_existing_evidence_not_found_in_content"
    return item, unresolved


def infer_record_label(spans, current):
    labels = {x.get("label") for x in spans}
    if labels == {"intrinsic_hallucination", "extrinsic_hallucination"}:
        return "mixed_hallucination"
    if labels == {"intrinsic_hallucination"}:
        return "intrinsic_hallucination"
    if labels == {"extrinsic_hallucination"}:
        return "extrinsic_hallucination"
    return current


def add_unresolved(unresolved, row, issue, item):
    qa_id = row.get("qa_id")
    if qa_id not in unresolved:
        unresolved[qa_id] = {
            "qa_id": qa_id,
            "chunk_id": row.get("chunk_id"),
            "doc_id": row.get("doc_id"),
            "domain": row.get("domain"),
            "content": row.get("content"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "label": row.get("label"),
            "span_label": row.get("span_label"),
            "question_type": row.get("question_type"),
            "repair_issues": [],
        }
    unresolved[qa_id]["repair_issues"].append({
        "issue": issue,
        "span_label": item,
    })


def main():
    rows = load(SOURCE)
    source_index = build_source_index()
    unresolved = {}
    stats = Counter()

    for row in rows:
        generated = source_index.get((row.get("chunk_id"), norm(row.get("question"))))
        source_components = components(generated) if generated else []
        old_items = row.get("span_label") if isinstance(row.get("span_label"), list) else []

        answer_was_missing = bool(generated and not str(row.get("answer") or "").strip())
        if answer_was_missing:
            row["answer"] = generated.get("hallucinated_response")
            row["label"] = generated.get("target_label")
            stats["answers_restored"] += 1

        # When the final record still uses the source generation label, rebuild
        # every component so mixed samples cannot lose one of their two parts.
        rebuild = bool(
            generated
            and source_components
            and (answer_was_missing or generated.get("target_label") == row.get("label"))
        )
        if rebuild:
            new_items = []
            for component in source_components:
                previous = next(
                    (x for x in old_items if norm(x.get("span")) == norm(component.get("hallucinated_span"))),
                    None,
                )
                item, issue = make_item(component, row.get("content", ""), previous)
                new_items.append(item)
                if issue:
                    add_unresolved(unresolved, row, issue, item)
            row["span_label"] = new_items
            stats["records_rebuilt_from_source"] += 1
        else:
            new_items = []
            for old in old_items:
                if old.get("label") == "extrinsic_hallucination":
                    new_items.append({"span": old.get("span"), "label": "extrinsic_hallucination", "evidence": None})
                    continue
                if old.get("label") != "intrinsic_hallucination":
                    new_items.append(old)
                    continue
                component = match_component(old, source_components)
                if component:
                    item, issue = make_item(component, row.get("content", ""), old)
                    stats["intrinsic_components_matched"] += 1
                else:
                    evidence = full_sentence(row.get("content", ""), old.get("evidence")) or old.get("evidence")
                    item = dict(old)
                    item["evidence"] = evidence
                    issue = "no_matching_source_component"
                new_items.append(item)
                if issue:
                    add_unresolved(unresolved, row, issue, item)
            row["span_label"] = new_items

        if row.get("label") != "no_hallucination":
            inferred = infer_record_label(row.get("span_label") or [], row.get("label"))
            if inferred != row.get("label"):
                row["label"] = inferred
                stats["record_labels_normalized"] += 1

    ids = [row.get("qa_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate qa_id after enrichment")
    save(TARGET, rows)

    unresolved_rows = []
    for qa_id, report_row in unresolved.items():
        repaired_row = next(row for row in rows if row.get("qa_id") == qa_id)
        report_row["label"] = repaired_row.get("label")
        report_row["span_label"] = repaired_row.get("span_label")
        report_row["answer"] = repaired_row.get("answer")
        unresolved_rows.append(report_row)
    save(REPORT, unresolved_rows)
    print("rows", len(rows))
    print("stats", dict(stats))
    print("unresolved_qa", len(unresolved_rows))
    print("unresolved_issues", sum(len(row["repair_issues"]) for row in unresolved_rows))
    print("labels", dict(Counter(row.get("label") for row in rows)))
    print("output", TARGET)
    print("report", REPORT)


if __name__ == "__main__":
    main()
