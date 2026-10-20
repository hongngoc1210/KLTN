"""Read frozen source annotations, preserve splits, construct all three tasks."""
from collections import Counter, defaultdict
from pathlib import Path
import unicodedata

from sum_models.src.core import (DIAGNOSES, LABELS, VERSION, canonical_diagnosis, digest, file_hash, iter_jsonl, lock,
                   read_json, verify_manifest, write_json, write_jsonl)
from sum_models.src.configuration import location
from sum_models.src.spans import canonical_spans, overlap, validate_span


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def derive_label(relations):
    errors = set(relations) - {"supported"}
    if not set(relations) <= {"supported", "contradicted", "unsupported"}:
        raise ValueError("Unknown verification label")
    return "mixed" if len(errors) == 2 else "intrinsic" if "contradicted" in errors else "extrinsic" if errors else "no_hallucination"


def convert_source(row, taxonomy):
    inverse = {key: {v: k for k, v in mapping.items()}
               for key, mapping in taxonomy["labels"].items()}
    raw_spans, atoms, relations, issues = [], [], [], []
    context, answer = row["context"], row["hallucinated_response"]
    for atom in row["diagnosis_repair"]:
        annotation = atom["annotation"]
        relation = inverse["claim_verification_label"][annotation["claim_verification_label"]]
        diagnosis = canonical_diagnosis(inverse["diagnosis_type"][annotation["diagnosis_type"]])
        relations.append(relation)
        if relation == "supported":
            if diagnosis != "N/A" or annotation["hallucinated_spans"]:
                raise ValueError("Supported annotation contains error supervision")
            continue
        if diagnosis not in DIAGNOSES or not annotation["hallucinated_spans"]:
            raise ValueError("Error annotation requires diagnosis and hallucinated spans")
        hs = [dict(validate_span(s, answer)) for s in annotation["hallucinated_spans"]]
        es = [dict(validate_span(s, context)) for s in atom["evidence_span"]]
        eligible = atom.get("evidence_scoring_eligible")
        if eligible is not (relation == "contradicted"):
            raise ValueError("Evidence eligibility differs from frozen relation")
        if eligible and (not es or atom.get("evidence_purpose") != "contradiction"):
            raise ValueError("Contradicted annotation requires contradiction evidence")
        if atom.get("evidence_status") == "adjusted":
            issues.append({"kind": "adjusted_evidence_requires_semantic_review", "claim_id": atom["claim_id"]})
        for s in hs:
            if s["quote"].strip().startswith("*") or s["quote"] != s["quote"].strip():
                issues.append({"kind": "span_boundary_review", "claim_id": atom["claim_id"], "span": s})
        atom_id = atom["claim_id"]
        atoms.append(dict(atom_id=atom_id, diagnosis=diagnosis, relation=relation,
                          spans=hs, evidence=es if eligible else [],
                          auxiliary_repair_evidence=es if not eligible else [],
                          evidence_status=atom.get("evidence_status")))
        raw_spans.extend(hs)
    if derive_label(relations) != row["response_label"]:
        raise ValueError("Response label disagrees with diagnosis annotations")
    regions = []
    for s in canonical_spans(raw_spans, answer):
        sources = [a for a in atoms if any(overlap(s, t) for t in a["spans"])]
        region = dict(s, region_id=digest([row["candidate_id"], s["start"], s["end"]]),
                      diagnosis_labels=sorted({a["diagnosis"] for a in sources}),
                      evidence=canonical_spans([e for a in sources for e in a["evidence"]], context),
                      evidence_eligible=any(a["relation"] == "contradicted" for a in sources),
                      source_atom_ids=sorted(a["atom_id"] for a in sources))
        regions.append(region)
    split = row["split"] if isinstance(row["split"], str) else row["split"]["split"]
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    return dict(id=row["candidate_id"], document_id=row["document_id"], split=split,
                domain=row.get("domain", "unknown"), annotation_source=row.get("factual_label_source", "unknown"),
                context=context, question=row["question"], response=answer,
                label=row["response_label"], regions=regions, source_atoms=atoms), issues


def prepare(config):
    source_path, output = location(config, "source_manifest"), location(config, "output_dir")
    print("Checking frozen release hashes...", flush=True)
    source_manifest = verify_manifest(source_path)
    source_root = source_path.parent
    source_hash = file_hash(source_path)
    if (output / "manifest.json").exists():
        result = verify_manifest(output / "manifest.json")
        if result["source_manifest_hash"] != source_hash or result["version"] != VERSION:
            raise ValueError("Prepared directory belongs to a different release/version")
        if result.get("diagnoses") != DIAGNOSES:
            raise ValueError("Prepared dataset uses a different diagnosis taxonomy; migrate or prepare in a new directory")
        return result
    expected = {}
    for r in iter_jsonl(source_root / "responses.jsonl"):
        if r["candidate_id"] in expected:
            raise ValueError("Duplicate release response")
        expected[r["candidate_id"]] = r
    taxonomy = read_json(source_root / "source/taxonomy.json")
    if {canonical_diagnosis(d) for d in taxonomy["labels"]["diagnosis_type"]} != set(DIAGNOSES + ["N/A"]):
        raise ValueError("Unsupported diagnosis taxonomy")
    rows, issues, seen = [], [], set()
    groups = {"document": {}, "context": {}}
    for raw in iter_jsonl(source_root / "source/dataset.jsonl"):
        # Check split leakage over ALL source rows, including excluded/pending rows.
        split = raw["split"] if isinstance(raw["split"], str) else raw["split"]["split"]
        for kind, key in (("document", raw["document_id"]), ("context", digest(normalized(raw["context"])))):
            previous = groups[kind].setdefault(key, split)
            if previous != split:
                raise ValueError("Cross-split " + kind + " leakage")
        cid = raw["candidate_id"]
        if cid not in expected:
            continue
        if cid in seen:
            raise ValueError("Duplicate eligible source response")
        seen.add(cid)
        original = expected[cid]
        if any(original[k] != raw[v] for k, v in
               (("context", "context"), ("question", "question"), ("response", "hallucinated_response"))):
            raise ValueError("Frozen release/source text mismatch")
        row, findings = convert_source(raw, taxonomy)
        rows.append(row)
        issues.extend(dict(x, candidate_id=cid) for x in findings)
    if seen != set(expected):
        raise ValueError("Missing eligible source annotations")
    rows.sort(key=lambda r: r["id"])
    counts = {"responses": len(rows), "regions": sum(len(r["regions"]) for r in rows),
              "multilabel_regions": sum(len(s["diagnosis_labels"]) > 1 for r in rows for s in r["regions"]),
              "splits": dict(Counter(r["split"] for r in rows)),
              "response_labels": dict(Counter(r["label"] for r in rows)),
              "diagnosis_by_split": {sp: dict(Counter(d for r in rows if r["split"] == sp
                   for s in r["regions"] for d in s["diagnosis_labels"])) for sp in ("train", "validation", "test")},
              "issues": dict(Counter(i["kind"] for i in issues))}
    with lock(output):
        if (output / "manifest.json").exists():
            raise ValueError("Concurrent prepare completed; rerun to verify")
        write_jsonl(output / "responses.jsonl", rows)
        write_jsonl(output / "annotation_review.jsonl", issues)
        write_json(output / "audit.json", counts)
        files = {name: file_hash(output / name) for name in
                 ("responses.jsonl", "annotation_review.jsonl", "audit.json")}
        manifest = dict(version=VERSION, source_release_id=source_manifest["release_id"],
                        source_manifest_hash=source_hash, files=files, counts=counts,
                        labels=LABELS, diagnoses=DIAGNOSES,
                        reference="existing_annotations_not_human_verified",
                        quality_certified=source_manifest["audit"].get("quality_certified", False),
                        exact_group_leakage=False, near_duplicate_leakage="not_established")
        write_json(output / "manifest.json", manifest)
    return manifest


def load_dataset(path):
    manifest = verify_manifest(path)
    if manifest["version"] != VERSION:
        raise ValueError("Dataset version mismatch")
    if manifest.get("diagnoses") != DIAGNOSES:
        raise ValueError("Dataset diagnosis taxonomy mismatch; use a dataset prepared for this code version")
    rows = list(iter_jsonl(Path(path).parent / "responses.jsonl"))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prepared response ID")
    return manifest, rows


def task_rows(rows, task, split=None):
    result = []
    for r in rows:
        if split is not None and r["split"] != split:
            continue
        base = {k: r[k] for k in ("id", "document_id", "split", "domain", "annotation_source", "context", "question", "response")}
        if task == "classification":
            result.append(dict(base, target={"label": r["label"]}))
        elif task == "localization":
            result.append(dict(base, target={"spans": [
                {k: s[k] for k in ("start", "end", "quote", "evidence")} for s in r["regions"]]},
                regions=r["regions"], response_label=r["label"]))
        elif task in {"diagnosis", "evidence"}:
            for s in r["regions"]:
                target = ({"diagnosis_labels": s["diagnosis_labels"]} if task == "diagnosis" else {"evidence": s["evidence"]})
                result.append(dict(base, id=s["region_id"], response_id=r["id"],
                    span={k: s[k] for k in ("start", "end", "quote")}, target=target,
                    evidence_eligible=s["evidence_eligible"]))
        else:
            raise ValueError("Unknown task " + task)
    return result
