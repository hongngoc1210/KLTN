"""Character-based, one-to-one span and multilabel metrics with document bootstrap."""
from collections import defaultdict
import numpy as np

from sum_models.src.core import DIAGNOSES, LABELS
from sum_models.src.spans import characters, iou


def prf(tp, predicted, reference):
    return dict(precision=float(tp / predicted) if predicted else 0.0,
                recall=float(tp / reference) if reference else 0.0,
                f1=float(2 * tp / (predicted + reference)) if predicted + reference else 0.0,
                true_positive=int(tp), predicted=int(predicted), support=int(reference))


def matches(gold, predicted, threshold=0.5, exact=False):
    """Maximum-cardinality matching first, then maximum total IoU, stable order."""
    if not gold or not predicted:
        return []
    from scipy.optimize import linear_sum_assignment
    weights = np.zeros((len(gold), len(predicted)))
    for i, g in enumerate(gold):
        for j, p in enumerate(predicted):
            value = iou(g, p)
            ok = (g["start"], g["end"]) == (p["start"], p["end"]) if exact else value >= threshold
            if ok:
                weights[i, j] = min(len(gold), len(predicted)) + 1 + value
    a, b = linear_sum_assignment(-weights)
    return [(int(i), int(j)) for i, j in zip(a, b) if weights[i, j] > 0]


def contribution(row, output, task, valid=True):
    """Flat additive sufficient statistics; bootstrapping never reruns span matching."""
    v = {"total": 1.0, "valid": float(valid)}
    target = row["target"]
    output = output if valid else {}
    if task == "classification":
        truth = LABELS.index(target["label"])
        pred = LABELS.index(output["label"]) if output.get("label") in LABELS else len(LABELS)
        v[f"matrix.{truth}.{pred}"] = 1.0
    elif task == "diagnosis":
        gold, pred = set(target["diagnosis_labels"]), set(output.get("diagnosis_labels", []))
        for i, name in enumerate(DIAGNOSES):
            v[f"diag.{i}.tp"] = float(name in gold & pred)
            v[f"diag.{i}.pred"] = float(name in pred)
            v[f"diag.{i}.gold"] = float(name in gold)
        v["set_correct"] = float(valid and gold == pred)
    elif task == "evidence":
        gold, pred = characters(target["evidence"]), characters(output.get("evidence", []))
        if row.get("evidence_eligible", bool(gold)):
            v.update({"evidence.tp": len(gold & pred), "evidence.pred": len(pred), "evidence.gold": len(gold),
                      "evidence_em.total": 1, "evidence_em.correct": float(valid and target["evidence"] == output.get("evidence"))})
        else:
            v["null.total"] = 1
            v["null.correct"] = float(valid and not pred)
    elif task in {"localization", "pipeline"}:
        gold, pred = target["spans"], output.get("spans", [])
        strict, relaxed = matches(gold, pred, exact=True), matches(gold, pred)
        gc, pc = characters(gold), characters(pred)
        v.update({"exact.tp": len(strict), "exact.pred": len(pred), "exact.gold": len(gold),
                  "relaxed.tp": len(relaxed), "relaxed.pred": len(pred), "relaxed.gold": len(gold),
                  "char.tp": len(gc & pc), "char.pred": len(pc), "char.gold": len(gc),
                  "clean.total": float(not gold), "clean.correct": float(valid and not gold and not pred)})
        pairing = dict(relaxed)
        # Evidence is conditioned on the matched hallucination span, never pooled across the response.
        for i, g in enumerate(gold):
            p = pred[pairing[i]] if i in pairing else None
            ge, pe = characters(g["evidence"]), characters(p["evidence"]) if p else set()
            if ge:
                for suffix, n in (("tp", len(ge & pe)), ("gold", len(ge)), ("pred", len(pe))):
                    v["evidence." + suffix] = v.get("evidence." + suffix, 0) + n
                v["evidence_em.total"] = v.get("evidence_em.total", 0) + 1
                v["evidence_em.correct"] = v.get("evidence_em.correct", 0) + float(valid and p is not None and g["evidence"] == p["evidence"])
            else:
                v["null.total"] = v.get("null.total", 0) + 1
                v["null.correct"] = v.get("null.correct", 0) + float(valid and p is not None and not pe)
        # Unmatched hallucination predictions carrying evidence contribute false positives.
        for j, p in enumerate(pred):
            if j not in pairing.values():
                v["evidence.pred"] = v.get("evidence.pred", 0) + len(characters(p["evidence"]))
        if task == "pipeline":
            for prefix, pairs in (("joint_exact", strict), ("joint_relaxed", relaxed)):
                tp = sum(len(set(gold[i]["diagnosis_labels"]) & set(pred[j].get("diagnosis_labels", []))) for i, j in pairs)
                v.update({prefix + ".tp": tp,
                          prefix + ".pred": sum(max(1, len(p.get("diagnosis_labels", []))) for p in pred),
                          prefix + ".gold": sum(len(g["diagnosis_labels"]) for g in gold)})
    else:
        raise ValueError(task)
    return v


def summarize(v, task):
    def score(prefix):
        return prf(v.get(prefix + ".tp", 0), v.get(prefix + ".pred", 0), v.get(prefix + ".gold", 0))
    def fraction(prefix):
        n = v.get(prefix + ".total", 0)
        return {"accuracy": v.get(prefix + ".correct", 0) / n if n else None, "support": int(n)}
    result = {"total": int(v.get("total", 0)), "valid": int(v.get("valid", 0))}
    result["coverage"] = result["valid"] / result["total"] if result["total"] else 0.0
    if task == "classification":
        matrix = np.array([[v.get(f"matrix.{i}.{j}", 0) for j in range(5)] for i in range(4)])
        result["per_class"] = {name: prf(matrix[i, i], matrix[:, i].sum(), matrix[i].sum()) for i, name in enumerate(LABELS)}
        result["confusion_matrix"] = matrix.astype(int).tolist()
        result["labels"], result["predicted_labels"] = LABELS, LABELS + ["__invalid__"]
        result["accuracy"] = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
        result["macro_f1"] = sum(x["f1"] for x in result["per_class"].values()) / 4
        result["primary"] = result["macro_f1"]
    elif task == "diagnosis":
        result["per_class"] = {name: score(f"diag.{i}") for i, name in enumerate(DIAGNOSES)}
        present = [s["f1"] for s in result["per_class"].values() if s["support"]]
        result["macro_f1"] = sum(present) / len(present) if present else 0.0
        result[f"macro_f1_all_{len(DIAGNOSES)}"] = sum(s["f1"] for s in result["per_class"].values()) / len(DIAGNOSES)
        result["micro"] = prf(*(sum(s[k] for s in result["per_class"].values()) for k in ("true_positive", "predicted", "support")))
        result["exact_set_accuracy"] = v.get("set_correct", 0) / result["total"] if result["total"] else 0.0
        result["primary"] = result["macro_f1"]
    else:
        result["evidence_char"] = score("evidence")
        result["evidence_exact_match"] = fraction("evidence_em")
        result["null_evidence"] = fraction("null")
        if task == "evidence":
            result["primary"] = result["evidence_char"]["f1"]
        else:
            for key in ("exact", "relaxed", "char"):
                result[key] = score(key)
            union = v.get("char.pred", 0) + v.get("char.gold", 0) - v.get("char.tp", 0)
            result["character_iou"] = v.get("char.tp", 0) / union if union else None
            result["clean_response_accuracy"] = fraction("clean")
            result["primary"] = (result["exact"]["f1"] + result["evidence_char"]["f1"]) / 2
            if task == "pipeline":
                result["joint_exact"] = score("joint_exact")
                result["joint_relaxed"] = score("joint_relaxed")
                result["primary"] = result["joint_exact"]["f1"]
    if "per_class" in result:
        for s in result["per_class"].values():
            s["low_support"] = s["support"] < 20
    return result


def add(items):
    total = defaultdict(float)
    for item in items:
        for k, v in item.items():
            total[k] += v
    return dict(total)


def evaluate(rows, predictions, task, bootstrap=0, seed=42):
    unknown = set(predictions) - {r["id"] for r in rows}
    if unknown:
        raise ValueError("Unknown prediction IDs")
    contributions = []
    for r in rows:
        p = predictions.get(r["id"], {})
        contributions.append(contribution(r, p.get("output", {}), task, p.get("status") == "ok"))
    result = summarize(add(contributions), task)
    breakdowns = {}
    for field in ("domain", "annotation_source", "length_bucket"):
        groups = defaultdict(list)
        for r, v in zip(rows, contributions):
            key = r.get(field, "unknown") if field != "length_bucket" else (
                "short_0_1999" if len(r["context"]) + len(r["response"]) < 2000 else
                "medium_2000_3999" if len(r["context"]) + len(r["response"]) < 4000 else "long_4000_plus")
            groups[key].append(v)
        breakdowns[field] = {key: summarize(add(vals), task) for key, vals in sorted(groups.items())}
    result["breakdowns"] = breakdowns
    if bootstrap and rows:
        grouped = defaultdict(list)
        for r, v in zip(rows, contributions):
            grouped[r["document_id"]].append(v)
        docs = [add(grouped[k]) for k in sorted(grouped)]
        keys = sorted(set().union(*(d.keys() for d in docs)))
        matrix = np.array([[d.get(k, 0) for k in keys] for d in docs])
        rng, values = np.random.default_rng(seed), []
        for _ in range(bootstrap):
            weights = np.bincount(rng.integers(0, len(docs), len(docs)), minlength=len(docs))
            total = dict(zip(keys, weights @ matrix))
            values.append(summarize(total, task)["primary"])
        result["primary_ci95"] = [float(x) for x in np.quantile(values, [0.025, 0.975])]
        result["bootstrap"] = {"unit": "document", "iterations": bootstrap, "seed": seed}
    return result
