"""Validation deterministic cho output evidence-first của V4."""

from copy import deepcopy


SCORE_NAMES = (
    "plausibility", "completeness", "question_relevance",
    "internal_consistency", "clarity", "fluency",
)
HALLUCINATION_LABELS = {
    "intrinsic_hallucination", "extrinsic_hallucination", "mixed_hallucination",
}
VALID_LABELS = {"no_hallucination", *HALLUCINATION_LABELS, "invalid"}
ALTERNATIVE_LABELS = VALID_LABELS - {"invalid"}
UNCERTAINTY_FACTORS = {
    "context_truncated_or_malformed",
    "ambiguous_question",
    "implicit_or_non_exhaustive_context",
    "replacement_vs_unsupported_boundary",
    "evidence_requires_nontrivial_inference",
    "multiple_reasonable_labels",
}


def normalize_evidence(raw, original_label, context, answer, *, include_scores=True):
    """Kiểm tra schema/trích dẫn và tính các field deterministic bằng Python."""
    if not isinstance(raw, dict):
        raise ValueError("Evidence phải là JSON object.")

    # Chỉ giữ schema công khai để output không phình nếu model tự thêm field.
    fields = (
        "question_status", "is_evaluable", "contradicted_claims",
        "unsupported_claims", "proposed_label", "alternative_label",
        "self_contradiction", "uncertainty_factors", "confidence",
        "short_reason",
    )
    result = {field: deepcopy(raw.get(field)) for field in fields}
    if include_scores:
        result["scores"] = deepcopy(raw.get("scores"))

    if type(result["is_evaluable"]) is not bool:
        raise ValueError("is_evaluable phải là boolean.")
    if result["question_status"] not in {
        "answerable", "unanswerable", "false_premise", "ambiguous",
    }:
        raise ValueError("question_status không hợp lệ.")
    if result["proposed_label"] not in VALID_LABELS:
        raise ValueError("proposed_label không hợp lệ.")
    if not isinstance(result["short_reason"], str) or not result["short_reason"].strip():
        raise ValueError("short_reason phải là chuỗi không rỗng.")

    factors = result["uncertainty_factors"]
    if not isinstance(factors, list) or any(item not in UNCERTAINTY_FACTORS for item in factors):
        raise ValueError("uncertainty_factors chứa giá trị không hợp lệ.")
    if len(factors) != len(set(factors)):
        raise ValueError("uncertainty_factors không được lặp.")

    alternative = result["alternative_label"]
    if alternative is not None and alternative not in ALTERNATIVE_LABELS:
        raise ValueError("alternative_label không hợp lệ.")
    if alternative == result["proposed_label"]:
        raise ValueError("alternative_label phải khác proposed_label.")
    if alternative is not None and not factors:
        raise ValueError("alternative_label khác null yêu cầu uncertainty_factors không rỗng.")

    confidence = result["confidence"]
    if type(confidence) not in (int, float) or not 0.50 <= confidence <= 1.00:
        raise ValueError("confidence phải là số trong khoảng 0.50–1.00.")
    result["confidence"] = round(float(confidence), 2)
    if factors and result["confidence"] > 0.84:
        raise ValueError("uncertainty_factors không rỗng yêu cầu confidence ≤ 0.84.")
    if alternative is not None and result["confidence"] > 0.84:
        raise ValueError("alternative_label khác null yêu cầu confidence ≤ 0.84.")
    if result["confidence"] == 1.00 and (factors or alternative is not None):
        raise ValueError("confidence=1.00 yêu cầu không có uncertainty hoặc alternative label.")
    result["confidence_source"] = "model_verbalized_uncalibrated"

    seen = set()
    spans_by_field = {}
    for field in ("contradicted_claims", "unsupported_claims"):
        claims = result[field]
        if not isinstance(claims, list):
            raise ValueError(f"{field} phải là list, không được bỏ field.")
        spans_by_field[field] = []
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError(f"{field}: claim phải là object.")
            span = claim.get("answer_span")
            if not isinstance(span, str) or not span.strip() or span not in answer:
                raise ValueError(f"{field}: answer_span không nguyên văn trong ANSWER: {span!r}")
            key = " ".join(span.split()).casefold()
            if key in seen:
                raise ValueError(f"Claim trùng trong/hai evidence arrays: {span!r}")
            seen.add(key)
            spans_by_field[field].append(span)
            if field == "contradicted_claims":
                quote = claim.get("context_evidence")
                if not isinstance(quote, str) or not quote.strip() or quote not in context:
                    raise ValueError(f"context_evidence không nguyên văn trong CONTEXT: {quote!r}")

    contradicted = result["contradicted_claims"]
    unsupported = result["unsupported_claims"]
    if any(a in b or b in a for a in spans_by_field["contradicted_claims"]
           for b in spans_by_field["unsupported_claims"]):
        raise ValueError("answer_span giữa contradicted và unsupported bị chồng lấn.")

    if not result["is_evaluable"]:
        if contradicted or unsupported or result["proposed_label"] != "invalid":
            raise ValueError("is_evaluable=false yêu cầu claims rỗng và proposed_label=invalid.")
        if alternative is not None or result["question_status"] != "ambiguous":
            raise ValueError("Sample invalid yêu cầu alternative_label=null và question_status=ambiguous.")
        if result["confidence"] != 0.50:
            raise ValueError("Sample invalid yêu cầu confidence=0.50.")
    elif result["proposed_label"] == "invalid":
        raise ValueError("is_evaluable=true không được proposed_label=invalid.")

    if not result["is_evaluable"]:
        actual = "invalid"
    elif contradicted and unsupported:
        actual = "mixed_hallucination"
    elif contradicted:
        actual = "intrinsic_hallucination"
    elif unsupported:
        actual = "extrinsic_hallucination"
    else:
        actual = "no_hallucination"
    if result["proposed_label"] != actual:
        raise ValueError(
            f"proposed_label={result['proposed_label']} không khớp evidence arrays={actual}."
        )

    result.update(
        actual_label=actual,
        original_label=original_label,
        label_match=actual == original_label,
        has_hallucination=actual in HALLUCINATION_LABELS,
        contradicted_spans=[claim["answer_span"] for claim in contradicted],
        unsupported_spans=[claim["answer_span"] for claim in unsupported],
        evidence_validation="schema_and_exact_quotes_passed",
    )

    if not include_scores:
        if type(result["self_contradiction"]) is not bool:
            raise ValueError("self_contradiction phải là boolean.")
        for field in ("needs_review", "auto_accept", "routing_decision", "routing_policy_status"):
            result.pop(field, None)
        return result

    # Lỗi quality không làm mất kết luận factual đã có evidence hợp lệ.
    errors = []
    scores = result["scores"]
    if not isinstance(scores, dict):
        errors.append("scores phải là object")
    else:
        for name in SCORE_NAMES:
            value = scores.get(name)
            if type(value) is not int or not 0 <= value <= 5:
                errors.append(f"{name} phải là integer 0–5")
    if type(result["self_contradiction"]) is not bool:
        errors.append("self_contradiction phải là boolean")
    if (not errors and result["self_contradiction"]
            and scores["internal_consistency"] > 1):
        errors.append("self_contradiction=true yêu cầu internal_consistency ≤ 1")

    result["quality_errors"] = errors
    result["quality_status"] = "error" if errors else "success"
    if errors:
        result["raw_scores"] = deepcopy(scores)
        result["raw_self_contradiction"] = result["self_contradiction"]
        result["scores"] = None
        if type(result["self_contradiction"]) is not bool:
            result["self_contradiction"] = None
    else:
        result["scores"] = {name: scores[name] for name in SCORE_NAMES}
        result["scores"]["total_score"] = sum(scores[name] for name in SCORE_NAMES)

    for field in ("needs_review", "auto_accept", "routing_decision", "routing_policy_status"):
        result.pop(field, None)
    return result
