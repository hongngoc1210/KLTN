# -*- coding: utf-8 -*-

"""V4 = rubric V3 mở rộng + evidence validation + verbalized confidence."""

import re

from src.judge import GemmaJudge
from src.v4_validation import normalize_evidence


class GemmaJudgeV4(GemmaJudge):
    """Evaluator một call qua OpenRouter, không áp policy routing."""

    def _normalize_result(self, result, original_label, context, answer):
        return normalize_evidence(result, original_label, context, answer)

    def _validate_result(self, result):
        if result.get("evidence_validation") != "schema_and_exact_quotes_passed":
            raise ValueError("Evidence chưa được validate.")

    def _build_user_prompt(self, context, question, answer, label):
        # Nhãn gốc chỉ được Python dùng để tính label_match sau khi model trả lời.
        return self._replace(self.user_prompt_template, {
            "{context}": context,
            "{question}": question,
            "{answer}": answer,
        })

    @staticmethod
    def _replace(template, values):
        # Thay một lượt để placeholder xuất hiện trong dữ liệu không bị thay tiếp.
        pattern = "|".join(re.escape(key) for key in values)
        return re.sub(pattern, lambda match: str(values[match.group()] or ""), template)

    def judge(self, sample):
        result = super().judge(sample)
        if not result["success"]:
            return result

        evaluation = result["evaluation"]
        for field in ("evidence_validation", "original_label"):
            evaluation.pop(field, None)
        if not evaluation.get("quality_errors"):
            evaluation.pop("quality_errors", None)
        return result
