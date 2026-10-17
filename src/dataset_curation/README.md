# Dataset curation

## Qwen-hard evidence annotation with OpenAI Batch

Entry point:

```bash
python3 main_evidence_openai_batch.py prepare
python3 main_evidence_openai_batch.py submit
python3 main_evidence_openai_batch.py status
python3 main_evidence_openai_batch.py collect
```

`prepare` only validates input and creates the request JSONL. `submit` is the only command that creates a paid Batch job.

Default model: `gpt-5.6-sol`. Override it before creating a state file:

```bash
export OPENAI_EVIDENCE_MODEL=gpt-5.6-sol
```

Files:

- Config: `configs/evidence_openai_batch.py`
- Core logic: `src/evidence_openai_batch.py`
- Input: `reports/evidence_unresolved_qwen_hard.json`
- Result proposals: `output/qwen_hard_evidence_openai_results.json`
- Batch state: `state/qwen_hard_evidence_openai_batch_state.json`

The model is only allowed to assign `evidence`. It cannot change the fixed
label, span, answer, or question type. The collect step validates
that intrinsic evidence is copied verbatim from `content` and extrinsic
evidence is null. It writes proposals to the output file and does not modify
the dataset automatically.
