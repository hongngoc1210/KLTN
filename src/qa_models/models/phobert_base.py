from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from qa_models.configs.schema import ExperimentConfig
from qa_models.tasks.layer1.encoder_adapter import EncoderLayer1Adapter


class PhoBERTBaseV2Adapter(EncoderLayer1Adapter):
    """Layer-1 PhoBERT adapter with VnCoreNLP word segmentation."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self._segmenter: Any | None = None
        
    def _get_vncorenlp_dir(self) -> Path:
        configured_path = self.config.model.params.get(
            "vncorenlp_dir",
            "qa_models/resources/vncorenlp",
        )

        model_dir = Path(configured_path)

        if not model_dir.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            model_dir = project_root / model_dir

        return model_dir.resolve()
    
    def _get_segmenter(self) -> Any:
        if self._segmenter is not None:
            return self._segmenter

        try:
            import py_vncorenlp
        except ImportError as error:
            raise RuntimeError(
                "PhoBERT requires py_vncorenlp. "
                "Install it before training."
            ) from error

        model_dir = self._get_vncorenlp_dir()

        if not model_dir.exists():
            raise FileNotFoundError(
                "VnCoreNLP model directory does not exist: "
                f"{model_dir}. Download the model before training."
            )

        self._segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=str(model_dir),
        )

        return self._segmenter
    
    @lru_cache(maxsize=32_768)
    def preprocess_text(self, text: str) -> str:
        normalized_text = " ".join(text.split())

        if not normalized_text:
            return ""

        segmenter = self._get_segmenter()
        segmented_sentences = segmenter.word_segment(normalized_text)

        return " ".join(
            sentence.strip()
            for sentence in segmented_sentences
            if sentence.strip()
        )
        
    def clear_preprocess_cache(self) -> None:
        self.preprocess_text.cache_clear()