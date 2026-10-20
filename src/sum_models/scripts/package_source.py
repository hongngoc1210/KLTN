"""Package only current source/config/notebooks, without datasets or run artifacts."""
from pathlib import Path
import sys
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sum_models.src.core import ROOT


def build():
    destination = ROOT / "artifacts" / "bundles" / "sum_models_source.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = list(ROOT.glob("*.py")) + list(ROOT.glob("*.md")) + list(ROOT.glob("requirements*.txt"))
    files.append(ROOT / "pyproject.toml")
    for directory in ("src", "models", "prompts", "configs", "scripts", "tests", "notebooks"):
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".toml", ".ipynb"})
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, "sum_models/" + path.relative_to(ROOT).as_posix())
    return destination


if __name__ == "__main__":
    print(build())
