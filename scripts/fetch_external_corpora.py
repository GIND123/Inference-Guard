"""Fetch external training corpora into data/external/ (with a dry-run plan).

Lists and downloads the sources documented in docs/DATA_SOURCES.md. Nothing
runs automatically: `--dry-run` (default) prints what would be fetched and the
license note; `--download --source <name>` actually pulls it.

    python scripts/fetch_external_corpora.py --dry-run
    python scripts/fetch_external_corpora.py --download --source personahub --limit 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXTERNAL_DIR = REPO_ROOT / "data" / "external"

SOURCES: Dict[str, Dict] = {
    "personahub": {
        "dataset": "proj-persona/PersonaHub",
        "config": "persona",
        "split": "train",
        "license": "cc-by-nc-sa-4.0 (non-commercial, share-alike; same family as SynthPAI)",
        "purpose": "persona-driven synthesis: seed the local teacher to write first-person texts, then position-control with src/rewriter/augment.py",
        "columns": ["persona", "general domain (top 1 percent)", "specific domain (top 1 percent)"],
        "size": "200K personas in the 'persona' split",
    },
    "ai4privacy": {
        "dataset": "ai4privacy/pii-masking-400k",
        "config": None,
        "split": "train",
        "license": "custom license.md - REVIEW BEFORE USE (not a standard OSS license)",
        "purpose": "real-register texts (business/legal/psychology) with PII spans; use as carriers or explicit-PII examples",
        "columns": ["source_text", "masked_text", "privacy_mask", "mbert_token_classes"],
        "size": "326K train rows, EN/FR/DE/ES/IT",
    },
    "no_robots": {
        "dataset": "HuggingFaceH4/no_robots",
        "config": None,
        "split": "train",
        "license": "check the dataset card before use",
        "purpose": "real human-written user prompts as diverse carriers after PII stripping",
        "columns": ["messages (prompt, response), category"],
        "size": "~9.5K prompts",
    },
    "evol_instruct": {
        "dataset": "WizardLM/Evol-Instruct-110K",
        "config": None,
        "split": "train",
        "license": "check the dataset card before use",
        "purpose": "additional instruction/query carriers (dedupe and filter to question-style)",
        "columns": ["instruction", "output"],
        "size": "110K rows",
    },
}


def plan() -> List[Dict]:
    rows = []
    for name, spec in SOURCES.items():
        rows.append({
            "source": name,
            "dataset": spec["dataset"],
            "rows": spec["size"],
            "license": spec["license"],
            "purpose": spec["purpose"],
            "download_command": f"python scripts/fetch_external_corpora.py --download --source {name}",
        })
    return rows


def fetch(source: str, limit: Optional[int], out_dir: Path) -> Path:
    from datasets import load_dataset

    spec = SOURCES[source]
    ds = load_dataset(
        spec["dataset"],
        spec["config"],
        split=spec["split"],
        streaming=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
            if limit is not None and n >= limit:
                break
    meta = {
        "source": source,
        "dataset": spec["dataset"],
        "license": spec["license"],
        "rows": n,
        "note": "External corpus. License recorded above; see docs/DATA_SOURCES.md before training on it.",
    }
    (out_dir / f"{source}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the source table (default)")
    parser.add_argument("--download", action="store_true", help="Actually download")
    parser.add_argument("--source", choices=sorted(SOURCES), default=None, help="Which source to download")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to stream per source")
    args = parser.parse_args(argv)

    if not args.download or args.source is None:
        print(json.dumps(plan(), indent=2))
        print("\nDry run only. To fetch: python scripts/fetch_external_corpora.py --download --source <name> [--limit N]")
        return 0

    path = fetch(args.source, args.limit, EXTERNAL_DIR)
    print(f"downloaded {args.source} -> {path}")
    print(f"license reminder: {SOURCES[args.source]['license']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
