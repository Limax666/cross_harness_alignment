"""Validate notebook syntax and execute its CPU-only filtering cells locally."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "experiments/cross_harness_sft/notebooks/Qwen3.5-4B_AgentDojo_Codex_Claude_SFT.ipynb"
OUTPUTS = ROOT / "experiments/cross_harness_sft/outputs"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.lstrip().startswith("%pip"):
            continue
        ast.parse(source, filename=f"cell-{index}")
    namespace = {
        "CODEX_INPUT": OUTPUTS / "codex_agentdojo_native_gpt56_sol_full/raw_native.jsonl",
        "CLAUDE_INPUT": OUTPUTS / "claude_agentdojo_openrouter_gpt56_sol_full/raw_native.jsonl",
        "RUN_DIR": ROOT / "experiments/cross_harness_sft/data/sft/agentdojo_colab_validation",
    }
    for index in (5, 6):
        source = "".join(notebook["cells"][index]["source"])
        exec(compile(source, f"cell-{index}", "exec"), namespace)
    report = namespace["filter_report"]
    print(json.dumps({
        "source": report["source_counts"],
        "accepted_before_dedup": report["accepted_before_dedup"],
        "duplicates": report["exact_duplicates"],
        "train": report["train"],
        "validation": report["validation"],
        "family_overlap": report["family_overlap"],
        "rejection_reasons": report["rejection_reasons"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
