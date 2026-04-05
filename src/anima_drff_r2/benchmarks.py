from __future__ import annotations

from pathlib import Path


def write_benchmark_report(
    output_path: str | Path,
    accuracy: float,
    macro_f1: float,
    class_metrics: list[dict[str, float]],
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Benchmark Report",
        "",
        f"- Accuracy: {accuracy:.6f}",
        f"- Macro F1: {macro_f1:.6f}",
        "",
        "## Per-Class Metrics",
        "",
        "| class | precision | recall | f1 |",
        "|---|---:|---:|---:|",
    ]

    for item in class_metrics:
        lines.append(
            f"| {int(item['class'])} | {item['precision']:.6f} | {item['recall']:.6f} | {item['f1']:.6f} |"
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
