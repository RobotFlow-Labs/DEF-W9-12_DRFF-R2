from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .data import build_manifest_entries, load_mat_record, read_manifest_jsonl, write_manifest_jsonl
from .features import prepare_iq_window, save_feature_npz, stft_spectrogram
from .pipeline import evaluate_model, train_model

app = typer.Typer(add_completion=False, help="ANIMA DRFF-R2 command line interface")


@app.command()
def index(
    data_root: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    manifest_out: Path = typer.Option(Path("artifacts/manifest.jsonl")),
    report_out: Path = typer.Option(Path("artifacts/index_report.json")),
) -> None:
    entries = build_manifest_entries(data_root)
    write_manifest_jsonl(entries, manifest_out)

    valid = sum(1 for item in entries if item["valid"])
    invalid = len(entries) - valid
    report = {
        "data_root": str(data_root.resolve()),
        "manifest": str(manifest_out.resolve()),
        "files_total": len(entries),
        "files_valid": valid,
        "files_invalid": invalid,
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


@app.command()
def features(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(Path("artifacts/features")),
    config: Optional[Path] = typer.Option(None),
    limit: int = typer.Option(32, min=1),
) -> None:
    cfg = load_config(config)
    rows = [row for row in read_manifest_jsonl(manifest) if row.get("valid")]
    if not rows:
        raise typer.BadParameter("No valid rows in manifest")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for row in rows[:limit]:
        record = load_mat_record(row["path"])
        iq = prepare_iq_window(record.iq, cfg.features.sample_size)
        spec = stft_spectrogram(
            iq,
            n_fft=cfg.features.n_fft,
            hop_length=cfg.features.hop_length,
            log_power=cfg.features.log_power,
        )

        feature_name = f"{Path(row['path']).stem}.npz"
        metadata = {
            "path": row["path"],
            "label": row.get("label"),
            "n_fft": cfg.features.n_fft,
            "hop_length": cfg.features.hop_length,
            "sample_size": cfg.features.sample_size,
        }
        save_feature_npz(out_dir / feature_name, spec, metadata)
        written += 1

    typer.echo(f"wrote {written} feature files to {out_dir}")


@app.command()
def train(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(Path("artifacts/train")),
    config: Optional[Path] = typer.Option(Path("configs/default.toml")),
) -> None:
    cfg = load_config(config)
    result = train_model(manifest, out_dir, cfg)
    typer.echo(json.dumps(result, indent=2, ensure_ascii=True))


@app.command()
def eval(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(Path("artifacts/train")),
    config: Optional[Path] = typer.Option(Path("configs/default.toml")),
    checkpoint: Optional[Path] = typer.Option(None),
) -> None:
    cfg = load_config(config)
    result = evaluate_model(manifest, out_dir, cfg, checkpoint=checkpoint)
    typer.echo(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":  # pragma: no cover
    app()
