from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .backends import resolve_device
from .benchmarks import write_benchmark_report
from .config import AppConfig
from .data import load_mat_record, read_manifest_jsonl
from .features import prepare_iq_window, stft_spectrogram
from .model import build_model


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for train/eval commands") from exc
    return torch, F, DataLoader, Dataset


def _labeled_entries(manifest_path: str | Path) -> list[dict[str, Any]]:
    rows = read_manifest_jsonl(manifest_path)
    return [row for row in rows if row.get("valid") and row.get("label")]


def _split_entries(
    entries: list[dict[str, Any]],
    seed: int,
    train_split: float,
    val_split: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    work = list(entries)
    rng.shuffle(work)

    n = len(work)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_train = min(max(1, n_train), n) if n > 0 else 0
    n_val = min(max(1, n_val), max(0, n - n_train)) if n - n_train > 1 else max(0, n - n_train)

    train = work[:n_train]
    val = work[n_train : n_train + n_val]
    test = work[n_train + n_val :]

    if not val and test:
        val, test = test[:1], test[1:]
    if not test and val:
        test, val = val[-1:], val[:-1]

    return train, val, test


def _build_label_map(entries: list[dict[str, Any]]) -> dict[str, int]:
    labels = sorted({str(item["label"]) for item in entries if item.get("label") is not None})
    return {label: idx for idx, label in enumerate(labels)}


def _to_image_tensor(spec: np.ndarray, image_size: int, torch: Any, F: Any) -> Any:
    x = torch.from_numpy(spec.astype(np.float32))
    mean = x.mean()
    std = x.std()
    x = (x - mean) / (std + 1e-6)
    x = x.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    x = x.squeeze(0).repeat(3, 1, 1)
    return x


def _compute_confusion(y_true: list[int], y_pred: list[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _metrics_from_confusion(cm: np.ndarray) -> tuple[float, float, list[dict[str, float]]]:
    eps = 1e-12
    tp = np.diag(cm).astype(np.float64)
    precision = tp / (cm.sum(axis=0) + eps)
    recall = tp / (cm.sum(axis=1) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    accuracy = float(tp.sum() / max(1.0, float(cm.sum())))
    macro_f1 = float(np.mean(f1)) if f1.size else 0.0
    class_metrics = [
        {
            "class": float(i),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i in range(len(f1))
    ]
    return accuracy, macro_f1, class_metrics


def train_model(manifest_path: str | Path, out_dir: str | Path, cfg: AppConfig) -> dict[str, Any]:
    torch, F, DataLoader, Dataset = _require_torch()

    entries = _labeled_entries(manifest_path)
    if len(entries) < 3:
        raise RuntimeError("Need at least 3 labeled valid samples to train")

    label_map = _build_label_map(entries)
    train_rows, val_rows, test_rows = _split_entries(
        entries,
        seed=cfg.train.seed,
        train_split=cfg.train.train_split,
        val_split=cfg.train.val_split,
    )

    class ManifestDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> tuple[Any, Any]:
            row = self.rows[idx]
            record = load_mat_record(row["path"])
            iq = prepare_iq_window(record.iq, cfg.features.sample_size)
            spec = stft_spectrogram(
                iq,
                n_fft=cfg.features.n_fft,
                hop_length=cfg.features.hop_length,
                log_power=cfg.features.log_power,
            )
            x = _to_image_tensor(spec, cfg.features.image_size, torch, F)
            y = torch.tensor(label_map[str(row["label"])], dtype=torch.long)
            return x, y

    train_loader = DataLoader(
        ManifestDataset(train_rows),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        ManifestDataset(val_rows),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    device_name = resolve_device(cfg.runtime.device)
    device = torch.device(device_name)
    model = build_model(len(label_map), arch=cfg.train.arch, pretrained=cfg.train.pretrained).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item())
                val_batches += 1

        train_loss /= max(1, train_batches)
        val_loss /= max(1, val_batches)

        epoch_row = {"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss}
        history.append(epoch_row)

        torch.save(model.state_dict(), out / "last.pt")
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out / "best.pt")

    (out / "label_map.json").write_text(json.dumps(label_map, indent=2, ensure_ascii=True), encoding="utf-8")
    (out / "train_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=True), encoding="utf-8")
    (out / "split.json").write_text(
        json.dumps(
            {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
                "total": len(entries),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        "out_dir": str(out.resolve()),
        "device": device_name,
        "num_labels": len(label_map),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
    }


def evaluate_model(
    manifest_path: str | Path,
    out_dir: str | Path,
    cfg: AppConfig,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    torch, F, DataLoader, Dataset = _require_torch()

    out = Path(out_dir)
    label_map_path = out / "label_map.json"
    if not label_map_path.exists():
        raise RuntimeError("Missing label_map.json. Run training first.")

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))

    entries = _labeled_entries(manifest_path)
    _, _, test_rows = _split_entries(
        entries,
        seed=cfg.train.seed,
        train_split=cfg.train.train_split,
        val_split=cfg.train.val_split,
    )

    if not test_rows:
        raise RuntimeError("No test rows available for evaluation")

    class EvalDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> tuple[Any, Any]:
            row = self.rows[idx]
            record = load_mat_record(row["path"])
            iq = prepare_iq_window(record.iq, cfg.features.sample_size)
            spec = stft_spectrogram(
                iq,
                n_fft=cfg.features.n_fft,
                hop_length=cfg.features.hop_length,
                log_power=cfg.features.log_power,
            )
            x = _to_image_tensor(spec, cfg.features.image_size, torch, F)
            y = torch.tensor(label_map[str(row["label"])], dtype=torch.long)
            return x, y

    test_loader = DataLoader(
        EvalDataset(test_rows),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    device = torch.device(resolve_device(cfg.runtime.device))
    model = build_model(len(label_map), arch=cfg.train.arch, pretrained=False).to(device)

    ckpt = Path(checkpoint) if checkpoint is not None else (out / "best.pt")
    if not ckpt.exists():
        ckpt = out / "last.pt"
    if not ckpt.exists():
        raise RuntimeError("No checkpoint found. Expected best.pt or last.pt")

    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds)

    cm = _compute_confusion(y_true, y_pred, n_classes=len(label_map))
    accuracy, macro_f1, class_metrics = _metrics_from_confusion(cm)

    np.save(out / "confusion_matrix.npy", cm)

    evaluation = {
        "checkpoint": str(ckpt.resolve()),
        "samples": len(y_true),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
    }
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=True), encoding="utf-8")

    write_benchmark_report(out / "BENCHMARK_REPORT.md", accuracy=accuracy, macro_f1=macro_f1, class_metrics=class_metrics)

    return evaluation
