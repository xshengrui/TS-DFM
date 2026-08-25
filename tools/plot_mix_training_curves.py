from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TRAIN_RE = re.compile(
    r"Train >> Epoch:\s*(\d+) \| Iter:\s*(\d+) \| loss: ([0-9.eE+-]+) "
    r"\| lr: ([0-9.eE+-]+) \| grad norm: ([0-9.eE+-]+)"
)
VAL_RE = re.compile(r"Val loss:\s*([0-9.eE+-]+)\s*\t epoch\s*(\d+)")
BEST_RE = re.compile(r"Best: val loss:\s*([0-9.eE+-]+)\s*\t epoch\s*(\d+)")
TEST_RE = re.compile(r"Test loss:\s*([0-9.eE+-]+)")


def median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def parse_log(label: str, path: Path, color: str) -> dict:
    epoch_losses: dict[int, list[float]] = {}
    epoch_lrs: dict[int, list[float]] = {}
    epoch_grads: dict[int, list[float]] = {}
    val_losses: dict[int, float] = {}
    best: tuple[int, float] | None = None
    test_loss: float | None = None
    train_points = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            train_match = TRAIN_RE.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(3))
                lr = float(train_match.group(4))
                grad = float(train_match.group(5))
                epoch_losses.setdefault(epoch, []).append(loss)
                epoch_lrs.setdefault(epoch, []).append(lr)
                epoch_grads.setdefault(epoch, []).append(grad)
                train_points += 1
                continue

            val_match = VAL_RE.search(line)
            if val_match:
                val_losses[int(val_match.group(2))] = float(val_match.group(1))
                continue

            best_match = BEST_RE.search(line)
            if best_match:
                best = (int(best_match.group(2)), float(best_match.group(1)))
                continue

            test_match = TEST_RE.search(line)
            if test_match:
                test_loss = float(test_match.group(1))

    epochs = sorted(epoch_losses)
    val_epochs = sorted(val_losses)
    if best is None and val_epochs:
        best_epoch = min(val_epochs, key=lambda epoch: val_losses[epoch])
        best = (best_epoch, val_losses[best_epoch])

    return {
        "label": label,
        "path": str(path),
        "color": color,
        "epochs": epochs,
        "train_mean": [sum(epoch_losses[e]) / len(epoch_losses[e]) for e in epochs],
        "train_median": [median(epoch_losses[e]) for e in epochs],
        "lr": [sum(epoch_lrs[e]) / len(epoch_lrs[e]) for e in epochs],
        "grad": [sum(epoch_grads[e]) / len(epoch_grads[e]) for e in epochs],
        "val_epochs": val_epochs,
        "val_loss": [val_losses[e] for e in val_epochs],
        "best": best,
        "test_loss": test_loss,
        "train_points": train_points,
    }


def write_summary(path: Path, runs: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run",
                "train_epochs",
                "val_epochs",
                "train_first",
                "train_last",
                "val_first",
                "val_last",
                "best_epoch",
                "best_val",
                "test_loss",
                "train_points",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run["label"],
                    len(run["epochs"]),
                    len(run["val_epochs"]),
                    run["train_mean"][0] if run["train_mean"] else "",
                    run["train_mean"][-1] if run["train_mean"] else "",
                    run["val_loss"][0] if run["val_loss"] else "",
                    run["val_loss"][-1] if run["val_loss"] else "",
                    run["best"][0] if run["best"] else "",
                    run["best"][1] if run["best"] else "",
                    run["test_loss"],
                    run["train_points"],
                ]
            )


def make_canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw, ImageFont.ImageFont]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    return image, draw, font


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def blend(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    return tuple(int(255 * (1 - alpha) + channel * alpha) for channel in color)


def nice_ticks(max_value: float, count: int = 5) -> list[float]:
    if max_value <= 0:
        return [0]
    step = max_value / count
    magnitude = 10 ** int(f"{step:e}".split("e")[1])
    normalized = step / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    tick = nice * magnitude
    values = []
    current = 0.0
    while current <= max_value * 1.001:
        values.append(current)
        current += tick
    return values


def draw_panel(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    box: tuple[int, int, int, int],
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[dict],
    y_log: bool = False,
    annotate_best: bool = False,
) -> None:
    left, top, right, bottom = box
    pad_left, pad_right, pad_top, pad_bottom = 70, 24, 42, 54
    px0, py0 = left + pad_left, top + pad_top
    px1, py1 = right - pad_right, bottom - pad_bottom

    all_x = [x for item in series for x in item["x"]]
    all_y = [y for item in series for y in item["y"] if y > 0 or not y_log]
    if not all_x or not all_y:
        return
    xmin, xmax = min(all_x), max(all_x)
    ymin = min(all_y) if y_log else 0.0
    ymax = max(all_y)
    if y_log:
        ymin = max(ymin, 1e-8)
        log_min, log_max = math.log10(ymin), math.log10(ymax)
        if log_min == log_max:
            log_max = log_min + 1
    else:
        ymax *= 1.08
    if xmin == xmax:
        xmax = xmin + 1
    if ymax <= 0:
        ymax = 1.0

    def sx(value: float) -> int:
        return int(px0 + (value - xmin) / (xmax - xmin) * (px1 - px0))

    def sy(value: float) -> int:
        if y_log:
            value = max(value, ymin)
            ratio = (math.log10(value) - log_min) / (log_max - log_min)
        else:
            ratio = value / ymax
        return int(py1 - ratio * (py1 - py0))

    draw.rectangle((px0, py0, px1, py1), outline=(190, 190, 190), width=1)
    draw.text((left + 12, top + 10), title, fill=(20, 20, 20), font=font)
    draw.text(((px0 + px1) // 2 - 20, bottom - 32), xlabel, fill=(50, 50, 50), font=font)
    draw.text((left + 10, (py0 + py1) // 2), ylabel, fill=(50, 50, 50), font=font)

    for tick in [xmin, xmin + (xmax - xmin) / 2, xmax]:
        x = sx(tick)
        draw.line((x, py1, x, py1 + 5), fill=(80, 80, 80))
        draw.text((x - 18, py1 + 9), f"{tick:.0f}", fill=(70, 70, 70), font=font)

    yticks = [10**p for p in range(math.floor(log_min), math.ceil(log_max) + 1)] if y_log else nice_ticks(ymax)
    for tick in yticks:
        y = sy(tick)
        draw.line((px0 - 5, y, px1, y), fill=(230, 230, 230))
        label = f"{tick:.0e}" if y_log else f"{tick:.3g}"
        draw.text((left + 12, y - 7), label, fill=(70, 70, 70), font=font)

    legend_x, legend_y = px1 - 260, py0 + 8
    for index, item in enumerate(series):
        color = hex_to_rgb(item["color"])
        if item.get("alpha"):
            color = blend(color, item["alpha"])
        y = legend_y + index * 20
        draw.line((legend_x, y + 8, legend_x + 30, y + 8), fill=color, width=item.get("width", 2))
        draw.text((legend_x + 36, y), item["label"], fill=(40, 40, 40), font=font)

        points = [(sx(x), sy(y)) for x, y in zip(item["x"], item["y"]) if y > 0 or not y_log]
        if len(points) > 1:
            draw.line(points, fill=color, width=item.get("width", 2))

    if annotate_best:
        for item in series:
            best = item.get("best")
            if not best:
                continue
            x, y = best
            color = hex_to_rgb(item["color"])
            cx, cy = sx(x), sy(y)
            draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)
            draw.text((cx + 8, cy - 18), f"best {y:.4f}, ep {x}", fill=color, font=font)


def plot_overview(path: Path, runs: list[dict]) -> None:
    image, draw, font = make_canvas(2400, 1600)
    draw.text((42, 24), "Mixed TS-DFM training logs: original vs noise fix", fill=(20, 20, 20), font=font)
    panels = [
        (40, 80, 1180, 780),
        (1220, 80, 2360, 780),
        (40, 840, 1180, 1540),
        (1220, 840, 2360, 1540),
    ]
    draw_panel(
        draw,
        font,
        panels[0],
        "Training loss, per-epoch mean",
        "Epoch",
        "Loss",
        [{"x": r["epochs"], "y": r["train_mean"], "color": r["color"], "label": r["label"], "width": 3} for r in runs],
    )
    draw_panel(
        draw,
        font,
        panels[1],
        "Validation loss",
        "Epoch",
        "Val loss",
        [
            {
                "x": r["val_epochs"],
                "y": r["val_loss"],
                "color": r["color"],
                "label": r["label"],
                "width": 3,
                "best": r["best"],
            }
            for r in runs
        ],
        annotate_best=True,
    )
    draw_panel(
        draw,
        font,
        panels[2],
        "Learning rate",
        "Epoch",
        "LR",
        [{"x": r["epochs"], "y": r["lr"], "color": r["color"], "label": r["label"], "width": 3} for r in runs],
        y_log=True,
    )
    draw_panel(
        draw,
        font,
        panels[3],
        "Gradient norm, per-epoch mean",
        "Epoch",
        "Grad norm",
        [{"x": r["epochs"], "y": r["grad"], "color": r["color"], "label": r["label"], "width": 2} for r in runs],
    )
    image.save(path)


def plot_train_vs_val(path: Path, runs: list[dict]) -> None:
    image, draw, font = make_canvas(2200, 820)
    series = []
    for run in runs:
        series.append(
            {
                "x": run["epochs"],
                "y": run["train_mean"],
                "color": run["color"],
                "alpha": 0.38,
                "label": f"{run['label']} train mean",
                "width": 2,
            }
        )
        series.append(
            {
                "x": run["val_epochs"],
                "y": run["val_loss"],
                "color": run["color"],
                "label": f"{run['label']} val",
                "width": 3,
            }
        )
    draw_panel(
        draw,
        font,
        (40, 40, 2160, 780),
        "Train mean vs validation loss",
        "Epoch",
        "Loss",
        series,
    )
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-log", required=True, type=Path)
    parser.add_argument("--noise-log", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        parse_log("Original mix / no noise", args.original_log, "#4C78A8"),
        parse_log("Noise fix", args.noise_log, "#F58518"),
    ]

    overview = args.out_dir / "mix_training_curves_overview.png"
    train_vs_val = args.out_dir / "mix_train_vs_val_loss.png"
    summary = args.out_dir / "mix_training_curve_summary.csv"
    plot_overview(overview, runs)
    plot_train_vs_val(train_vs_val, runs)
    write_summary(summary, runs)

    print(overview)
    print(train_vs_val)
    print(summary)
    for run in runs:
        best = run["best"]
        print(
            f"{run['label']}: train_epochs={len(run['epochs'])} "
            f"val_epochs={len(run['val_epochs'])} best={best} "
            f"test={run['test_loss']} last_train={run['train_mean'][-1]:.6f} "
            f"last_val={run['val_loss'][-1]:.6f}"
        )


if __name__ == "__main__":
    main()
