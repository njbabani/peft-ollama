"""Report figures. Every figure is drawn from files saved in a run directory."""

import collections
import json
import math
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from groundedqa.metrics import question_outcomes, roc_curve, threshold_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
GOOD = "#0ca30c"
GOOD_TEXT = "#006300"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order

VARIANTS = ["base", "base_rag", "adapter", "adapter_rag"]
VARIANT_LABELS = {
    "base": "Base",
    "base_rag": "Base + RAG",
    "adapter": "Adapter",
    "adapter_rag": "Adapter + RAG",
}
VARIANT_COLORS = dict(zip(VARIANTS, SERIES))
VARIANT_MARKERS = {"base": "o", "base_rag": "s", "adapter": "D", "adapter_rag": "^"}
MODE_LABELS = {
    "dense": "Dense (MiniLM)",
    "bm25": "BM25",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
}
ROLE_LABELS = {
    "answerable": "Answerable",
    "unanswerable": "Source-unanswerable",
    "retrieval_miss": "Retrieval miss",
}

# Outcome colours encode correctness; labels carry the behaviour.
OUTCOMES = {
    "answerable": [
        ("answer_good", "Answered, F1 ≥ 0.5", GOOD),
        ("answer_bad", "Answered, F1 < 0.5", CRITICAL),
        ("abstained", "Abstained (missed)", SERIOUS),
        ("invalid", "Invalid JSON", WARNING),
    ],
    "unanswerable": [
        ("abstained", "Abstained (correct)", GOOD),
        ("answer_bad", "Answered (unsupported)", CRITICAL),
        ("invalid", "Invalid JSON", WARNING),
    ],
}

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10.5,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "svg.fonttype": "none",
    }
)


def finite(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def fmt(value, spec=".2f", missing="n/a"):
    number = finite(value)
    return missing if number is None else format(number, spec)


def pct(value, missing="n/a"):
    number = finite(value)
    return missing if number is None else f"{number:.0%}"


def _top(fig, inches):
    """Figure fraction for a distance measured down from the top edge."""
    return 1 - inches / fig.get_figheight()


def figure_header(fig, eyebrow, title, subtitle=None, x=0.045):
    fig.text(x, _top(fig, 0.32), eyebrow.upper(), color=MUTED, fontsize=9.5, weight="bold", va="top")
    fig.text(x, _top(fig, 0.55), title, color=INK, fontsize=19, weight="bold", va="top")
    if subtitle:
        fig.text(x, _top(fig, 1.02), subtitle, color=INK_2, fontsize=11, va="top")


def stat_tiles(fig, tiles, top_in, x0=0.045, x1=0.97):
    """tiles: (label, value, note, note_color[, (text, color) ...]) in one row."""
    width = (x1 - x0) / max(len(tiles), 1)
    for index, (label, value, note, note_color, *extra) in enumerate(tiles):
        x = x0 + index * width
        fig.text(x, _top(fig, top_in), label, color=INK_2, fontsize=9.5, va="top")
        fig.text(x, _top(fig, top_in + 0.22), value, color=INK, fontsize=21, weight="bold", va="top")
        lines = ([(note, note_color)] if note else []) + list(extra)
        for line, (text, color) in enumerate(lines):
            fig.text(
                x, _top(fig, top_in + 0.66 + 0.19 * line), text, color=color or INK_2, fontsize=9.5, va="top"
            )


def clean_axis(ax, grid="x"):
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def empty_axis(ax, message="Not recorded"):
    clean_axis(ax, grid=None)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=11)


def footnote(fig, text, x=0.045):
    fig.text(x, 0.014, text, color=MUTED, fontsize=8.5, va="bottom")


def save_figure(fig, path: Path, dpi: int = 160) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=dpi, facecolor=SURFACE)
            path.write_bytes(buffer.getvalue())
    finally:
        plt.close(fig)
    return path


def ema(values, alpha=0.15):
    smoothed, current = [], None
    for value in values:
        current = value if current is None else alpha * value + (1 - alpha) * current
        smoothed.append(current)
    return smoothed


def log_series(rows, key):
    points = []
    for row in rows:
        step, value = finite(row.get("step")), finite(row.get(key))
        if step is not None and value is not None:
            points.append((step, value))
    return sorted(points)


def ink_on(color):
    r, g, b = mpl.colors.to_rgb(color)
    return INK if 0.2126 * r**2.2 + 0.7152 * g**2.2 + 0.0722 * b**2.2 > 0.28 else "#ffffff"


def stacked_share_bars(ax, rows, categories):
    """rows: [(label, Counter of category keys)], drawn as 100% horizontal stacks."""
    axis_inches = ax.get_position().width * ax.figure.get_figwidth()
    for y, (_label, counts) in enumerate(rows):
        total = sum(counts.values())
        left = 0.0
        for key, _name, color in categories:
            share = counts.get(key, 0) / total if total else 0.0
            if share <= 0:
                continue
            ax.barh(y, share, left=left, height=0.56, color=color, edgecolor=SURFACE, linewidth=2)
            label = f"{share:.0%}"
            if share * axis_inches >= 0.1 * len(label) + 0.16:
                ax.text(
                    left + share / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    color=ink_on(color),
                    fontsize=9.5,
                    weight="bold",
                )
            left += share
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    clean_axis(ax, grid=None)


LABEL_SLOTS = [
    (10, 0, "left", "center"),
    (-10, 0, "right", "center"),
    (0, 10, "center", "bottom"),
    (0, -10, "center", "top"),
    (8, 8, "left", "bottom"),
    (-8, 8, "right", "bottom"),
    (8, -8, "left", "top"),
    (-8, -8, "right", "top"),
    (0, 22, "center", "bottom"),
    (0, -22, "center", "top"),
]


def place_labels(ax, labels, obstacles=(), pad=3):
    """Direct-label points without collisions; skip any label that cannot fit."""
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer).expanded(1.02, 1.08)
    taken = list(obstacles)
    for text, x, y, style in labels:
        for dx, dy, ha, va in LABEL_SLOTS:
            note = ax.annotate(
                text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **style
            )
            box = note.get_window_extent(renderer).padded(pad)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
            if inside and not any(box.overlaps(other) for other in taken):
                taken.append(box)
                break
            note.remove()


def plot_evidence_audit(audit: dict):
    top_k = audit["top_k"]
    split = audit["splits"]["train"]
    passed = {name: not failures for name, failures in audit["shortcut_failures"].items()}
    roles = [r for r in ROLE_LABELS if r in split["by_role"]]
    fig = plt.figure(figsize=(14, 5.8))
    figure_header(
        fig,
        "Training data · shortcut check",
        "Evidence layout carries no label information"
        if all(passed.values())
        else "Evidence layout leaks the label",
        f"{split['retained']:,} of {split['selected']:,} training questions kept · every example gets "
        f"{top_k} retrieved passages ({audit['retrieval_mode']}); the source passage rotates position",
    )
    for index, (name, ok) in enumerate(passed.items()):
        fig.text(
            0.70 + index * 0.14,
            _top(fig, 0.6),
            f"{name}  {'✓ PASS' if ok else '✗ FAIL'}",
            color=GOOD_TEXT if ok else CRITICAL,
            fontsize=11,
            weight="bold",
            va="top",
        )
    grid = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=_top(fig, 2.1), bottom=0.14, wspace=0.18)
    panels = [
        ("evidence_count", "Passages supplied per example", list(range(1, top_k + 1)), str, roles),
        (
            "anchor_position",
            "Position of the source passage",
            list(range(top_k)),
            lambda v: f"E{v + 1}",
            [r for r in roles if r != "retrieval_miss"],
        ),
    ]
    for column, (key, title, keys, tick, panel_roles) in enumerate(panels):
        ax = fig.add_subplot(grid[0, column])
        clean_axis(ax, grid="y")
        width = 0.8 / max(len(panel_roles), 1)
        for offset, role in enumerate(panel_roles):
            stats = split["by_role"][role]
            counts = {int(k): v for k, v in stats[key].items()}
            total = sum(counts.values()) or 1
            shares = [counts.get(k, 0) / total for k in keys]
            xs = [i - 0.4 + width * (offset + 0.5) for i in range(len(keys))]
            bars = ax.bar(
                xs,
                shares,
                width=width * 0.9,
                color=SERIES[list(ROLE_LABELS).index(role)],
                label=f"{ROLE_LABELS[role]} (n = {stats['n']:,})",
            )
            for bar, share in zip(bars, shares):
                if share > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        share + 0.02,
                        f"{share:.0%}",
                        ha="center",
                        va="bottom",
                        color=INK_2,
                        fontsize=8.5,
                    )
        ax.set_xticks(range(len(keys)), [tick(k) for k in keys])
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Share of role's examples")
        if column == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.1), ncols=3)
    footnote(
        fig,
        "Retrieval-miss examples withhold the source passage and have no anchor; "
        "none of their passages contains the gold answer string.",
    )
    return fig


def plot_training_dashboard(history: list[dict], summary: dict, mode: str):
    train_loss = log_series(history, "loss")
    eval_loss = log_series(history, "eval_loss")
    auroc = log_series(history, "eval_decision_auroc")
    lr = log_series(history, "learning_rate")
    grad = log_series(history, "grad_norm")
    initial, final = summary.get("initial_monitor", {}), summary.get("final_monitor", {})
    fig = plt.figure(figsize=(15, 9.4))
    figure_header(
        fig,
        f"Training · {mode} · {summary.get('device', '')}",
        "LoRA fine-tuning overview",
        "Weighted reply-only loss (decision token up-weighted) · thin: per step · bold: moving average",
    )
    wall = finite(summary.get("wall_seconds"))
    stat_tiles(
        fig,
        [
            (
                "Wall time",
                f"{wall / 60:.0f} min" if wall is not None else "n/a",
                f"{summary.get('optimizer_steps', 0)} steps · {summary.get('examples_seen', 0):,} examples",
                None,
            ),
            (
                "Held-out decision AUROC",
                fmt(final.get("eval_decision_auroc"), ".3f"),
                f"from {fmt(initial.get('eval_decision_auroc'), '.3f')} before training",
                None,
            ),
            (
                "Held-out loss",
                fmt(final.get("eval_loss"), ".3f"),
                f"from {fmt(initial.get('eval_loss'), '.3f')}",
                None,
            ),
            (
                "Trainable parameters",
                f"{summary.get('trainable_fraction', 0):.2%}",
                f"{summary.get('trainable_parameters', 0) / 1e6:.1f}M LoRA weights",
                None,
            ),
            ("Peak memory", f"{fmt(summary.get('peak_memory_gib'), '.1f')} GiB", "device allocator", None),
        ],
        top_in=1.5,
    )
    grid = fig.add_gridspec(
        2, 2, left=0.06, right=0.97, top=_top(fig, 2.75), bottom=0.1, hspace=0.45, wspace=0.16
    )

    ax = fig.add_subplot(grid[0, 0])
    clean_axis(ax, grid="y")
    ax.set_title("Loss")
    if train_loss:
        steps, values = zip(*train_loss)
        ax.plot(steps, values, color=SERIES[0], linewidth=1, alpha=0.3)
        ax.plot(steps, ema(values), color=SERIES[0], label="Training (moving average)")
    if eval_loss:
        ax.plot(
            *zip(*eval_loss),
            color=SERIES[1],
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label="Held-out monitor",
        )
    if train_loss or eval_loss:
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Optimizer step")
        ax.legend(loc="upper right")
    else:
        empty_axis(ax, "No loss logged")

    ax = fig.add_subplot(grid[0, 1])
    clean_axis(ax, grid="y")
    ax.set_title("Held-out abstain-decision AUROC")
    if auroc:
        ax.plot(
            *zip(*auroc),
            color=SERIES[1],
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.axhline(0.5, color=BASELINE, linewidth=1)
        ax.text(0, 0.5, " chance", color=MUTED, fontsize=8.5, va="bottom")
        last_step, last = auroc[-1]
        ax.annotate(
            f"{last:.3f}",
            (last_step, last),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=INK_2,
            fontsize=9,
        )
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("Optimizer step")
    else:
        empty_axis(ax, "No monitor evaluations")

    ax = fig.add_subplot(grid[1, 0])
    clean_axis(ax, grid="y")
    ax.set_title("Learning rate")
    if lr:
        ax.plot(*zip(*lr), color=SERIES[0])
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{v:.1e}" if v else "0"))
        ax.set_xlabel("Optimizer step")
    else:
        empty_axis(ax)

    ax = fig.add_subplot(grid[1, 1])
    clean_axis(ax, grid="y")
    ax.set_title("Gradient norm (before clipping)")
    if grad:
        steps, values = zip(*grad)
        ax.plot(steps, values, color=SERIES[0], linewidth=1, alpha=0.3)
        ax.plot(steps, ema(values), color=SERIES[0])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Optimizer step")
    else:
        empty_axis(ax)
    footnote(
        fig,
        "Monitor = held-out validation questions with training-style evidence. "
        "Decision AUROC ranks P(abstain) for should-abstain examples above should-answer ones.",
    )
    return fig


def plot_retrieval_benchmark(bench: dict):
    modes = [m for m in MODE_LABELS if m in bench["modes"]]
    ks = [1, 3, 5, 10]
    fig = plt.figure(figsize=(14, 5.6))
    best = max(modes, key=lambda m: bench["modes"][m]["recall@3"])
    figure_header(
        fig,
        "Retrieval benchmark · test split",
        f"{MODE_LABELS[best]} finds the gold passage in the top 3 for "
        f"{bench['modes'][best]['recall@3']:.0%} of questions",
        f"{bench['n_questions']:,} answerable questions over {bench['corpus_chunks']:,} chunks "
        "from every paragraph of the held-out articles · relevant = chunk contains the answer span",
    )
    grid = fig.add_gridspec(
        1, 2, left=0.06, right=0.97, top=_top(fig, 1.75), bottom=0.13, wspace=0.28, width_ratios=[1.2, 1]
    )
    ax = fig.add_subplot(grid[0, 0])
    clean_axis(ax, grid="y")
    ax.set_title("Recall@k (gold chunk in top k)")
    for index, mode in enumerate(modes):
        values = [bench["modes"][mode][f"recall@{k}"] for k in ks]
        ax.plot(
            ks,
            values,
            color=SERIES[index],
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=MODE_LABELS[mode],
        )
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower right")

    ax = fig.add_subplot(grid[0, 1])
    clean_axis(ax, grid="x")
    ax.set_title("MRR and time per query")
    mrr = [bench["modes"][m]["mrr"] for m in modes]
    ax.barh(range(len(modes)), mrr, color=[SERIES[i] for i in range(len(modes))], height=0.5)
    for y, mode in enumerate(modes):
        ms = bench["modes"][mode]["seconds_per_query"] * 1000
        ax.text(mrr[y], y, f"  {mrr[y]:.3f} · {ms:.2g} ms", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(range(len(modes)), [MODE_LABELS[m] for m in modes])
    ax.set_ylim(len(modes) - 0.4, -0.6)
    ax.set_xlim(0, 1.3)
    ax.set_xlabel("Mean reciprocal rank (top 10)")
    footnote(
        fig,
        f"Hybrid fuses dense and BM25 top-{bench['candidates']} lists with reciprocal rank fusion; "
        "rerank scores those candidates with a MiniLM cross-encoder. Batched time on this machine.",
    )
    return fig


def plot_calibration(val_records: dict, test_records: dict, summary: dict):
    variants = [v for v in VARIANTS if v in summary["variants"]]
    fig = plt.figure(figsize=(14, 6.4))
    figure_header(
        fig,
        "Abstention calibration",
        "Separating answerable from unanswerable questions",
        'No-answer probability = P(" true") at the decision token · thresholds chosen on validation only',
    )
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=_top(fig, 1.75), bottom=0.12, wspace=0.22)
    ax = fig.add_subplot(grid[0, 0])
    clean_axis(ax, grid="both")
    ax.set_title("Test ROC: abstaining on unanswerable questions")
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1)
    for variant in variants:
        rows = test_records[variant]
        fpr, tpr = roc_curve([r["unanswerable"] for r in rows], [r["na_prob"] for r in rows])
        auc = summary["variants"][variant]["test"]["abstention_auroc"]
        ax.plot(
            fpr,
            tpr,
            color=VARIANT_COLORS[variant],
            label=f"{VARIANT_LABELS[variant]}  AUROC {fmt(auc, '.3f')}",
        )
    ax.set_xlabel("False-abstention rate on answerable")
    ax.set_ylabel("Abstention rate on unanswerable")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(grid[0, 1])
    clean_axis(ax, grid="y")
    ax.set_title("Validation F1 by abstention threshold")
    for variant in variants:
        thresholds, f1 = threshold_curve(val_records[variant])
        ax.plot(
            thresholds,
            f1,
            color=VARIANT_COLORS[variant],
            linewidth=1.8,
            drawstyle="steps-post",
            label=VARIANT_LABELS[variant],
        )
        chosen = summary["variants"][variant]["threshold"]
        best = f1[min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - chosen))]
        ax.scatter(
            [min(chosen, 1)],
            [best],
            color=VARIANT_COLORS[variant],
            s=64,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
            marker=VARIANT_MARKERS[variant],
        )
    ax.axvline(0.5, color=BASELINE, linewidth=1)
    ax.text(0.5, 0.02, " 0.5 (uncalibrated)", color=MUTED, fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Abstain when P(no answer) ≥ threshold")
    ax.set_ylabel("Mean token F1")
    ax.legend(loc="lower left", title="marker = chosen threshold", title_fontsize=8.5)
    footnote(
        fig,
        "Test ROC is threshold-free. Each variant's threshold maximises validation F1 "
        "(the SQuAD 2.0 'best_f1_thresh' rule) and is then applied unchanged to test.",
    )
    return fig


def _outcome_key(outcome: dict, record: dict) -> str:
    if outcome["abstained"]:
        return "abstained"
    if not record["answer_valid"]:
        return "invalid"
    if record["unanswerable"]:
        return "answer_bad"
    return "answer_good" if outcome["token_f1"] >= 0.5 else "answer_bad"


def plot_results_overview(test_records: dict, outcomes: dict, summary: dict, mode: str):
    present = [v for v in VARIANTS if v in summary["variants"]]
    stats = {v: summary["variants"][v] for v in present}
    questions = test_records[present[0]]
    n_answerable = sum(not r["unanswerable"] for r in questions)
    hero = stats.get("adapter_rag", {}).get("test", {})
    ref = stats.get("base_rag", {}).get("test", {})
    f1, f1_ref = finite(hero.get("token_f1")), finite(ref.get("token_f1"))
    delta = f1 - f1_ref if f1 is not None and f1_ref is not None else None

    fig = plt.figure(figsize=(15, 12.6))
    figure_header(
        fig,
        f"Held-out evaluation · {mode} · calibrated thresholds",
        "Four-way ablation: adapter × retrieval",
        f"{len(questions)} title-disjoint test questions · {n_answerable} answerable · "
        f"{len(questions) - n_answerable} source-unanswerable · retrieval: {summary['retrieval_mode']}",
    )
    stat_tiles(
        fig,
        [
            (
                "Adapter + RAG token F1",
                fmt(f1),
                f"{delta:+.2f} vs base + RAG" if delta is not None else None,
                GOOD_TEXT if (delta or 0) > 0 else CRITICAL if (delta or 0) < 0 else INK_2,
                (f"abstain-only baseline {(len(questions) - n_answerable) / len(questions):.2f}", INK_2),
            ),
            (
                "Answerable F1 (adapter + RAG)",
                fmt(hero.get("answerable_f1")),
                f"answers {pct(hero.get('answerable_answer_rate'))} of answerable",
                None,
            ),
            (
                "Unanswerable abstention",
                pct(hero.get("unanswerable_abstain_rate")),
                f"adapter + RAG · threshold {fmt(stats.get('adapter_rag', {}).get('threshold'), '.2f')}",
                None,
            ),
            (
                "Abstention AUROC",
                fmt(hero.get("abstention_auroc"), ".3f"),
                f"base + RAG {fmt(ref.get('abstention_auroc'), '.3f')}",
                None,
            ),
            (
                "Gold passage in top-3",
                pct(hero.get("gold_chunk_recall")),
                f"citation hits gold {pct(hero.get('citation_hit_rate'))}",
                None,
            ),
        ],
        top_in=1.55,
    )
    grid = fig.add_gridspec(
        2,
        3,
        left=0.15,
        right=0.975,
        top=_top(fig, 3.2),
        bottom=0.1,
        hspace=0.36,
        wspace=0.34,
        width_ratios=[1, 1, 1.1],
        height_ratios=[1, 1.35],
    )
    for column, kind in enumerate(("answerable", "unanswerable")):
        ax = fig.add_subplot(grid[0, column])
        rows = []
        for variant in present:
            counts = collections.Counter(
                _outcome_key(o, r)
                for o, r in zip(outcomes[variant], test_records[variant])
                if r["unanswerable"] == (kind == "unanswerable")
            )
            rows.append((VARIANT_LABELS[variant], counts))
        stacked_share_bars(ax, rows, OUTCOMES[kind])
        if column:
            ax.set_yticklabels([])
        ax.set_title("Answerable questions" if kind == "answerable" else "Source-unanswerable questions")
        ax.legend(
            handles=[Patch(color=c, label=n) for _, n, c in OUTCOMES[kind]],
            loc="upper left",
            bbox_to_anchor=(0, -0.1),
            ncols=2,
        )

    scatter = ax = fig.add_subplot(grid[0, 2])
    clean_axis(ax, grid="both")
    ax.set_title("Answering vs abstaining")
    ax.plot([0, 1], [1, 0], color=BASELINE, linewidth=1, zorder=1)
    points = []
    for variant in present:
        test = stats[variant]["test"]
        x, y = finite(test.get("answerable_answer_rate")), finite(test.get("unanswerable_abstain_rate"))
        if x is None or y is None:
            continue
        points.append((variant, x, y))
        ax.scatter(
            [x],
            [y],
            s=110,
            marker=VARIANT_MARKERS[variant],
            color=VARIANT_COLORS[variant],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
            clip_on=False,
        )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Answer rate on answerable")
    ax.set_ylabel("Abstention on unanswerable")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))

    ax = fig.add_subplot(grid[1, 0:2])
    clean_axis(ax, grid="x")
    ax.set_title("Scores with 95% bootstrap intervals")
    metric_rows = [
        ("token_f1", "Token F1 · all"),
        ("exact_match", "Exact match · all"),
        ("token_f1@answerable", "Token F1 · answerable"),
        ("answered@answerable", "Answer rate · answerable"),
        ("abstained@unanswerable", "Abstention · unanswerable"),
    ]
    offsets = np.linspace(-0.27, 0.27, len(VARIANTS))
    for row, (key, _label) in enumerate(metric_rows):
        for variant in present:
            mean, low, high = stats[variant]["intervals"][key]
            if mean is None:
                continue
            y = row + offsets[VARIANTS.index(variant)]
            color = VARIANT_COLORS[variant]
            ax.plot([low, high], [y, y], color=color, linewidth=2, alpha=0.85)
            ax.scatter(
                [mean],
                [y],
                s=46,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
                zorder=3,
                marker=VARIANT_MARKERS[variant],
            )
    for index in range(1, len(metric_rows)):
        ax.axhline(index - 0.5, color=GRID, linewidth=0.8)
    ax.set_yticks(range(len(metric_rows)), [label for _, label in metric_rows])
    ax.set_ylim(len(metric_rows) - 0.5, -0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=VARIANT_MARKERS[v],
                linestyle="",
                markersize=8,
                color=VARIANT_COLORS[v],
                label=VARIANT_LABELS[v],
            )
            for v in present
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncols=len(present),
    )

    ax = fig.add_subplot(grid[1, 2])
    clean_axis(ax, grid="x")
    ax.set_title("Per-question time (amortised, batch)")
    rng = np.random.default_rng(0)
    for y, variant in enumerate(present):
        values = np.array([r["latency_s"] for r in test_records[variant]])
        ax.scatter(
            values,
            y + rng.uniform(-0.17, 0.17, values.size),
            s=12,
            color=VARIANT_COLORS[variant],
            alpha=0.35,
            linewidth=0,
        )
        p50, p95 = np.median(values), np.quantile(values, 0.95)
        ax.plot([p95, p95], [y - 0.3, y + 0.3], color=INK, linewidth=2, solid_capstyle="butt")
        ax.scatter([p50], [y], s=80, color=VARIANT_COLORS[variant], edgecolor=INK, linewidth=1.5, zorder=3)
        ax.text(
            1.0,
            y - 0.36,
            f"p50 {p50:.2f}s · p95 {p95:.2f}s",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=INK_2,
            fontsize=8.5,
        )
    ax.set_yticks(range(len(present)), [VARIANT_LABELS[v] for v in present])
    ax.set_ylim(len(present) - 0.5, -0.75)
    ax.set_xlim(left=0)
    ax.set_xlabel("Seconds (retrieval + prefill + decode) · dot p50, bar p95")

    fig.canvas.draw()
    markers = [
        mpl.transforms.Bbox.from_bounds(*(scatter.transData.transform((x, y)) - 10), 20, 20)
        for _, x, y in points
    ]
    clusters: list[list] = []
    for variant, x, y in points:
        for cluster in clusters:
            if math.dist((x, y), (cluster[1], cluster[2])) < 0.04:
                cluster[0].append(VARIANT_LABELS[variant])
                break
        else:
            clusters.append([[VARIANT_LABELS[variant]], x, y])
    place_labels(
        scatter,
        [("\n".join(n), x, y, {"color": INK, "fontsize": 9.5}) for n, x, y in clusters]
        + [
            ("ideal", 1, 1, {"color": MUTED, "fontsize": 8.5}),
            ("always abstains", 0, 1, {"color": MUTED, "fontsize": 8.5}),
            ("always answers", 1, 0, {"color": MUTED, "fontsize": 8.5}),
            ("no discrimination", 0.5, 0.5, {"color": MUTED, "fontsize": 8.5}),
        ],
        obstacles=markers,
    )
    footnote(
        fig,
        "Abstention is relative to the original SQuAD passage. Intervals resample questions; "
        "they do not capture training-seed variance.",
    )
    return fig


def plot_paired_contrasts(contrasts: list[dict], mode: str):
    metrics = list(dict.fromkeys(c["metric"] for c in contrasts))
    labels = list(dict.fromkeys(c["contrast"] for c in contrasts))
    fig = plt.figure(figsize=(15, 5.6))
    figure_header(
        fig,
        f"Paired comparison · {mode}",
        "What each component changes, question by question",
        "Mean paired difference (treated − control) with 95% bootstrap interval over questions",
    )
    grid = fig.add_gridspec(
        1, len(metrics), left=0.155, right=0.975, top=_top(fig, 1.95), bottom=0.25, wspace=0.12
    )
    spans = [abs(v) for c in contrasts if c["mean"] is not None for v in c["ci95"]]
    limit = max(max(spans, default=0) * 1.2, 0.05)
    for column, metric in enumerate(metrics):
        ax = fig.add_subplot(grid[0, column])
        clean_axis(ax, grid="x")
        rows = [c for c in contrasts if c["metric"] == metric]
        ax.axvline(0, color=BASELINE, linewidth=1)
        for c in rows:
            y = labels.index(c["contrast"])
            if c["mean"] is None:
                continue
            low, high = c["ci95"]
            color = GOOD if low > 0 else CRITICAL if high < 0 else MUTED
            ax.plot([low, high], [y, y], color=color, linewidth=2)
            ax.scatter([c["mean"]], [y], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            edge = c["mean"] / limit
            ax.text(
                c["mean"],
                y - 0.22,
                f"{c['mean']:+.2f}  [{low:+.2f}, {high:+.2f}]",
                ha="right" if edge > 0.55 else "left" if edge < -0.55 else "center",
                va="bottom",
                color=INK_2,
                fontsize=9,
            )
        ax.set_xlim(-limit, limit)
        ax.set_ylim(len(labels) - 0.5, -0.7)
        ax.set_yticks(range(len(labels)), labels if column == 0 else [])
        ax.set_title(f"{metric}  (n = {rows[0]['n'] if rows else 0})")
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{v:+.2f}"))
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="-", color=GOOD, label="▲ interval above 0"),
            Line2D([], [], marker="o", linestyle="-", color=CRITICAL, label="▼ interval below 0"),
            Line2D([], [], marker="o", linestyle="-", color=MUTED, label="interval spans 0"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.04, 0.055),
        ncols=3,
    )
    footnote(
        fig,
        "Each variant uses its own validation-calibrated threshold. One trained adapter; "
        "repeat training across seeds before claiming an effect.",
    )
    return fig


def plot_quantization(fp16: dict, q4: dict):
    """Dumbbells: each metric for fp16 (PyTorch) vs 4-bit Q4_K_M (Ollama), both RAG variants."""
    rows = [
        ("token_f1", "Token F1 · all"),
        ("answerable_f1", "Token F1 · answerable"),
        ("unanswerable_abstain_rate", "Abstention · unanswerable"),
        ("abstention_auroc", "Abstention AUROC"),
    ]
    variants = [v for v in ("base_rag", "adapter_rag") if v in fp16["variants"] and v in q4["variants"]]
    backends = [("PyTorch fp16", fp16, "o"), ("Ollama Q4_K_M", q4, "s")]
    fig = plt.figure(figsize=(14, 5.4))
    hero = variants[-1]
    delta = q4["variants"][hero]["test"]["token_f1"] - fp16["variants"][hero]["test"]["token_f1"]
    figure_header(
        fig,
        "Deployment · quantisation",
        "What 4-bit quantisation costs",
        f"{VARIANT_LABELS[hero]} token F1 changes by {delta:+.3f} at Q4_K_M · same prompts, retrieval "
        "and validation-calibrated thresholds · test split",
    )
    grid = fig.add_gridspec(
        1, 2, left=0.16, right=0.97, top=_top(fig, 1.75), bottom=0.16, wspace=0.35, width_ratios=[1.6, 1]
    )
    ax = fig.add_subplot(grid[0, 0])
    clean_axis(ax, grid="x")
    ax.set_title("Quality")
    offsets = {variants[0]: -0.14, variants[-1]: 0.14} if len(variants) > 1 else {variants[0]: 0.0}
    for row, (key, _label) in enumerate(rows):
        for variant in variants:
            y = row + offsets[variant]
            values = [src["variants"][variant]["test"][key] for _, src, _ in backends]
            ax.plot(values, [y, y], color=VARIANT_COLORS[variant], linewidth=2, alpha=0.6)
            for (_name, _src, marker), value in zip(backends, values):
                ax.scatter(
                    [value],
                    [y],
                    s=64,
                    marker=marker,
                    color=VARIANT_COLORS[variant],
                    edgecolor=SURFACE,
                    linewidth=2,
                    zorder=3,
                )
    for index in range(1, len(rows)):
        ax.axhline(index - 0.5, color=GRID, linewidth=0.8)
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.legend(
        handles=[
            Line2D([], [], color=VARIANT_COLORS[v], linewidth=2, label=VARIANT_LABELS[v]) for v in variants
        ]
        + [Line2D([], [], marker=m, linestyle="", color=INK_2, label=n) for n, _s, m in backends],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncols=4,
    )
    ax = fig.add_subplot(grid[0, 1])
    clean_axis(ax, grid="x")
    ax.set_title("Median seconds per question")
    labels, values, colors = [], [], []
    for variant in variants:
        for name, src, _m in backends:
            labels.append(f"{VARIANT_LABELS[variant]}\n{name}")
            values.append(src["variants"][variant]["test"]["latency_p50_s"])
            colors.append(VARIANT_COLORS[variant])
    ax.barh(range(len(values)), values, color=colors, height=0.5)
    for y, value in enumerate(values):
        ax.text(value, y, f"  {value:.2f}s", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.set_xlim(0, max(values) * 1.35)
    footnote(
        fig,
        "PyTorch time is amortised over batches of 16 on MPS; Ollama answers one question at a time "
        "(two requests: decision + answer) on Metal.",
    )
    return fig


def render_all(run_dir: Path, mode: str) -> list[Path]:
    """Draw every figure whose inputs exist; returns the written paths."""

    def read(name):
        path = run_dir / name
        return json.loads(path.read_text()) if path.exists() else None

    out = run_dir / "figures"
    written = []
    if (audit := read("training-data-audit.json")) is not None:
        written.append(save_figure(plot_evidence_audit(audit), out / "training-data-audit.png"))
    if (bench := read("retrieval-benchmark.json")) is not None:
        written.append(save_figure(plot_retrieval_benchmark(bench), out / "retrieval-benchmark.png"))
    history, training = read("training-log.json"), read("training.json")
    if history and training:
        written.append(
            save_figure(plot_training_dashboard(history, training, mode), out / "training-dashboard.png")
        )
    summary = read("metrics.json")
    val, test = read("predictions-validation.json"), read("predictions-test.json")
    if summary and val and test:
        by_val: dict[str, list] = collections.defaultdict(list)
        by_test: dict[str, list] = collections.defaultdict(list)
        for r in val:
            by_val[r["variant"]].append(r)
        for r in test:
            by_test[r["variant"]].append(r)
        outcomes = {v: question_outcomes(by_test[v], s["threshold"]) for v, s in summary["variants"].items()}
        written.append(save_figure(plot_calibration(by_val, by_test, summary), out / "calibration.png"))
        written.append(
            save_figure(plot_results_overview(by_test, outcomes, summary, mode), out / "results-overview.png")
        )
        written.append(
            save_figure(plot_paired_contrasts(summary["contrasts"], mode), out / "paired-contrasts.png")
        )
        if (quantised := read("ollama-q4_k_m/metrics.json")) is not None:
            written.append(save_figure(plot_quantization(summary, quantised), out / "quantization.png"))
    return written
