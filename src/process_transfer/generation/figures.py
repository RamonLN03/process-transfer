"""Basic figures of a data set, drawn from its export and from nothing else.

They show what a model would be given: readings and known inputs. No exact state, no
error and nothing of the truth is read, so these figures can be shown to anyone who may
see the data. Colours follow the figures of M0: source blue, target orange, near-black
text, on an off-white surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from process_transfer.data.export import open_export_directory  # noqa: E402

INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PLANT_COLOR = {"source": "#2a78d6", "target": "#eb6834"}
OTHER_PLANT = "#52514e"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_readings(export_directory: Path, destination: Path) -> list[Path]:
    """One figure per measured variable: a row of panels per run definition, a column per
    plant, the readings as dots against process time in minutes."""
    export = open_export_directory(export_directory)  # verified before anything is drawn
    manifest = export.manifest
    runs = manifest["runs"]
    plants = sorted(manifest["plants"])
    definitions = sorted({run["run_id"].split(".", 1)[1] for run in runs})
    by_key = {(run["plant_id"], run["run_id"].split(".", 1)[1]): run for run in runs}
    measured = [
        channel
        for channel in manifest["plants"][plants[0]]["channels"]
        if channel["channel_kind"] == "measured"
    ]
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    written = []
    for channel in measured:
        name, unit = channel["column"], channel["unit"]
        fig, axes = plt.subplots(
            len(definitions),
            len(plants),
            figsize=(5.5 * len(plants), 2.3 * len(definitions) + 0.8),
            sharex=True,
            squeeze=False,
            facecolor=SURFACE,
        )
        for row, definition in enumerate(definitions):
            for column, plant in enumerate(plants):
                ax = axes[row, column]
                _style(ax)
                run = by_key.get((plant, definition))
                if run is None:
                    ax.set_axis_off()
                    continue
                table = export.table(run["run_id"]).select(["time_s", name])
                ax.plot(
                    table["time_s"].to_numpy() / 60.0,
                    table[name].to_numpy(),
                    linestyle="none",
                    marker="o",
                    markersize=1.6,
                    color=PLANT_COLOR.get(plant, OTHER_PLANT),
                )
                ax.set_title(run["run_id"], fontsize=9, color=INK, loc="left")
                if column == 0:
                    ax.set_ylabel(f"{name}, {unit}", fontsize=9, color=INK_SECONDARY)
                if row == len(definitions) - 1:
                    ax.set_xlabel("process time, min", fontsize=9, color=INK_SECONDARY)
        fig.suptitle(
            f"{manifest['dataset_id']}: readings of {name} every "
            f"{channel['sampling_period_s']:g} s, noise included (sigma = {channel['noise_std']:g} "
            f"{unit}); observations only",
            fontsize=11,
            color=INK,
            x=0.01,
            ha="left",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = destination / f"readings_{name}.png"
        fig.savefig(path, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)
    return written
