"""Opt-in, per-second plots of the uncertainty sent from tracker v3 to DAIDALUS."""

import csv
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np


LOG = logging.getLogger(__name__)


def _get(item, key):
    return item.get(key) if isinstance(item, dict) else getattr(item, key)


def _ellipse(center, std_x, std_y, signed_cross_std, color, label):
    """Draw the 2-sigma NE covariance used to construct DAIDALUS input."""
    cross_cov = math.copysign(signed_cross_std ** 2, signed_cross_std)
    cov = np.array([[std_x ** 2, cross_cov], [cross_cov, std_y ** 2]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    axis = eigvecs[:, order[0]]
    angle = math.degrees(math.atan2(axis[1], axis[0]))
    return Ellipse(
        center, 4 * math.sqrt(eigvals[0]), 4 * math.sqrt(eigvals[1]),
        angle=angle, fill=False, color=color, linewidth=2,
        linestyle="--", label=label,
    )


def _constituents_for_track(track_id, daidalus_intruders):
    prefix = f"{track_id}_const_"
    return [item for item in daidalus_intruders
            if str(item.get("i", "")).startswith(prefix)]


class TrackerV3UncertaintyPlotter:
    """Plot the parent 2-sigma ellipses and actual DAIDALUS constituents."""

    def __init__(self):
        self.last_saved_second = None
        self.position_history = defaultdict(list)

    def save_if_new_second(self, ownship, tracks, payload, output_dir):
        second = math.floor(float(_get(ownship, "time")))
        if second < 0 or second == self.last_saved_second or not tracks:
            return None

        daidalus_intruders = payload.get("d", [])
        if not daidalus_intruders:
            return None
        if any(item.get("cf") != "NEU" for item in daidalus_intruders):
            LOG.warning("Skipping v3 uncertainty plot: DAIDALUS frame is not NEU")
            return None

        start = time.perf_counter()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
        overview = fig.add_subplot(grid[:, 0])
        position = fig.add_subplot(grid[0, 1])
        velocity = fig.add_subplot(grid[1, 1])
        own_ne = np.array([float(_get(ownship, "north_m")),
                           float(_get(ownship, "east_m"))])
        overview.plot(*own_ne, "ko", label="Ownship")
        n_position_circles = 0
        n_velocity_circles = 0

        for index, (track_id, track) in enumerate(tracks.items()):
            if track is None:
                continue
            color = f"C{index % 10}"
            center = np.asarray(_get(track, "abs_cartesian_x_y_z_smoothed"),
                                dtype=float)[:2]
            velocity_center = np.asarray(
                _get(track, "velocity_x_y_z_smoothed"), dtype=float)[:2]
            if not (np.isfinite(center).all() and np.isfinite(velocity_center).all()):
                continue

            history = self.position_history[track_id]
            if not history or not np.array_equal(history[-1], center):
                history.append(center.copy())
            path = np.asarray(history)
            overview.plot(path[:, 0], path[:, 1], color=color,
                          label=f"Track {track_id}")
            overview.plot([own_ne[0], center[0]], [own_ne[1], center[1]],
                          color=color, linestyle=":", alpha=0.55)
            overview.plot(*center, marker="o", color=color)
            position.plot(*center, marker="o", color=color,
                          label=f"Mean {track_id}")
            velocity.plot(*velocity_center, marker="o", color=color,
                          label=f"Mean {track_id}")

            std = np.asarray(_get(track, "uncertainty_cartesian_x_y_z_smoothed"),
                             dtype=float)
            std_v = np.asarray(_get(track, "uncertainty_velocity_x_y_z_smoothed"),
                               dtype=float)
            position.add_patch(_ellipse(
                center, std[0], std[1],
                float(_get(track, "uncertainty_cartesian_yx_std_smoothed")),
                color, "Parent position 2 sigma" if index == 0 else "_nolegend_",
            ))
            velocity.add_patch(_ellipse(
                velocity_center, std_v[0], std_v[1],
                float(_get(track, "uncertainty_velocity_yx_std_smoothed")),
                color, "Parent velocity 2 sigma" if index == 0 else "_nolegend_",
            ))

            # Velocity variants repeat each position sphere, and vice versa.
            seen_position = set()
            seen_velocity = set()
            for item in _constituents_for_track(track_id, daidalus_intruders):
                pos_key = (float(item["la_no"]), float(item["lo_ea"]),
                           float(item["d_x"]))
                vel_heading = math.radians(float(item["t"]))
                vel_key = (float(item["gs"]) * math.cos(vel_heading),
                           float(item["gs"]) * math.sin(vel_heading),
                           float(item["d_vx"]))
                if pos_key not in seen_position:
                    position.add_patch(Circle(
                        pos_key[:2], 2 * pos_key[2], fill=False,
                        edgecolor="teal", alpha=0.7,
                        label=("DAIDALUS position circles" if
                               n_position_circles == 0 else "_nolegend_"),
                    ))
                    seen_position.add(pos_key)
                    n_position_circles += 1
                if vel_key not in seen_velocity:
                    velocity.add_patch(Circle(
                        vel_key[:2], 2 * vel_key[2], fill=False,
                        edgecolor="darkorange", alpha=0.7,
                        label=("DAIDALUS velocity circles" if
                               n_velocity_circles == 0 else "_nolegend_"),
                    ))
                    seen_velocity.add(vel_key)
                    n_velocity_circles += 1

        for ax, title, xlabel, ylabel in (
            (overview, "Track and ownship overview", "North (m)", "East (m)"),
            (position, "Position uncertainty (2 sigma)", "North (m)", "East (m)"),
            (velocity, "Velocity uncertainty (2 sigma)", "North (m/s)", "East (m/s)"),
        ):
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=7)

        # Autoscale the zoomed panels from the current means and patches, not
        # the full track history; otherwise the uncertainty becomes invisible.
        for ax in (position, velocity):
            ax.relim()
            ax.autoscale_view()
            ax.margins(0.20)
        fig.suptitle(f"Tracker v3 uncertainty at t={second} s")
        path = output_dir / f"tracker_uncertainty_t{second:04d}s.png"
        try:
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)

        elapsed_ms = 1000 * (time.perf_counter() - start)
        timing_path = output_dir / "plot_timing.csv"
        new_file = not timing_path.exists()
        with timing_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if new_file:
                writer.writerow(("second", "track_count", "position_circles",
                                 "velocity_circles", "render_ms"))
            writer.writerow((second, len(tracks), n_position_circles,
                             n_velocity_circles, round(elapsed_ms, 2)))
        self.last_saved_second = second
        return path
