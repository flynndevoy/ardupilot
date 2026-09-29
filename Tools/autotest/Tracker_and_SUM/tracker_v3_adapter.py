from datetime import datetime
import importlib
import math
from pathlib import Path
import sys
import json
import logging
import os

from copy import copy

from .ownship_camera_classes import Ownship as SyntheticOwnship


class TrackerV3Adapter:
    def __init__(self, ownship_id="OS", script_dir="."):
        self._debug_log_path = os.environ.get("ARDUPLANE_TRACKER_V3_LOG")
        self._debug_logger = None

        if self._debug_log_path:
            self._debug_logger = logging.getLogger(
                f"tracker_v3_adapter.{ownship_id}.{id(self)}"
            )
            self._debug_logger.setLevel(logging.DEBUG)
            self._debug_logger.propagate = False

            handler = logging.FileHandler(
                self._debug_log_path,
                mode="a",
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._debug_logger.addHandler(handler)

            self._debug_log(
                "adapter_init",
                ownship_id=ownship_id,
                script_dir=str(script_dir),
                log_path=self._debug_log_path,
            )

        source_root = Path(__file__).resolve().parents[4] / "tracker-source-v3"
        if not source_root.is_dir():
            raise FileNotFoundError(f"Tracker v3 source not found: {source_root}")
        source_root_str = str(source_root)
        if source_root_str not in sys.path:
            sys.path.insert(0, source_root_str)

        tracker_package = importlib.import_module("tracker")
        self._tracker = tracker_package.Tracker(
            ownship_id=ownship_id,
            script_dir=script_dir,
        )
        self._tracker.eo_tracker.association.debug_sink = (
            lambda record: self._debug_log("v3_association", **record)
        )
        self._tracker.track_state_manager.debug_sink = (
            lambda record: self._debug_log("v3_range_update", **record)
        )        
        self._tracker.track_state_manager.ukf_config.range_bias_scale = 0.0
        self.plot = False
        self.plot_save_path = None
        self._uncertainty_plotter = None
        models = importlib.import_module("tracker.models.detections")
        self._detection_summary_type = models.DetectionSummary
        states = importlib.import_module("tracker.models.states")
        self._adsb_state_type = states.ADSBIntruderState

        self._camera_names = [
            camera["name"] for camera in self._tracker.camera_cfg["cameras"]
        ]
        self.ownship = SyntheticOwnship(ownship_id)
        self.ownship.mount_cameras_from_config_file(
            camera_configs=self._tracker.camera_cfg,
            script_dir=script_dir,
        )

    @property
    def iso_decomp_guidance_hist(self):
        return self._tracker.iso_decomp_guidance_hist

    # def process_ownship_state(self, position, attitude):
    #     # the simulator/legacy tracker and tracker v3 might use opposite roll signs, so we flip the roll angle here to match the legacy tracker
    #     if isinstance(attitude, dict):
    #         v3_attitude = dict(attitude)
    #         v3_attitude["roll"] = -v3_attitude["roll"]
    #     else:
    #         v3_attitude = copy(attitude)
    #         v3_attitude.roll = -v3_attitude.roll

    #     return self._tracker.process_ownship_state(position, v3_attitude)

    def process_ownship_state(self, position, attitude):
        # Legacy and tracker v3 use opposite roll signs.
        # Copy first so the simulator's attitude object is never mutated.
        if isinstance(attitude, dict):
            v3_attitude = dict(attitude)

            roll_key = next(
                (
                    key
                    for key in ("roll", "roll_deg", "roll_rad", "phi")
                    if key in v3_attitude
                ),
                None,
            )
            if roll_key is None:
                raise KeyError(
                    "Could not find roll in simulator attitude. "
                    f"Available keys: {list(v3_attitude.keys())}"
                )

            v3_attitude[roll_key] = -v3_attitude[roll_key]
            simulator_roll = attitude[roll_key]
            v3_roll = v3_attitude[roll_key]
        else:
            v3_attitude = copy(attitude)
            v3_attitude.roll = -v3_attitude.roll
            simulator_roll = attitude.roll
            v3_roll = v3_attitude.roll

        self._debug_log(
            "v3_attitude_transform",
            simulator_roll_deg=simulator_roll,
            v3_roll_deg=v3_roll,
        )

        return self._tracker.process_ownship_state(position, v3_attitude)

    def save_tracker_plot(self, path=None, dpi=150):
        return None

    @staticmethod
    def _field(item, name, fallback=None):
        if isinstance(item, dict):
            return item.get(name, fallback)
        return getattr(item, name, fallback)

    @staticmethod
    def _debug_value(value):
        """Convert tracker objects into compact JSON-safe diagnostic data."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {
                str(key): TrackerV3Adapter._debug_value(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                TrackerV3Adapter._debug_value(item)
                for item in value
            ]

        data = {}
        for name in (
            "id",
            "track_id",
            "intruder_id",
            "callsign",
            "lat",
            "lon",
            "alt",
            "north_m",
            "east_m",
            "vertical_m",
            "north_speed",
            "east_speed",
            "vertical_speed",
            "ground_speed",
            "trk",
            "time",
            "timestamp",
            "age",
            "confidence",
        ):
            if hasattr(value, name):
                try:
                    data[name] = TrackerV3Adapter._debug_value(
                        getattr(value, name)
                    )
                except Exception as exc:
                    data[name] = f"<error reading {name}: {exc}>"

        if data:
            data["_type"] = type(value).__name__
            return data

        return {
            "_type": type(value).__name__,
            "_repr": repr(value),
        }

    def _debug_log(self, event, **fields):
        if self._debug_logger is None:
            return

        record = {
            "event": event,
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
        }
        record.update({
            key: self._debug_value(value)
            for key, value in fields.items()
        })

        try:
            self._debug_logger.debug(
                json.dumps(record, default=str, allow_nan=True)
            )
            for handler in self._debug_logger.handlers:
                handler.flush()
        except Exception:
            # Diagnostics must never break the tracker.
            pass

    def _adapt_detections(self, summaries, ownship_time):
        self._debug_log(
            "detections_received",
            ownship_time=ownship_time,
            summary_count=len(summaries or []),
            summaries=summaries,
        )

        if summaries is None:
            return None

        adapted = []
        for summary in summaries:
            frame_ids = summary.get("frame_id", [])
            frame_detections = summary.get("detections", [])
            detections = []
            range_corrections = []
            for frame_index, items in enumerate(frame_detections):
                frame_id = frame_ids[frame_index] if frame_index < len(frame_ids) else ""
                try:
                    detection_time = float(frame_id.rsplit("_", 1)[-1])
                except (ValueError, IndexError):
                    detection_time = ownship_time

                for item in items:
                    camera_id = self._field(item, "cam_id", frame_index)
                    if isinstance(camera_id, str) and camera_id in self._camera_names:
                        camera_name = camera_id
                        camera_index = self._camera_names.index(camera_id)
                    else:
                        camera_index = int(camera_id)
                        camera_name = self._camera_names[camera_index]

                    dx = self._field(item, "dx", 0.0)
                    dy = self._field(item, "dy", 0.0)

                    raw_range = float(self._field(item, "distance", 0.0))
                    camera = self.ownship.cameras[camera_index]["camera"]

                    scaled_range = camera.calibrate_ranges(raw_range, pre_scale_factor=1.5)
                    legacy_bias, _, _ = camera.get_detected_uncertainty_params(scaled_range)
                    corrected_range = scaled_range - legacy_bias

                    detections.append({
                        "camera_name": camera_name,
                        "camera_idx": camera_index,
                        "acquisition_timestamp": detection_time,
                        "global_cx": self._field(item, "abs_cx", self._field(item, "cx", 0.0)),
                        "global_cy": self._field(item, "abs_cy", self._field(item, "cy", 0.0)),
                        "width": self._field(item, "w", 0.0),
                        "height": self._field(item, "h", 0.0),
                        "distance_m": corrected_range,
                        "confidence": self._field(item, "confidence", 0.0),
                        "above_horizon": self._field(item, "above_horizon", 0.0),
                        "item_tracking_x": dx if math.isfinite(dx) else 0.0,
                        "item_tracking_y": dy if math.isfinite(dy) else 0.0,
                    })
                    range_corrections.append({
                        "frame_id": frame_id,
                        "detection_time": detection_time,
                        "camera_name": camera_name,
                        "pixel": [
                            self._field(item, "abs_cx", self._field(item, "cx", 0.0)),
                            self._field(item, "abs_cy", self._field(item, "cy", 0.0)),
                        ],
                        "raw_range": raw_range,
                        "scaled_range": scaled_range,
                        "legacy_bias": legacy_bias,
                        "corrected_range": corrected_range,
                    })

            self._debug_log(
                "detection_summary_adapted",
                ownship_time=ownship_time,
                frame_ids=frame_ids,
                detection_count=len(detections),
                detections=detections,
                range_corrections=range_corrections,
            )

            adapted.append(self._detection_summary_type(
                detections=detections,
                frame_id=list(frame_ids),
                camera_names=tuple(self._camera_names),
                timestamp=datetime.fromtimestamp(ownship_time),
            ))

        return adapted

    def _adapt_adsb(self, intruders):
        self._debug_log(
            "adsb_received",
            intruder_count=len(intruders or []),
            intruders=intruders or [],
        )

        adapted = []

        for item in intruders or []:
            if isinstance(item, self._adsb_state_type):
                adapted.append(item)
                continue

            adapted.append(self._adsb_state_type(
                lat=item["lat"],
                lon=item["lon"],
                alt=item["alt"],
                trk=item["trk"],
                north_m=item["north_m"],
                east_m=item["east_m"],
                ground_speed=item["ground_speed"],
                vertical_speed=item["vertical_speed"],
                type="ADSB",
                callsign=str(item["callsign"]),
                id=str(item["id"]),
                time=item["time"],
                NIC=item.get("NIC", 6),
                NACp=item.get("NACp", 6),
                SIL=item.get("SIL", 0),
                SDA=item.get("SDA", 0),
            ))

        self._debug_log(
            "adsb_adapted",
            intruder_count=len(adapted),
            intruders=adapted,
        )

        return adapted

    def update_tracks_v3(
        self,
        ownship_state,
        detection_summaries=None,
        adsb_intruders=None,
    ):
        ownship_time = self._field(ownship_state, "time")

        self._debug_log(
            "update_tracks_input",
            ownship_time=ownship_time,
            ownship_state=ownship_state,
            detection_summary_count=len(detection_summaries or []),
            adsb_intruder_count=len(adsb_intruders or []),
        )

        adapted_detections = self._adapt_detections(
            detection_summaries,
            ownship_time,
        )
        adapted_adsb = self._adapt_adsb(adsb_intruders)

        result = self._tracker.update_tracks(
            ownship_state=ownship_state,
            detection_summaries=adapted_detections,
            adsb_intruders=adapted_adsb,
        )

        output_tracks = result.output_tracks
        latest_track_histories = result.latest_track_histories
        terminated_track_ids = result.terminated_track_ids

        self._debug_log(
            "update_tracks_output",
            ownship_time=ownship_time,
            output_track_count=len(output_tracks or []),
            output_tracks=output_tracks or [],
            latest_track_history_count=len(
                latest_track_histories or []
            ),
            latest_track_histories=latest_track_histories or [],
            terminated_track_ids=terminated_track_ids or [],
        )

        return (
            output_tracks,
            latest_track_histories,
            terminated_track_ids,
        )

    def create_daidalus_input(self, ownship_state, intruder_states):
        payload = self._tracker.create_daidalus_input(
            ownship_state, intruder_states
        ).to_dict()
        if self.plot and self.plot_save_path:
            try:
                if self._uncertainty_plotter is None:
                    from .tracker_v3_uncertainty_plot import TrackerV3UncertaintyPlotter
                    self._uncertainty_plotter = TrackerV3UncertaintyPlotter()
                saved = self._uncertainty_plotter.save_if_new_second(
                    ownship_state, intruder_states, payload, self.plot_save_path
                )
                if saved is not None:
                    self._debug_log("v3_uncertainty_plot_saved", path=str(saved))
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to save tracker v3 uncertainty plot"
                )
        return payload