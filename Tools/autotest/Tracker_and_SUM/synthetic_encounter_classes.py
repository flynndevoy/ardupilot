import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import matplotlib.colors as mcolors
# from matplotlib.lines import Line2D
from typing import *
# from matplotlib import cm
# from matplotlib.gridspec import GridSpec
from Tracker_and_SUM.prediction_structure_v2 import *
import yaml
import os
import navpy
from copy import deepcopy

#### INTRUDER CLASS ####    

class Intruder:
    def __init__(self, idx, position_xyz: Tuple[float, float, float],
                 orientation_rpy_deg: Tuple[float, float, float] = None):
        self.position = np.array(position_xyz, dtype=float)
        if orientation_rpy_deg is not None:
            self.orientation_rpy_deg = np.array(orientation_rpy_deg, dtype=float)
        else:
            self.orientation_rpy_deg = None
        self.id = idx

    def __repr__(self):
        if self.orientation_rpy_deg is not None:
            orient = f", orientation=({self.orientation_rpy_deg[0]:.1f}, {self.orientation_rpy_deg[1]:.1f}, {self.orientation_rpy_deg[2]:.1f})"
        else:
            orient = ""
        return f"Intruder(id={self.id}, position=({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}){orient})"
    
    def update_state(self, position_xyz: Tuple[float, float, float],
                 orientation_rpy_deg: Tuple[float, float, float] = None):
        self.position = np.array(position_xyz, dtype=float)
        self.orientation_rpy_deg = np.array(orientation_rpy_deg, dtype=float)

### ENCOUNTER CLASS ####    

class Encounter:
    def __init__(self, ownship: 'Ownship', frame = None, intruders: List['Intruder'] = [], plot_cartesian_history = False, script_dir=".", config_file="cameras_config.yaml"):
        self.frame = frame
        self.ownship = ownship
        self.ownship_ground_truth_history = Deque()
        self.ownship_ground_truth_history_sec = 5.0

        self.intruders = intruders if intruders else []
        self.camera_views = {idx: (x, []) for idx, x in enumerate(ownship.cameras)}
        self.intruder_views = {intr.id: {} for intr in intruders}
        self.encounter_history = {}
        self.plot_cartesian_history = plot_cartesian_history

        # === Load YAML Config ===
        config_path = os.path.join(script_dir, config_file)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.sum_on = config["configs"].get("generate_synthetic_uncertainty", False)

        if frame is not None:
            self.create_camera_intruder_views()
            self.update_encounter_history()
            if self.plot_cartesian_history:
                self.cartesian_history = {}
                self.update_cartesian_history()
            if self.sum_on:
                self.add_uncertainty_to_intruders()

    def update_encounter(self, frame, ownship: 'Ownship', intruders: List['Intruder'] = []):
        if frame == self.frame:
            raise AssertionError(f"Encounter already updated for frame {frame}")
        if frame is None:
            raise AssertionError(f"Frame must be specified to update encounter!")
        if ownship.id != self.ownship.id:
            raise AssertionError(f"Cannot change ownship ID mid-encounter! Start new encounter for new ownship!")
        
        self.frame = frame
        self.ownship = ownship
        self.intruders = intruders
        self.camera_views = {idx: (x, []) for idx, x in enumerate(ownship.cameras)}
        self.intruder_views = {intr.id: {} for intr in intruders}

        self.create_camera_intruder_views()
        self.update_encounter_history()
        if self.plot_cartesian_history:
                self.update_cartesian_history()
        if self.sum_on:
            self.add_uncertainty_to_intruders()

    def create_camera_intruder_views(self, original_amazon_range_estimate = True, detector_pre_scale_factor = 1.5):
        # Determine line of sight intruders
        for intr in self.intruders:
            for cam_idx, cam in enumerate(self.ownship.cameras):
                cam_obj = cam["camera"]
                pos_local = cam["position_xyz"]
                rpy_local = np.radians(cam["orientation_rpy_deg"])
        
                R_cam = self.ownship.intrinsic_neu_rpy_to_matrix(*rpy_local)
                R_own = self.ownship.intrinsic_neu_rpy_to_matrix(*np.radians(self.ownship_ground_truth_orientation_rpy_deg()))
                T_own = self.ownship_ground_truth_position()
        
                # Global camera position and direction
                cam_global = R_own @ pos_local + T_own
                R_total = R_own @ R_cam
        
                # Transform intruder position to camera local frame
                intr_vec_global = intr.position - cam_global
                intr_cam_frame = R_total.T @ intr_vec_global  # Rotate into camera frame
        
                x, y, z = intr_cam_frame
                if x <= 0:
                    continue  # Behind camera
        
                h_angle = np.degrees(np.arctan2(y, x))
                v_angle = np.degrees(np.arctan2(z, x))
        
                if (abs(h_angle) <= cam_obj.fov_x_deg / 2) and (abs(v_angle) <= cam_obj.fov_y_deg / 2):
                    # Calculate Pixel Located on Camera
                    fx = cam_obj.pixels_x / (2 * np.tan(np.deg2rad(cam_obj.fov_x_deg) / 2))
                    fy = cam_obj.pixels_y / (2 * np.tan(np.deg2rad(cam_obj.fov_y_deg) / 2))
                    cx = cam_obj.pixels_x / 2
                    cy = cam_obj.pixels_y / 2
                    
                    pixel_X = cx + fx * (y / x)  # positive y → right → higher pixel_x
                    pixel_Y = cy - fy * (z / x)  # positive z → up → lower pixel_y

                    # Calculate true horizontal range in global XY plane (ignoring Z/altitude)
                    dx = intr.position[0] - cam_global[0]
                    dy = intr.position[1] - cam_global[1]
                    horizontal_range = np.sqrt(dx**2 + dy**2)

                    if original_amazon_range_estimate:
                        horizontal_range = cam_obj.calibrate_ranges(horizontal_range, model_output_to_true_scale=False, empirical_post_scale_factor = detector_pre_scale_factor)

                    self.camera_views[cam_idx][1].append((intr.id, [pixel_X, pixel_Y, horizontal_range]))
                    self.intruder_views[intr.id].setdefault(self.frame, {})[cam_idx] = (pixel_X, pixel_Y, horizontal_range)

    def update_cartesian_history(self):
        if self.ownship.id not in self.cartesian_history:
            # Initialize ownship history
            self.cartesian_history[self.ownship.id] = {
                'frame': [],
                'position': [],
                'orientation': []
            }

        # Update ownship state history
        self.cartesian_history[self.ownship.id]['frame'].append(self.frame)
        self.cartesian_history[self.ownship.id]['position'].append(self.ownship_ground_truth_position())
        self.cartesian_history[self.ownship.id]['orientation'].append(self.ownship_ground_truth_orientation_rpy_deg())
        all_cart_ids = set(self.cartesian_history.keys()).union([intr.id for intr in self.intruders])
        all_cart_ids.discard(self.ownship.id)
        if not all_cart_ids:
            all_cart_ids = set()
        
        for intr_id in all_cart_ids:
            if intr_id not in self.cartesian_history:
                # Initialize new intruder history
                self.cartesian_history[intr_id] = {
                    'frame': [],
                    'position': [],
                    'orientation': [],
                }
      
            intr = next((intr for intr in self.intruders if intr.id == intr_id), None)
            if intr:
                self.cartesian_history[intr_id]['frame'].append(self.frame)
                self.cartesian_history[intr_id]['position'].append(intr.position)
                self.cartesian_history[intr_id]['orientation'].append(intr.orientation_rpy_deg)
    
    def _trim(self, lst, history_len):
        if len(lst) > history_len:
            del lst[:-history_len]   # keep only the newest history_len items

    def update_encounter_history(self, history_len = 15):
        if self.ownship.id not in self.encounter_history:
            # Initialize ownship history
            self.encounter_history[self.ownship.id] = {
                'frame': [],
                'position': [],
                'orientation': []
            }

        # Update ownship state history
        own_hist = self.encounter_history[self.ownship.id]
        own_hist['frame'].append(self.frame)
        self._trim(own_hist['frame'], history_len)

        own_hist['position'].append(self.ownship_ground_truth_position())
        self._trim(own_hist['position'], history_len)

        own_hist['orientation'].append(self.ownship_ground_truth_orientation_rpy_deg())
        self._trim(own_hist['orientation'], history_len)

        # --- Intruder histories ---
        all_intruder_ids = set(self.encounter_history.keys()).union(self.intruder_views.keys())
        all_intruder_ids.discard(self.ownship.id)

        if not all_intruder_ids:
            all_intruder_ids = set()
        
        for intr_id in all_intruder_ids:
            intr_views = self.intruder_views.get(intr_id, {}).get(self.frame, {})

            if intr_id not in self.encounter_history:
                # Initialize new intruder history
                encounter_frame_dict = {
                    'frame': [],
                    'pixel_x': [],
                    'pixel_y': [],
                    'range': [],
                    'error_px': [],
                    'error_py': [],
                    'error_range': [],
                    'd1_error_range': [],
                    'd2_error_range': [],
                    'confidence': [],
                    'horizon_state': []
                }
                self.encounter_history[intr_id] = {idx: deepcopy(encounter_frame_dict) for idx, _ in enumerate(self.ownship.cameras)}
            
            for cam_id, intr_hist in self.encounter_history[intr_id].items():

                if cam_id in intr_views:
                    px, py, rng = intr_views[cam_id]
                else:
                    px = py = rng = np.nan

                # Append Entries
                intr_hist['frame'].append(self.frame)
                intr_hist['pixel_x'].append(px)
                intr_hist['pixel_y'].append(py)
                intr_hist['range'].append(rng)
                intr_hist['error_px'].append(np.nan)
                intr_hist['error_py'].append(np.nan)
                intr_hist['error_range'].append(np.nan)
                intr_hist['d1_error_range'].append(np.nan)
                intr_hist['d2_error_range'].append(np.nan)
                intr_hist['confidence'].append(np.nan)
                intr_hist['horizon_state'].append(np.nan)

                # Trim History to Max Length
                self._trim(intr_hist['frame'], history_len)
                self._trim(intr_hist['pixel_x'], history_len)
                self._trim(intr_hist['pixel_y'], history_len)
                self._trim(intr_hist['range'], history_len)
                self._trim(intr_hist['error_px'], history_len)
                self._trim(intr_hist['error_py'], history_len)
                self._trim(intr_hist['error_range'], history_len)
                self._trim(intr_hist['d1_error_range'], history_len)
                self._trim(intr_hist['d2_error_range'], history_len)
                self._trim(intr_hist['confidence'], history_len)
                self._trim(intr_hist['horizon_state'], history_len)

    def add_uncertainty_to_intruders(self):
        for intr_idx, intr_hist_cams in self.encounter_history.items():
            if intr_idx == self.ownship.id:
                continue

            for cam_id, intr_hist in intr_hist_cams.items():
                if np.isnan(intr_hist['pixel_x'][-1]):
                    continue
                
                cam = self.ownship.cameras[cam_id]['camera']

                ##### PIXEL UNCERTAINTY #####
                # Sample Pixel Error
                pixel_x_error, pixel_y_error = cam.synthetic_uncertainty_models["pixel_uncertainty_model"](gt_dist=intr_hist['range'][-1])
                intr_hist['error_px'][-1] = pixel_x_error
                intr_hist['error_py'][-1] = pixel_y_error

                ##### RANGE UNCERTAINTY #####

                # Validate format of input data
                range_error_input_space_format = cam.synthetic_uncertainty_models["range_error_model_input_format"]
                
                expected_input_vector_data_types = ['history_first_deriv_error',
                                            'history_second_deriv_error',
                                            'history_error',
                                            'history_error_rel',
                                            'history_gt_dist']

                assert set(range_error_input_space_format.keys()) == set(expected_input_vector_data_types), \
                    "Mismatch between expected and ingested range uncertainty model inputs!"

                # Create input arrays for range error model
                error_history = intr_hist['error_range']
                d1_history = intr_hist['d1_error_range']
                d2_history = intr_hist['d2_error_range']
                gt_range_history = intr_hist['range']
                
                if len(gt_range_history) == 1:
                    error_sample_input_array = [[np.nan] + gt_range_history, error_history + [np.nan], d1_history + [np.nan], d2_history + [np.nan]]
                else:
                    if np.isnan(intr_hist['error_range'][-2]):
                        for cam_id_prev, intr_hist_prev in intr_hist_cams.items():
                            if np.isnan(intr_hist_prev['error_range'][-2]):
                                continue
                            if cam_id_prev == cam_id:
                                continue
                            
                            #range_error_this_init, _, _ = cam.synthetic_uncertainty_models["range_uncertainty_model"](input_history = [gt_range_history, error_history, d1_history, d2_history])
                            error_history[-max(range_error_input_space_format['history_error'])-2:-1] = [x for x in intr_hist_prev['error_range'].copy()[-max(range_error_input_space_format['history_error'])-1:]]
                            d1_history[-max(range_error_input_space_format['history_first_deriv_error'])-2:-1] = [x for x in intr_hist_prev['d1_error_range'].copy()[-max(range_error_input_space_format['history_first_deriv_error'])-1:]]
                            d2_history[-max(range_error_input_space_format['history_second_deriv_error'])-2:-1] = [x for x in intr_hist_prev['d2_error_range'].copy()[-max(range_error_input_space_format['history_second_deriv_error'])-1:]]
                            intr_hist['error_range'] =  error_history
                            intr_hist['d1_error_range'] =  d1_history
                            intr_hist['d2_error_range'] =  d2_history

                            break
                    error_sample_input_array = [gt_range_history, error_history, d1_history, d2_history]

                # Sample range uncertainty for this detection for this frame
                range_error_this, d1_range_error_this, d2_range_error_this = cam.synthetic_uncertainty_models["range_uncertainty_model"](input_history = error_sample_input_array)
                intr_hist['error_range'][-1] = range_error_this
                intr_hist['d1_error_range'][-1] = d1_range_error_this
                intr_hist['d2_error_range'][-1] = d2_range_error_this

                ##### CONFIDENCE / PDET MODEL #####

                # Determine if the intruder is above or below the horizon

                # Camera Positioning Specs
                pos_local = self.ownship.cameras[cam_id]["position_xyz"]
                rpy_local = np.radians(self.ownship.cameras[cam_id]["orientation_rpy_deg"])
        
                R_own = self.ownship.intrinsic_neu_rpy_to_matrix(*np.radians(self.ownship_ground_truth_orientation_rpy_deg()))
                T_own = self.ownship_ground_truth_position()

                R_cam = self.ownship.intrinsic_neu_rpy_to_matrix(*rpy_local)
                pos_global = R_own @ pos_local + T_own
                R_total = R_own @ R_cam

                above_horizon_state = np.isnan(self.pixels_to_ground(pos_global, R_total, cam, [[intr_hist['pixel_x'][-1], intr_hist['pixel_y'][-1]]])).any()

                reversed_confidence_history = intr_hist['confidence'][-2:0:-1]
                # Get confidence history
                if len(intr_hist['confidence']) > 1:
                    if np.isnan(intr_hist['confidence'][-2]):
                        for cam_id_prev, intr_hist_prev in intr_hist_cams.items():
                            if np.isnan(intr_hist_prev['confidence'][-2]):
                                continue
                            if cam_id_prev == cam_id:
                                continue
                            reversed_confidence_history = intr_hist_prev['confidence'].copy()[-2:0:-1]
                            break
                    error_sample_input_array = [gt_range_history, error_history, d1_history, d2_history]
                
                # Sample detection confidence
                pdet_this = cam.synthetic_uncertainty_models["pdet_uncertainty_model"](
                    above_horizon_state = above_horizon_state,
                    gt_dist = gt_range_history[-1],
                    lagged_confs = reversed_confidence_history
                    )[0]
                
                intr_hist['confidence'][-1] = pdet_this
                intr_hist['horizon_state'][-1] = 1.0 if above_horizon_state else -1.0

    def return_camera_views(self):
        return self.camera_views
    
    def detections_this_frame(self):
        detections = {cam_id: [] for cam_id in range(len(self.ownship.cameras))}
        
        for intr_id in self.intruder_views.keys():
            if intr_id == self.ownship.id:
                continue

            for cam_id, intr_hist in self.encounter_history[intr_id].items():
                # Ignore if no detection last frame for this camera
                if np.isnan(intr_hist['pixel_x'][-1]):
                    continue

                ########
                if not self.sum_on:
                    detection = DetectedItem(
                        cx=intr_hist['pixel_x'][-1], # TODO: I REMOVED THE OFFSET TERM?????
                        cy=intr_hist['pixel_y'][-1], # TODO: I REMOVED THE OFFSET TERM?????
                        abs_cx=intr_hist['pixel_x'][-1], # TODO: I REMOVED THE OFFSET TERM?????
                        abs_cy=intr_hist['pixel_y'][-1], # TODO: I REMOVED THE OFFSET TERM?????
                        w=100, # TODO
                        h=100, # TODO
                        distance=intr_hist['range'][-1],
                        above_horizon=1.0,
                        confidence=1.0,
                        primary_conf=1,
                        track_id=-1,
                        dx=intr_hist['pixel_x'][-1] - intr_hist['pixel_x'][-2] if len(intr_hist['pixel_x']) >= 2 and not np.isnan(intr_hist['pixel_x'][-2]) else np.nan,
                        dy=intr_hist['pixel_y'][-1] - intr_hist['pixel_y'][-2] if len(intr_hist['pixel_y']) >= 2 and not np.isnan(intr_hist['pixel_y'][-2]) else np.nan,
                        item_id='',
                        cls_idx = 0,
                        cam_id = cam_id
                    )

                    detections[cam_id].append(detection)
                else:
                    detection = DetectedItem(
                        cx=np.nansum([intr_hist['pixel_x'][-1], intr_hist['error_px'][-1]]), # TODO: I REMOVED THE OFFSET TERM?????
                        cy=np.nansum([intr_hist['pixel_y'][-1], intr_hist['error_py'][-1]]), # TODO: I REMOVED THE OFFSET TERM?????
                        abs_cx=np.nansum([intr_hist['pixel_x'][-1], intr_hist['error_px'][-1]]), # TODO: I REMOVED THE OFFSET TERM?????
                        abs_cy=np.nansum([intr_hist['pixel_y'][-1], intr_hist['error_py'][-1]]), # TODO: I REMOVED THE OFFSET TERM?????
                        w=100, # TODO
                        h=100, # TODO
                        distance=np.nansum([intr_hist['range'][-1], intr_hist['error_range'][-1]]),
                        above_horizon=intr_hist['horizon_state'][-1],
                        confidence=intr_hist['confidence'][-1],
                        primary_conf=intr_hist['confidence'][-1],
                        track_id=-1,
                        dx=intr_hist['pixel_x'][-1] - intr_hist['pixel_x'][-2] if len(intr_hist['pixel_x']) >= 2 and not np.isnan(intr_hist['pixel_x'][-2]) else np.nan,
                        dy=intr_hist['pixel_y'][-1] - intr_hist['pixel_y'][-2] if len(intr_hist['pixel_y']) >= 2 and not np.isnan(intr_hist['pixel_y'][-2]) else np.nan,
                        item_id='',
                        cls_idx = 0,
                        cam_id = cam_id
                    )
                    detections[cam_id].append(detection)

        return detections

    #### Helpers for accessing ownship ground truth for pixel generation ####
    def set_ground_truth_ownship_state(self, position_xyz: Tuple[float, float, float],
                  orientation_rpy_deg: Tuple[float, float, float],
                  ownship_time: float):
        
        # Ignore duplicate or superceded ownship telemetry
        if self.ownship_ground_truth_history:
            if ownship_time <= self.ownship_ground_truth_history[-1][0]:
                return
            
        pos = np.asarray(position_xyz, float)
        ori = np.asarray(orientation_rpy_deg, float)
        cutoff = ownship_time - self.ownship_ground_truth_history_sec

        while self.ownship_ground_truth_history and self.ownship_ground_truth_history[0][0] < cutoff:
            self.ownship_ground_truth_history.popleft()

        self.ownship_ground_truth_history.append((ownship_time, pos, ori))

        return
    
     # ---------------------------
    # Get the position or orientation of ownship at a given time
    # ---------------------------
    def ownship_ground_truth_position(self, ownship_time: Optional[float] = None) -> np.ndarray:
        return self._interp(ownship_time, idx=1)

    def ownship_ground_truth_orientation_rpy_deg(self, ownship_time: Optional[float] = None) -> np.ndarray:
        return self._interp(ownship_time, idx=2)

    # ==========================================================================================
    # Interpolation code for ownship position and orientation at given time based on its history
    # ==========================================================================================

    # ---------------------------
    # Angle helpers (degrees)
    # ---------------------------
    @staticmethod
    def _angle_diff_shortest(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
        """
        Minimal signed difference a1 - a0 in [-180, 180).
        Works elementwise for arrays/vectors.
        """
        return (a1 - a0 + 180.0) % 360.0 - 180.0

    @classmethod
    def _angle_lerp(cls, a0: np.ndarray, a1: np.ndarray, alpha: float) -> np.ndarray:
        """
        Interpolate angles (deg) along the shortest arc.
        Inputs can be scalars or same-shaped arrays (e.g., 3-vector RPY).
        Returns values wrapped to [0, 360).
        """
        a0 = np.mod(np.asarray(a0, dtype=float), 360.0)
        a1 = np.mod(np.asarray(a1, dtype=float), 360.0)
        diff = cls._angle_diff_shortest(a0, a1)
        return np.mod(a0 + alpha * diff, 360.0)

    @classmethod
    def _angle_extrap(cls, a_prev: np.ndarray, a_now: np.ndarray, alpha: float) -> np.ndarray:
        """
        Extrapolate angles using shortest-arc delta from a_prev -> a_now,
        then extend by factor alpha (e.g., alpha=(t - t1)/dt).
        """
        a_prev = np.mod(np.asarray(a_prev, dtype=float), 360.0)
        a_now  = np.mod(np.asarray(a_now,  dtype=float), 360.0)
        step = cls._angle_diff_shortest(a_prev, a_now)  # signed step in [-180,180)
        return np.mod(a_now + alpha * step, 360.0)

    # ---------------------------
    # Core interpolation helper
    # ---------------------------
    def _interp(self, t: Optional[float], idx: int, snap_threshold: float = 1e-3):
        """
        Interpolates position (idx=1) linearly, and orientation RPY in degrees (idx=2)
        using shortest-arc interpolation per axis in [0, 360).

        track_history entries are tuples like:
            (time, position_vec3, orientation_rpy_deg_vec3)
        """
        if not self.ownship_ground_truth_history:
            print("Track_history is empty; no state available. Returning default zeros")
            return np.zeros(3)

        # Choose whether we're angle-aware (orientation) or linear (position)
        angle_mode = (idx == 2)

        if t is None:
            return self.ownship_ground_truth_history[-1][idx].copy()
        
        # Snap if close to last time
        if abs(t - self.ownship_ground_truth_history[-1][0]) <= snap_threshold:
            return self.ownship_ground_truth_history[-1][idx].copy()

        # Before earliest
        if t < self.ownship_ground_truth_history[0][0]:
            print(f"[Ownship] Requested time {t:.6f} is before earliest history time {self.ownship_ground_truth_history[0][0]:.6f}. "
                f"Returning earliest sample.")
            return self.ownship_ground_truth_history[0][idx].copy()

        # After latest → extrapolate
        if t > self.ownship_ground_truth_history[-1][0]:
            if len(self.ownship_ground_truth_history) >= 2:
                t0, v0 = self.ownship_ground_truth_history[-2][0], self.ownship_ground_truth_history[-2][idx]
                t1, v1 = self.ownship_ground_truth_history[-1][0], self.ownship_ground_truth_history[-1][idx]
                dt = t1 - t0
                if dt > 0:
                    alpha = (t - t1) / dt
                    if angle_mode:
                        v = self._angle_extrap(v0, v1, alpha)
                    else:
                        v = v1 + alpha * (v1 - v0)
                    # print(f"[Ownship] Requested time {t:.6f} is after latest history time {t1:.6f}. "
                    #     f"Extrapolating {'angular' if angle_mode else 'linear'}ly (Δt={dt:.6f}, alpha={alpha:.6f}).")
                    return v
            print(f"[Ownship] Requested time {t:.6f} is after latest history time {self.ownship_ground_truth_history[-1][0]:.6f}, "
                f"but insufficient/degenerate history to extrapolate. Returning latest sample.")
            return self.ownship_ground_truth_history[-1][idx].copy()

        # Walk backwards from the latest until we find bounding samples
        later_t, later_v = self.ownship_ground_truth_history[-1][0], self.ownship_ground_truth_history[-1][idx]
        for earlier_t, pos, ori in reversed(list(self.ownship_ground_truth_history)[:-1]):
            earlier_v = pos if idx == 1 else ori
            if earlier_t <= t <= later_t:
                # Snap if close to either bound
                if abs(t - earlier_t) <= snap_threshold:
                    return earlier_v.copy()
                if abs(t - later_t) <= snap_threshold:
                    return later_v.copy()
                # Interpolate
                dt = later_t - earlier_t
                alpha = (t - earlier_t) / dt if dt > 0 else 0.0
                if angle_mode:
                    return self._angle_lerp(earlier_v, later_v, alpha)
                else:
                    return (1 - alpha) * earlier_v + alpha * later_v
            later_t, later_v = earlier_t, earlier_v

        # Fallback (shouldn’t hit because earlier bound already checked)
        return self.ownship_ground_truth_history[0][idx].copy()

    def process_synthetic_data(self, frame, ownship, ownship_state, intruder_states, home_lla):
        os_neu = navpy.lla2ned(
            ownship_state['lat'],
            ownship_state['lon'],
            -ownship_state['alt'],
            home_lla[0],
            home_lla[1],
            0
        )

        self.set_ground_truth_ownship_state(
            (os_neu[0], os_neu[1], ownship_state["alt"]), 
            (ownship_state["rl"], ownship_state["pt"], ownship_state["yw"]),
            ownship_state['time']
            )
        
        intruder_class_list = []
        for idx, intruder_state in enumerate(intruder_states):
            int_neu = navpy.lla2ned(
                intruder_state['lat'],
                intruder_state['lon'],
                -intruder_state['alt'],
                home_lla[0],
                home_lla[1],
                0
            )
            
            intruder_class_list.append(Intruder(idx=idx, position_xyz=(int_neu[0], int_neu[1], intruder_state["alt"])))

        self.update_encounter(frame, ownship, intruder_class_list)
        AFMats = self.generate_affine_transform_matrices()
        detections = self.detections_this_frame()

        # Wrap result in a list to match queue format
        results_queue = [{
            "curr_frames": [None] * len(detections.keys()),
            "prev_frames": [None] * len(detections.keys()),
            "predicted_transforms": [np.eye(3)] * len(detections.keys()) if AFMats is None else [AFMat for AFMat in AFMats.values()],
            "detections": [detections[i] for i in detections.keys()],
            "tracks": [None] * len(detections.keys()),
            "frame_id": [str(i) + '_' + str(ownship_state["time"]) for i in detections.keys()],
            "prev_frames_aligned_affine": [None] * len(detections.keys()),
            "total_detect_track_duration": [None] * len(detections.keys())
            }]
        
        return results_queue

    def get_ownship_mavlink_format(self, ownship_state):
        get_position = {
                'lat_deg': ownship_state['lat'],    # Position in degrees
                'lon_deg': ownship_state['lon'],
                'alt_msl_m': ownship_state['alt'],  # Altitude in meters
                'relative_home_alt_m': ownship_state['alt'],
                'ground_vx_m_s': ownship_state['ground_speed'] * np.cos(np.radians(ownship_state['trk'])), # Speed in m/s
                'ground_vy_m_s': ownship_state['ground_speed'] * np.sin(np.radians(ownship_state['trk'])),
                'ground_vz_m_s': ownship_state['vertical_speed'],
                'hdg_deg': ownship_state['trk'],  # Heading in degrees
                'timestamp': ownship_state['time'] 
            }
        
        get_attitude = {
                'timestamp': ownship_state['time'],
                'roll_deg': ownship_state['rl'],
                'pitch_deg': ownship_state['pt'],
                'yaw_deg': ownship_state['yw'],
                'roll_rate_deg_s': 0,
                'pitch_rate_deg_s': 0,
                'yaw_rate_deg_s': 0
            }
        
        return get_position, get_attitude


    # def plot(self, mode="3d", whole_history = False, fov_length=2.0):
    #     assert mode in {"3d", "camera_view", "combined"}, "Mode must be '3d', 'camera_view', or 'combined'"
    
    #     cameras = self.ownship.cameras
    #     num_cams = len(cameras)
    #     n_cols = int(np.ceil(np.sqrt(num_cams)))
    #     n_rows = int(np.ceil(num_cams / n_cols))
    
    #     if mode == "combined":
    #         fig = plt.figure(figsize=(6 * n_cols, 6 + 3 * n_rows))
    #         gs = GridSpec(n_rows + 2, n_cols, figure=fig)  # Reserve upper rows for 3D
    #         ax3d = fig.add_subplot(gs[:2, :], projection='3d')
    #     elif mode == "3d":
    #         fig = plt.figure(figsize=(10, 8))
    #         ax3d = fig.add_subplot(111, projection='3d')
    #     elif mode == "camera_view":
    #         fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    #         axs = np.array(axs).flatten()
    
    #     # === 3D Plot ===
    #     if mode in {"3d", "combined"}:
    #         cmap = cm.get_cmap('tab10')
    #         legend_elements = []
    #         global_positions = []
    
    #         R_own = self.ownship.intrinsic_neu_rpy_to_matrix(*np.radians(self.ownship_ground_truth_orientation_rpy_deg()))
    #         T_own = self.ownship_ground_truth_position()
    
    #         for idx, cam in enumerate(self.ownship.cameras):
    #             pos_local = cam["position_xyz"]
    #             rpy_local = np.radians(cam["orientation_rpy_deg"])
    #             cam_obj = cam["camera"]
    #             color = cmap(idx % 10)
    
    #             R_cam = self.ownship.intrinsic_neu_rpy_to_matrix(*rpy_local)
    #             pos_global = R_own @ pos_local + T_own
    #             R_total = R_own @ R_cam
    
    #             # FOV pyramid
    #             dx = fov_length
    #             dy = dx * np.tan(np.radians(cam_obj.fov_x_deg / 2))
    #             dz = dx * np.tan(np.radians(cam_obj.fov_y_deg / 2))
    #             corners = np.array([
    #                 [0, 0, 0],
    #                 [dx, -dy, -dz],
    #                 [dx, dy, -dz],
    #                 [dx, dy, dz],
    #                 [dx, -dy, dz]
    #             ])
    #             corners_world = (corners @ R_total.T).T + pos_global
    #             global_positions.append(pos_global)
    
    #             faces = [
    #                 [corners_world[0], corners_world[1], corners_world[2]],
    #                 [corners_world[0], corners_world[2], corners_world[3]],
    #                 [corners_world[0], corners_world[3], corners_world[4]],
    #                 [corners_world[0], corners_world[4], corners_world[1]],
    #                 [corners_world[1], corners_world[2], corners_world[3], corners_world[4]]
    #             ]
    #             ax3d.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='k'))
    #             ax3d.quiver(*pos_global, *(R_total @ [1, 0, 0]), length=0.5, color=color, normalize=True)
    #             ax3d.scatter(*pos_global, color=color, s=50)
    #             legend_elements.append((color, f"Camera {idx + 1}"))
    
    #         # Ownship last frame
    #         ax3d.scatter(*T_own, color='black', s=100)
    #         ax3d.quiver(*T_own, *(R_own @ [1, 0, 0]), length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)
    #         legend_elements.append(('black', 'Ownship'))
    
    #         intruder_colors = plt.cm.tab10(np.arange(len(self.cartesian_history.keys())))

    #         if whole_history:
    #             for idx, obj_hist in self.cartesian_history.items():
    #                 positions = np.vstack(obj_hist['position'][:-1])
    #                 orientations = [
    #                     np.array(o) if o is not None else np.array([np.nan, np.nan, np.nan])
    #                     for o in obj_hist['orientation'][:-1]
    #                 ]

    #                 if len(positions) == 0:
    #                     continue  # Skip if no history

    #                 if idx == self.ownship.id:
    #                     # === Ownship ===
    #                     ax3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='black', s=100)

    #                     for pos, orient in zip(positions, orientations):
    #                         R = self.ownship.intrinsic_neu_rpy_to_matrix(*np.radians(orient))
    #                         dir_vector = R @ np.array([1, 0, 0])
    #                         ax3d.quiver(*pos, *dir_vector, length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)

    #                     legend_elements.append(('black', 'Ownship'))
    #                 else:
    #                     # === Intruders ===
    #                     color = intruder_colors[idx % len(intruder_colors)]

    #                     ax3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color=color, s=80, marker='X')

    #                     for pos, orient in zip(positions, orientations):
    #                         R_intr = self.ownship.intrinsic_neu_rpy_to_matrix(*np.radians(orient))
    #                         dir_vector = R_intr @ np.array([1, 0, 0])
    #                         ax3d.quiver(*pos, *dir_vector, length=0.7, color=color, linewidth=2, arrow_length_ratio=0.1, normalize=True)

    #                     legend_elements.append((color, f"Intruder {idx}"))
    #         else:
    #             for i, intr in enumerate(self.intruders):
    #                 color = intruder_colors[i % 10]
    #                 ax3d.scatter(*intr.position, color=color, s=80, marker='X', label=f"Intruder {i + 1}")
    #                 if intr.orientation_rpy_deg is not None:
    #                     R_intr = self.ownship.intrinsic_neu_rpy_to_matrix(*np.radians(intr.orientation_rpy_deg))
    #                     ax3d.quiver(*intr.position, *(R_intr @ [1, 0, 0]), length=0.7, color=color, linewidth=2, arrow_length_ratio=0.1)
            
    #                 legend_elements.append((color, f"Intruder {i + 1}"))

    #         # LOS rays and red X from camera_views
    #         for cam_idx, (cam, intr_list) in self.camera_views.items():
    #             cam_obj = cam["camera"]
    #             pos_local = cam["position_xyz"]
    #             rpy_local = np.radians(cam["orientation_rpy_deg"])
    #             R_cam = self.ownship.intrinsic_neu_rpy_to_matrix(*rpy_local)
    #             R_total = R_own @ R_cam
    #             cam_global = R_own @ pos_local + T_own
    
    #             fx = cam_obj.pixels_x / (2 * np.tan(np.deg2rad(cam_obj.fov_x_deg) / 2))
    #             fy = cam_obj.pixels_y / (2 * np.tan(np.deg2rad(cam_obj.fov_y_deg) / 2))

    #             for intr_idx, (pixel_x, pixel_y, depth) in intr_list:
    #                 y = (pixel_x - cam_obj.pixels_x / 2) * fov_length / fx
    #                 z = -(pixel_y - cam_obj.pixels_y / 2) * fov_length / fy
    #                 x = fov_length
    #                 point_local = np.array([x, y, z])
    #                 point_global = R_total @ point_local + cam_global
    #                 intr_this = next(i for i in self.intruders if i.id == intr_idx)
    #                 intr_pos = intr_this.position
                            
    #                 if depth < fov_length:
    #                     ax3d.plot([cam_global[0], point_global[0]], [cam_global[1], point_global[1]], [cam_global[2], point_global[2]], 'r--')
    #                 else:
    #                     ax3d.plot([cam_global[0], intr_pos[0]], [cam_global[1], intr_pos[1]], [cam_global[2], intr_pos[2]], 'r--')

    #                 intr_idx = intr_this.id % min(10, len(intruder_colors))
    #                 color = intruder_colors[intr_idx]
    #                 ax3d.scatter(*point_global, color=color, marker='x', s=60, linewidths=2)
    
    #         if global_positions:
    #             all_points = np.vstack(global_positions)
    #             max_range = np.ptp(all_points, axis=0).max()
    #             mid = np.mean(all_points, axis=0)
    #             min_range = 1.0  # Minimum size for the bounding box
    #             max_range = max(np.ptp(all_points, axis=0).max(), min_range)
    #             zoom_out_factor = 5  # Or more, like 2.0 or 3.0
    #             for setter, m in zip([ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim], mid):
    #                 setter([m - zoom_out_factor * max_range / 2, m + zoom_out_factor * max_range / 2])
    
    #         seen = set()
    #         legend_handles = []
    #         for color, label in legend_elements:
    #             if label not in seen:
    #                 style = {'marker': 'o', 'markersize': 10, 'linestyle': 'None'}
    #                 if label == 'Ownship':
    #                     style = {'linestyle': '-', 'linewidth': 3}
    #                 elif label == 'Intruder':
    #                     style = {'marker': 'X', 'markersize': 10, 'linestyle': 'None'}
    #                 legend_handles.append(Line2D([0], [0], color=color, label=label, **style))
    #                 seen.add(label)
    
    #         ax3d.legend(handles=legend_handles)
    #         ax3d.invert_yaxis()
    #         ax3d.set_xlabel("X (forward)")
    #         ax3d.set_ylabel("Y (right)")
    #         ax3d.set_zlabel("Z (up)")
    #         ax3d.set_title("Encounter Visualization")
    #         ax3d.grid(True)
    #         ax3d.set_box_aspect([1, 1, 1])
    
    #     # === 2D Camera View Subplots ===
    #     if mode in {"camera_view", "combined"}:
    #         intruder_colors = plt.cm.tab10(np.arange(len(self.encounter_history.keys())))
    #         if mode == "camera_view":
    #             axs = np.array([
    #                 plt.subplot(n_rows, n_cols, i + 1) for i in range(num_cams)
    #             ])
    #         else:
    #             axs = [fig.add_subplot(gs[2 + i // n_cols, i % n_cols]) for i in range(num_cams)]
    
    #         axs = np.array(axs).flatten()

    #         if whole_history:
    #             for idx, cam in enumerate(cameras):
    #                 cam_obj = cam["camera"]
    #                 ax = axs[idx]
    #                 ax.set_title(f'Camera {idx}')
    #                 ax.set_xlim([0, cam_obj.pixels_x])
    #                 ax.set_ylim([0, cam_obj.pixels_y])
    #                 ax.set_aspect('equal')
    #                 ax.invert_yaxis()
    #                 ax.set_xlabel("Pixel X")
    #                 ax.set_ylabel("Pixel Y")
    #                 ax.grid(True)

    #                 for intr_id, intr in self.encounter_history.items():
    #                     if intr_id == self.ownship.id:
    #                         continue
    #                     for frame_idx in range(len(intr['frame'])):
    #                         if intr['cam'][frame_idx] == idx:
    #                             color = intruder_colors[intr_id]
    #                             ax.scatter(intr['pixel_x'][frame_idx], intr['pixel_y'][frame_idx], color=color, s=20)
    #         else:
    #             for idx, cam in enumerate(cameras):
    #                 cam_obj = cam["camera"]
    #                 ax = axs[idx]
    #                 ax.set_title(f"Camera {idx + 1}")
    #                 ax.set_xlim(0, cam_obj.pixels_x)
    #                 ax.set_ylim(0, cam_obj.pixels_y)
    #                 ax.set_aspect('equal')
    #                 ax.invert_yaxis()
    #                 ax.set_xlabel("Pixel X")
    #                 ax.set_ylabel("Pixel Y")
    #                 ax.grid(True)
        
    #                 if idx in self.camera_views:
    #                     _, intruder_list = self.camera_views[idx]
    #                     for intr_idx, (px, py, _) in intruder_list:
    #                         intr_this = next(i for i in self.intruders if i.id == intr_idx)
    #                         intr_idx = intr_this.id % min(10, len(intruder_colors))
    #                         color = intruder_colors[intr_idx]
    #                         ax.plot(px, py, marker='o', color=color, markersize=6, label=f"Intruder {intr_idx + 1}")
    
    #         for j in range(num_cams, len(axs)):
    #             fig.delaxes(axs[j])
    
    #     plt.tight_layout()
    #     plt.show()

    # def plot_perspective(self, mode="combined", fov_length=2.0):
    #     assert mode in {"3d", "camera_view", "combined"}, "Mode must be '3d', 'camera_view', or 'combined'"
    
    #     cameras = self.ownship.cameras
    #     num_cams = len(cameras)
    #     n_cols = int(np.ceil(np.sqrt(num_cams)))
    #     n_rows = int(np.ceil(num_cams / n_cols))
    
    #     if mode == "combined":
    #         fig = plt.figure(figsize=(6 * n_cols, 6 + 3 * n_rows))
    #         gs = GridSpec(n_rows + 2, n_cols, figure=fig)  # Reserve upper rows for 3D
    #         ax3d = fig.add_subplot(gs[:2, :], projection='3d')
    #     elif mode == "3d":
    #         fig = plt.figure(figsize=(10, 8))
    #         ax3d = fig.add_subplot(111, projection='3d')
    #     elif mode == "camera_view":
    #         fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    #         axs = np.array(axs).flatten()
    
    #     # === 3D Plot ===
    #     if mode in {"3d", "combined"}:
    #         cmap = cm.get_cmap('tab10')
    #         legend_elements = []
    #         global_positions = []
    
    #         R_own = self.ownship.intrinsic_neu_rpy_to_matrix(*np.radians(self.ownship_ground_truth_orientation_rpy_deg()))
    #         T_own = self.ownship_ground_truth_position()
    
    #         for idx, cam in enumerate(self.ownship.cameras):
    #             pos_local = cam["position_xyz"]
    #             rpy_local = np.radians(cam["orientation_rpy_deg"])
    #             cam_obj = cam["camera"]
    #             color = cmap(idx % 10)
    
    #             R_cam = self.ownship.intrinsic_neu_rpy_to_matrix(*rpy_local)
    #             pos_global = R_own @ pos_local + T_own
    #             R_total = R_own @ R_cam
    
    #             # FOV pyramid
    #             dx = fov_length
    #             dy = dx * np.tan(np.radians(cam_obj.fov_x_deg / 2))
    #             dz = dx * np.tan(np.radians(cam_obj.fov_y_deg / 2))
    #             corners = np.array([
    #                 [0, 0, 0],
    #                 [dx, -dy, -dz],
    #                 [dx, dy, -dz],
    #                 [dx, dy, dz],
    #                 [dx, -dy, dz]
    #             ])
    #             corners_world = (corners @ R_cam.T @ R_own.T) + pos_global
    #             global_positions.append(pos_global)
    
    #             faces = [
    #                 [corners_world[0], corners_world[1], corners_world[2]],
    #                 [corners_world[0], corners_world[2], corners_world[3]],
    #                 [corners_world[0], corners_world[3], corners_world[4]],
    #                 [corners_world[0], corners_world[4], corners_world[1]],
    #                 [corners_world[1], corners_world[2], corners_world[3], corners_world[4]]
    #             ]
    #             ax3d.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='k'))
    #             ax3d.quiver(*pos_global, *(R_total @ [1, 0, 0]), length=0.5, color=color, normalize=True)
    #             ax3d.scatter(*pos_global, color=color, s=50)
    #             legend_elements.append((color, f"Camera {idx + 1}"))
    
    #         # Ownship + intruders
    #         ax3d.scatter(*T_own, color='black', s=100)
    #         ax3d.quiver(*T_own, *(R_own @ [1, 0, 0]), length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)
    #         legend_elements.append(('black', 'Ownship'))

    #         # Extract x-values of intruders to normalize
    #         x_vals_intr = np.array([intr.position[0] for intr in self.intruders])
    #         norm = mcolors.Normalize(vmin=x_vals_intr.min(), vmax=x_vals_intr.max())
    #         cmap = cm.get_cmap('coolwarm')

    #         for i, intr in enumerate(self.intruders):
    #             x = intr.position[0]
    #             color = cmap(norm(x))
    #             ax3d.scatter(*intr.position, color=color, s=80, marker='X')
    
    #         if global_positions:
    #             all_points = np.vstack(global_positions)
    #             max_range = np.ptp(all_points, axis=0).max()
    #             mid = np.mean(all_points, axis=0)
    #             min_range = 1.0  # Minimum size for the bounding box
    #             max_range = max(np.ptp(all_points, axis=0).max(), min_range)
    #             zoom_out_factor = 10  # Or more, like 2.0 or 3.0
    #             for setter, m in zip([ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim], mid):
    #                 setter([m - zoom_out_factor * max_range / 2, m + zoom_out_factor * max_range / 2])
    
    #         seen = set()
    #         legend_handles = []
    #         for color, label in legend_elements:
    #             if label not in seen:
    #                 style = {'marker': 'o', 'markersize': 10, 'linestyle': 'None'}
    #                 if label == 'Ownship':
    #                     style = {'linestyle': '-', 'linewidth': 3}
    #                 elif label == 'Intruder':
    #                     style = {'marker': 'X', 'markersize': 10, 'linestyle': 'None'}
    #                 legend_handles.append(Line2D([0], [0], color=color, label=label, **style))
    #                 seen.add(label)

    #         ax3d.legend(handles=legend_handles)
    #         ax3d.invert_yaxis()
    #         ax3d.set_xlabel("X (forward)")
    #         ax3d.set_ylabel("Y (right)")
    #         ax3d.set_zlabel("Z (up)")
    #         ax3d.set_title("Encounter Visualization")
    #         ax3d.grid(True)
    #         ax3d.set_box_aspect([1, 1, 1])
    
    #     # === 2D Camera View Subplots ===
    #     if mode in {"camera_view", "combined"}:
    #         intruder_colors = plt.cm.tab10(np.arange(len(self.intruders)))
    #         if mode == "camera_view":
    #             axs = np.array([
    #                 plt.subplot(n_rows, n_cols, i + 1) for i in range(num_cams)
    #             ])
    #         else:
    #             axs = [fig.add_subplot(gs[2 + i // n_cols, i % n_cols]) for i in range(num_cams)]
    
    #         axs = np.array(axs).flatten()
    #         for idx, cam in enumerate(cameras):
    #             cam_obj = cam["camera"]
    #             ax = axs[idx]
    #             ax.set_title(f"Camera {idx + 1}")
    #             ax.set_xlim(0, cam_obj.pixels_x)
    #             ax.set_ylim(0, cam_obj.pixels_y)
    #             ax.set_aspect('equal')
    #             ax.invert_yaxis()
    #             ax.set_xlabel("Pixel X")
    #             ax.set_ylabel("Pixel Y")
    #             ax.grid(True)
    
    #             if idx in self.camera_views:
    #                 _, intruder_list = self.camera_views[idx]
    #                 for intr_idx, (px, py, _) in intruder_list:
    #                     intr = self.intruders[intr_idx]
    #                     x = intr.position[0]
    #                     color = cmap(norm(x))
    #                     ax.plot(px, py, marker='o', color=color, markersize=6)
    
    #         for j in range(num_cams, len(axs)):
    #             fig.delaxes(axs[j])
    
    #     plt.tight_layout()
    #     plt.show()

    # Function to compute affine transformation
    # TODO: Add Weighting of pixels around intruders and regions
    def compute_affine_transform_prev_to_current(self, current_pts, prev_pts, intruder_region = None):
        if len(prev_pts) == 0:
            return np.eye(3)

        A = []
        b = []
        for (x, y), (xp, yp) in zip(prev_pts, current_pts):
            A.append([x, y, 1, 0, 0, 0])
            A.append([0, 0, 0, x, y, 1])
            b.append(xp)
            b.append(yp)

        A = np.array(A)
        b = np.array(b)

        affine_params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        affine_matrix_2x3 = affine_params.reshape(2, 3)

        # Convert to 3x3 homogeneous matrix
        T = np.vstack([affine_matrix_2x3, [0, 0, 1]])

        return T

    def pixels_to_ground(self, cam_pos, R_cam_to_world, cam_obj, pixels):
        """
        Intersect pixel rays with the ground plane z=0.

        Args:
            cam_pos: (3,) world position of camera
            R_cam_to_world: (3,3) rotation matrix mapping camera coords -> world coords
            cam_obj: camera with pixels_x, pixels_y, fov_x_deg, fov_y_deg
            pixels: Nx2 array-like of (px, py)

        Returns:
            (N,3) array of world points on z=0 (np.nan rows if no intersection)
        """
        # --- intrinsics ---
        fx = cam_obj.pixels_x / (2.0 * np.tan(np.deg2rad(cam_obj.fov_x_deg) / 2.0))
        fy = cam_obj.pixels_y / (2.0 * np.tan(np.deg2rad(cam_obj.fov_y_deg) / 2.0))
        cx = cam_obj.pixels_x / 2.0
        cy = cam_obj.pixels_y / 2.0

        pixels = np.asarray(pixels, dtype=float)
        px = pixels[:, 0]
        py = pixels[:, 1]

        # Rays in CAMERA frame (your convention: +x forward, +y right, +z up)
        x = np.ones_like(px)
        y = (px - cx) / fx
        z = -(py - cy) / fy     # image y down -> z up
        rays_cam = np.stack([x, y, z], axis=1)  # (N,3)

        # Rotate rays to WORLD frame (row-vectors -> right-multiply by R^T)
        R = np.asarray(R_cam_to_world, dtype=float)
        rays_world = rays_cam @ R.T

        cam_pos = np.asarray(cam_pos, dtype=float)

        # Intersect with plane z=0 along ray: cam_pos + t * ray; solve cam_pos[2] + t*ray_z = 0
        ray_z = rays_world[:, 2]
        eps = 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -cam_pos[2] / np.where(np.abs(ray_z) < eps, np.nan, ray_z)

        # valid only if ray points downward toward the ground and t >= 0
        valid = np.isfinite(t) & (t >= 0.0) & (ray_z != 0.0)
        ground_points = np.full((len(pixels), 3), np.nan, dtype=float)
        ground_points[valid] = cam_pos + rays_world[valid] * t[valid, None]

        return ground_points

    def compute_horizon_line(self, cam_pos, R_total, cam_obj, radius=1000000, num_points=100):

        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        circle_xy = cam_pos[:2] + np.column_stack((np.cos(angles), np.sin(angles))) * radius
        z = np.zeros((num_points, 1))  # z=0 for all points
        
        points = np.hstack((circle_xy, z))

        horizon_line_coords = []
        for point in points:
            intr_vec_global = point - cam_pos
            intr_cam_frame = R_total.T @ intr_vec_global  # Rotate into camera frame
    
            x, y, z = intr_cam_frame
            if x <= 0:
                continue  # Behind camera
    
            h_angle = np.degrees(np.arctan2(y, x))
            v_angle = np.degrees(np.arctan2(z, x))
    
            if (abs(h_angle) <= cam_obj.fov_x_deg / 2) and (abs(v_angle) <= cam_obj.fov_y_deg / 2):
                # Calculate Pixel Located on Camera
                fx = cam_obj.pixels_x / (2 * np.tan(np.deg2rad(cam_obj.fov_x_deg) / 2))
                fy = cam_obj.pixels_y / (2 * np.tan(np.deg2rad(cam_obj.fov_y_deg) / 2))
                cx = cam_obj.pixels_x / 2
                cy = cam_obj.pixels_y / 2
                
                pixel_X = cx + fx * (y / x)  # positive y → right → higher pixel_x
                pixel_Y = cy - fy * (z / x)  # positive z → up → lower pixel_y

                horizon_line_coords.append((pixel_X, pixel_Y))

        return horizon_line_coords

    def generate_affine_transform_matrices(self, fov_length=2.0, plot = False):
        if len(self.encounter_history[self.ownship.id]['frame']) < 2:
            return None
    
        # if plot:
        #     num_cams = len(self.ownship.cameras)
        #     n_cols = int(np.ceil(np.sqrt(num_cams)))
        #     n_rows = int(np.ceil(num_cams / n_cols))
        #     fig = plt.figure(figsize=(6 * n_cols, 6 + 3 * n_rows))
        #     gs = GridSpec(n_rows + 2, n_cols, figure=fig)  # Reserve upper rows for 3D
        #     ax3d = fig.add_subplot(gs[:2, :], projection='3d')
        #     cmap = cm.get_cmap('tab10')
        #     legend_elements = []
        #     global_positions = []
        #     horizon_lines = {}
    
        affine_trans_matrices = {}
        frame_pair_ground_pixels = {}
        ground_coords = {}
        
        for idx, cam in enumerate(self.ownship.cameras):
            frame_pair_ground_pixels[idx] = {'current_frame': [], 'prev_frame': [], 'transformed_current': []}
            ground_coords[idx] = []
            # Camera Positioning Specs
            pos_local = cam["position_xyz"]
            rpy_local = np.radians(cam["orientation_rpy_deg"])
            
            cam_obj = cam["camera"]

            # Generate Input Pixels
            num_grid = 10
            margin_ratio = 0.025  # 2.5% margin on each side

            x_start = cam_obj.pixels_x * margin_ratio
            x_end = cam_obj.pixels_x * (1 - margin_ratio)
            y_start = cam_obj.pixels_y * margin_ratio
            y_end = cam_obj.pixels_y * (1 - margin_ratio)

            x_vals = np.linspace(x_start, x_end, num_grid)
            y_vals = np.linspace(y_start, y_end, num_grid)

            input_pixels = [(x, y) for y in y_vals for x in x_vals]

            frame_pair_ground_pixels[idx]['current_frame'] = input_pixels
    
            R_own = self.ownship.intrinsic_neu_rpy_to_matrix(*np.radians(self.ownship_ground_truth_orientation_rpy_deg()))
            T_own = self.ownship_ground_truth_position()

            R_cam = self.ownship.intrinsic_neu_rpy_to_matrix(*rpy_local)
            pos_global = R_own @ pos_local + T_own
            R_total = R_own @ R_cam

            # if plot:
            #     horizon_lines[idx] = self.compute_horizon_line(pos_global, R_total, cam_obj)

            ground_coords[idx] = self.pixels_to_ground(pos_global, R_total, cam_obj, input_pixels)

            ##################################
            R_own_prev = self.ownship.intrinsic_neu_rpy_to_matrix(*np.radians(self.encounter_history[self.ownship.id]['orientation'][-2]))
            T_own_prev = self.encounter_history[self.ownship.id]['position'][-2]
    
            # Global camera position and direction
            pos_global_prev = R_own_prev @ pos_local + T_own_prev
            R_total_prev = R_own_prev @ R_cam
            
            if len([item for item in ground_coords[idx] if isinstance(item, np.ndarray) and np.all(np.isfinite(item))]) < 3:
                affine_trans_matrices[idx] = np.eye(3)

            for ground_coord in ground_coords[idx]:
            # Transform intruder position to camera local frame
                intr_vec_global_prev = ground_coord - pos_global_prev
                intr_cam_frame_prev = R_total_prev.T @ intr_vec_global_prev  # Rotate into camera frame
        
                x, y, z = intr_cam_frame_prev
                if x <= 0:
                    frame_pair_ground_pixels[idx]['prev_frame'].append((np.nan, np.nan))
                    continue  # Behind camera
                else:
                    h_angle = np.degrees(np.arctan2(y, x))
                    v_angle = np.degrees(np.arctan2(z, x))
            
                    if (abs(h_angle) <= cam_obj.fov_x_deg / 2) and (abs(v_angle) <= cam_obj.fov_y_deg / 2):
                        # Calculate Pixel Located on Camera
                        fx = cam_obj.pixels_x / (2 * np.tan(np.deg2rad(cam_obj.fov_x_deg) / 2))
                        fy = cam_obj.pixels_y / (2 * np.tan(np.deg2rad(cam_obj.fov_y_deg) / 2))
                        cx = cam_obj.pixels_x / 2
                        cy = cam_obj.pixels_y / 2
                        
                        pixel_X = cx + fx * (y / x)  # positive y → right → higher pixel_x
                        pixel_Y = cy - fy * (z / x)  # positive z → up → lower pixel_y

                        frame_pair_ground_pixels[idx]['prev_frame'].append((pixel_X, pixel_Y))
                    else:
                        frame_pair_ground_pixels[idx]['prev_frame'].append((np.nan, np.nan))

            current_frame = np.array(frame_pair_ground_pixels[idx]['current_frame'])
            prev_frame = np.array(frame_pair_ground_pixels[idx]['prev_frame'])

            # Keep rows where both x and y in prev_frame are finite
            mask = np.all(np.isfinite(prev_frame), axis=1)

            # Apply mask
            frame_pair_ground_pixels[idx]['current_frame'] = current_frame[mask]
            frame_pair_ground_pixels[idx]['prev_frame'] = prev_frame[mask]

            affine_trans_matrices[idx] = self.compute_affine_transform_prev_to_current(frame_pair_ground_pixels[idx]['current_frame'], frame_pair_ground_pixels[idx]['prev_frame'])

            # ###################################
            # if plot:
            #     # FOV pyramid
            #     dx = fov_length
            #     dy = dx * np.tan(np.radians(cam_obj.fov_x_deg / 2))
            #     dz = dx * np.tan(np.radians(cam_obj.fov_y_deg / 2))
            #     corners = np.array([
            #         [0, 0, 0],
            #         [dx, -dy, -dz],
            #         [dx, dy, -dz],
            #         [dx, dy, dz],
            #         [dx, -dy, dz]
            #     ])
            #     corners_world = (corners @ R_cam.T @ R_own.T) + pos_global
            #     global_positions.append(pos_global)
    
            #     faces = [
            #         [corners_world[0], corners_world[1], corners_world[2]],
            #         [corners_world[0], corners_world[2], corners_world[3]],
            #         [corners_world[0], corners_world[3], corners_world[4]],
            #         [corners_world[0], corners_world[4], corners_world[1]],
            #         [corners_world[1], corners_world[2], corners_world[3], corners_world[4]]
            #     ]

            #     color = cmap(idx % 10)

            #     ax3d.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='k'))
            #     ax3d.quiver(*pos_global, *(R_total @ [1, 0, 0]), length=0.5, color=color, normalize=True)
            #     ax3d.scatter(*pos_global, color=color, s=50)
            #     legend_elements.append((color, f"Camera {idx + 1}"))

        # if plot:
        #     # Ownship
        #     ax3d.scatter(*T_own, color='black', s=100)
        #     ax3d.quiver(*T_own, *(R_own @ [1, 0, 0]), length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)
        #     legend_elements.append(('black', 'Ownship'))

        #     for cam, coord_list in ground_coords.items():
        #         for coord in coord_list:
        #             if not np.any(np.isnan(coord)):
        #                 ax3d.scatter(*coord, color='red', s=20, marker='o')

        #     # Intruders
        #     intruder_colors = plt.cm.tab10(np.arange(len(self.intruders)))

        #     for i, intr in enumerate(self.intruders):
        #         color = intruder_colors[i % 10]
        #         ax3d.scatter(*intr.position, color=color, s=80, marker='X', label=f"Intruder {i + 1}")
        #         if intr.orientation_rpy_deg is not None:
        #             R_intr = self.ownship.intrinsic_neu_rpy_to_matrix(*np.radians(intr.orientation_rpy_deg))
        #             ax3d.quiver(*intr.position, *(R_intr @ [1, 0, 0]), length=0.7, color=color, linewidth=2, arrow_length_ratio=0.1)
            
        #         legend_elements.append((color, f"Intruder {i + 1}"))

        #     if global_positions:
        #         all_points = np.vstack(global_positions)
        #         max_range = np.ptp(all_points, axis=0).max()
        #         mid = np.mean(all_points, axis=0)
        #         min_range = 1.0  # Minimum size for the bounding box
        #         max_range = max(np.ptp(all_points, axis=0).max(), min_range)
        #         zoom_out_factor = 10  # Or more, like 2.0 or 3.0
        #         for setter, m in zip([ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim], mid):
        #             setter([m - zoom_out_factor * max_range / 2, m + zoom_out_factor * max_range / 2])
    
        #     seen = set()
        #     legend_handles = []
        #     for color, label in legend_elements:
        #         if label not in seen:
        #             style = {'marker': 'o', 'markersize': 10, 'linestyle': 'None'}
        #             if label == 'Ownship':
        #                 style = {'linestyle': '-', 'linewidth': 3}
        #             elif label == 'Intruder':
        #                 style = {'marker': 'X', 'markersize': 10, 'linestyle': 'None'}
        #             legend_handles.append(Line2D([0], [0], color=color, label=label, **style))
        #             seen.add(label)

        #     ax3d.legend(handles=legend_handles)
        #     ax3d.invert_yaxis()
        #     ax3d.set_xlabel("X (forward)")
        #     ax3d.set_ylabel("Y (right)")
        #     ax3d.set_zlabel("Z (up)")
        #     ax3d.set_title("Encounter Visualization")
        #     ax3d.grid(True)
        #     ax3d.set_box_aspect([1, 1, 1])
    
        #     # === 2D Camera View Subplots ===
        #     axs = [fig.add_subplot(gs[2 + i // n_cols, i % n_cols]) for i in range(num_cams)]
        
        #     axs = np.array(axs).flatten()
        #     for idx, cam in enumerate(self.ownship.cameras):
        #         cam_obj = cam["camera"]
        #         ax = axs[idx]
        #         ax.set_title(f"Camera {idx + 1}")
        #         ax.set_xlim(0, cam_obj.pixels_x)
        #         ax.set_ylim(0, cam_obj.pixels_y)
        #         ax.set_aspect('equal')
        #         ax.invert_yaxis()
        #         ax.set_xlabel("Pixel X")
        #         ax.set_ylabel("Pixel Y")
        #         ax.grid(True)

        #         ax.plot(*zip(*horizon_lines[idx]), linestyle='--', color='orange')

        #         if idx in self.camera_views:
        #             ground_coord_dict = frame_pair_ground_pixels[idx]
        #             for px, py in ground_coord_dict['current_frame']:
        #                 ax.plot(px, py, marker='o', color='red', markersize=6)
        #             for px, py in ground_coord_dict['prev_frame']:
        #                 ax.plot(px, py, marker='x', color='green', markersize=6)
        #             if affine_trans_matrices[idx] is not None:
        #                 for px, py in ground_coord_dict['prev_frame']:
        #                     px_new, py_new = (affine_trans_matrices[idx] @ np.array([px, py, 1]))[:2]
        #                     ax.plot(px_new, py_new, marker='x', color='blue', markersize=6)

        #         if idx in self.camera_views:
        #             _, intruder_list = self.camera_views[idx]
        #             for intr_idx, (px, py, _) in intruder_list:
        #                 intr_this = next(i for i in self.intruders if i.id == intr_idx)
        #                 intr_idx = intr_this.id % min(10, len(intruder_colors))
        #                 color = intruder_colors[intr_idx]
        #                 ax.plot(px, py, marker='o', color=color, markersize=6, label=f"Intruder {intr_idx + 1}")
    
        #     for j in range(num_cams, len(axs)):
        #         fig.delaxes(axs[j])
        
        #     plt.tight_layout()
        #     plt.show()   

        return affine_trans_matrices     