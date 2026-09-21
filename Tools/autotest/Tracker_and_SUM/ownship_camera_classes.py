import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import *
# from matplotlib import cm
from functools import partial
from Tracker_and_SUM.prediction_structure_v2 import *
import os
import pickle
from Tracker_and_SUM.sum_functions import *

#### OWNSHIP CLASS ####

AMAZON_CAMERA_RES = [2448, 2048]
AMAZON_CAMERA_FOV = [67.32, 56.52]
class Camera:
    def __init__(self, resolution_xy: Tuple[int, int], fov_deg_xy: Tuple[float, float]):
        self.pixels_x, self.pixels_y = resolution_xy
        self.fov_x_deg, self.fov_y_deg = fov_deg_xy
        self.range_error_model_input_format = None
        self.synthetic_uncertainty_models = {}
        self.detection_uncertainty_models = {}

    def __repr__(self):
        return (f"Camera(resolution=({self.pixels_x}, {self.pixels_y}), "
                f"FOV=({self.fov_x_deg} deg, {self.fov_y_deg} deg))")
    
    def add_synthetic_uncertainty_models(self, pixel_uncertainty_model_input, range_uncertainty_model_input, range_error_model_input_format_input, pdet_uncertainty_model_input, pdet_transition_model_input):
        # Partially apply the resolution and FOV to the functions
        self.synthetic_uncertainty_models["pixel_uncertainty_model"] = partial(
            pixel_uncertainty_model_input,
            new_res=(self.pixels_x, self.pixels_y)[1],
            new_fov=(self.fov_x_deg, self.fov_y_deg)[1]
        )
        self.synthetic_uncertainty_models["range_uncertainty_model"] = partial(
            range_uncertainty_model_input,
            new_res=(self.pixels_x, self.pixels_y)[1],
            new_fov=(self.fov_x_deg, self.fov_y_deg)[1]
        )
        self.synthetic_uncertainty_models["range_error_model_input_format"] = range_error_model_input_format_input
        self.synthetic_uncertainty_models["pdet_uncertainty_model"] = partial(
            pdet_uncertainty_model_mixture,
            model_params_init = pdet_uncertainty_model_input['init_model'],
            model_params = pdet_uncertainty_model_input['gen_model'],
            distance_bins = list(pdet_uncertainty_model_input['init_model'][True].keys()),
            acf_vals = pdet_uncertainty_model_input['lag_autocorrs'],
            lags = pdet_uncertainty_model_input['lags'],
            new_res=(self.pixels_x, self.pixels_y)[1],
            new_fov=(self.fov_x_deg, self.fov_y_deg)[1],
            trained_models = pdet_transition_model_input
        )

    def add_detection_uncertainty_models(self, pixel_uncertainty_model_input = None, detected_range_uncertainty_params = None):
        if pixel_uncertainty_model_input is not None:
            self.detection_uncertainty_models["detected_pixel_uncertainty_params"] = partial(
                pixel_uncertainty_model_input,
                new_res=(self.pixels_x, self.pixels_y)[1],
                new_fov=(self.fov_x_deg, self.fov_y_deg)[1],
                params_only = True
            )
        else:
            self.detection_uncertainty_models["detected_pixel_uncertainty_params"] = None
        self.detection_uncertainty_models["detected_range_uncertainty_params"] = detected_range_uncertainty_params

    def _pixel_ratio_axis(self, new_fov_deg, old_fov_deg, new_res_px, old_res_px):
        fov_old_rad = np.deg2rad(old_fov_deg)
        fov_new_rad = np.deg2rad(new_fov_deg)
        return (new_res_px / old_res_px) * (np.tan(fov_old_rad / 2) / np.tan(fov_new_rad / 2))

    def calibrate_ranges(self, distances, model_output_to_true_scale=True, pre_scale_factor = 1.0, empirical_post_scale_factor = 1.0):  
        ###### Reset distance by removing previous scaling factor ###### 
        distances = empirical_post_scale_factor * (distances / pre_scale_factor)

        #######################################################################
        ######  Calculate new scaling factor based on camera instrinsics ###### 
        #######################################################################
        # X-Pixels (additional dimension for area-based calibration)
        pixel_ratio_x = self._pixel_ratio_axis(
            new_fov_deg=self.fov_x_deg,
            old_fov_deg=AMAZON_CAMERA_FOV[0],
            new_res_px=self.pixels_x,
            old_res_px=AMAZON_CAMERA_RES[0],
        )

        # Y-Pixels (original conservative dimension)
        pixel_ratio_y = self._pixel_ratio_axis(
            new_fov_deg=self.fov_y_deg,
            old_fov_deg=AMAZON_CAMERA_FOV[1],
            new_res_px=self.pixels_y,
            old_res_px=AMAZON_CAMERA_RES[1],
        )

        ###### Compute final pixel area ratio by taking product of ratios to get pixel area-based factor ######
        pixel_ratio = pixel_ratio_x * pixel_ratio_y

        ################################################
        ######  Apply Scaling Factor to Distances ###### 
        ################################################

        # If converting model output range to true range
        if model_output_to_true_scale:
            scale = pixel_ratio
        # If converting true range to model output range
        else:
            scale = 1.0 / pixel_ratio
        
        # Apply scaling
        if np.isscalar(distances):
            return distances * scale
        else:
            return list(np.array(distances) * scale)

    def get_detected_uncertainty_params(self, distance):
        if list(self.detection_uncertainty_models.values()) == [None, None]:
            return 0.0, 0.0, 0.0
        
        # Initialise
        range_bias, range_var, pixel_var = 0.0, 0.0, 0.0

        # Convert range from true scale to model scale to sample SUM 
        distance_scaled = self.calibrate_ranges(distance, model_output_to_true_scale=False)

        range_uncertainty_params = self.detection_uncertainty_models.get("detected_range_uncertainty_params", None)
        if range_uncertainty_params is not None:
            # Loop over bins to find the right range bin for range uncertainty
            prev_mean_inverse = 0
            for (bin_start, bin_end), params in range_uncertainty_params.items():
                mean_inverse_scale, variance_inverse_scale = params['mean'], params['variance']
                # mean_inverse_scale = max(prev_mean_inverse, mean_inverse_scale)
                # prev_mean_inverse = mean_inverse_scale
                if bin_start <= distance_scaled < bin_end:
                    range_bias = self.calibrate_ranges(mean_inverse_scale, model_output_to_true_scale=True)
 
                    range_std = self.calibrate_ranges(np.sqrt(variance_inverse_scale), model_output_to_true_scale=True)

                    range_var = range_std ** 2

        pixel_var_model = self.detection_uncertainty_models.get("detected_pixel_uncertainty_params", None)
        if pixel_var_model is not None:
            pixel_var = pixel_var_model(distance_scaled)

        return range_bias, range_var, pixel_var

from collections import deque

class Ownship:
    def __init__(self, id):
        self.id = id
        self.cameras = []
        self.track_history = deque()
        self.history_sec: float = 5.0

    def reset(self, id):
        self.id = id
        self.track_history.clear()

    def set_state(self, position_xyz: Tuple[float, float, float],
                  orientation_rpy_deg: Tuple[float, float, float],
                  ownship_time: float):
        
        # Ignore duplicate or superceded ownship telemetry
        if self.track_history:
            if ownship_time <= self.track_history[-1][0]:
                return False
            
        pos = np.asarray(position_xyz, float)
        ori = np.asarray(orientation_rpy_deg, float)
        cutoff = ownship_time - self.history_sec

        while self.track_history and self.track_history[0][0] < cutoff:
            self.track_history.popleft()

        self.track_history.append((ownship_time, pos, ori))

        return True

    # ---------------------------
    # Get the position or orientation of ownship at a given time
    # ---------------------------
    def position(self, ownship_time: Optional[float] = None) -> np.ndarray:
        return self._interp(ownship_time, idx=1)

    def orientation_rpy_deg(self, ownship_time: Optional[float] = None) -> np.ndarray:
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
        if not self.track_history:
            print("Track_history is empty; no state available. Returning default zeros")
            return np.zeros(3)

        # Choose whether we're angle-aware (orientation) or linear (position)
        angle_mode = (idx == 2)

        if t is None:
            return self.track_history[-1][idx].copy()
        
        # Snap if close to last time
        if abs(t - self.track_history[-1][0]) <= snap_threshold:
            return self.track_history[-1][idx].copy()

        # Before earliest
        if t < self.track_history[0][0]:
            print(f"[Ownship] Requested time {t:.6f} is before earliest history time {self.track_history[0][0]:.6f}. "
                f"Returning earliest sample.")
            return self.track_history[0][idx].copy()

        # After latest → extrapolate
        if t > self.track_history[-1][0]:
            if len(self.track_history) >= 2:
                t0, v0 = self.track_history[-2][0], self.track_history[-2][idx]
                t1, v1 = self.track_history[-1][0], self.track_history[-1][idx]
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
            print(f"[Ownship] Requested time {t:.6f} is after latest history time {self.track_history[-1][0]:.6f}, "
                f"but insufficient/degenerate history to extrapolate. Returning latest sample.")
            return self.track_history[-1][idx].copy()

        # Walk backwards from the latest until we find bounding samples
        later_t, later_v = self.track_history[-1][0], self.track_history[-1][idx]
        for earlier_t, pos, ori in reversed(list(self.track_history)[:-1]):
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
        return self.track_history[0][idx].copy()

    def add_camera(self,
                   camera: Camera,
                   position_xyz: Tuple[float, float, float],
                   orientation_rpy_deg: Tuple[float, float, float]):
        self.cameras.append({
            "camera": camera,
            "position_xyz": np.array(position_xyz),
            "orientation_rpy_deg": np.array(orientation_rpy_deg)
        })


    def mount_cameras_from_config_file(self, camera_configs, script_dir):
        generate_synthetic_uncertainty = camera_configs.get("generate_synthetic_uncertainty", False)
        generate_detection_uncertainty = camera_configs.get("generate_detection_uncertainty", False)
        camera_list = camera_configs["cameras"]

        # === Optional Model Loading Utilities ===
        def load_pickle(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)

        def load_numpy(filename):
            return np.load(filename, allow_pickle=True)

        # === Load Synthetic Uncertainty Models ===
        if generate_synthetic_uncertainty:
            model_paths = camera_configs.get("uncertainty_model_files", {})

            try:
                pixel_model_state_path = model_paths["pixel_model_state_path"]
                range_bins_path = model_paths["range_bins_path"]
                range_model_input_format_path = model_paths["range_model_input_format_path"]
                range_model_params_path = model_paths["range_model_params_path"]
                pdet_uncertainty_model_path = model_paths["pdet_uncertainty_model"]
                pdet_transition_model_path = model_paths["pdet_transition_models"]

                print("[Ownship] Loading synthetic uncertainty models from config paths...")

                pixel_uncertainty_model_state_data = tuple(load_pickle(os.path.join(script_dir, pixel_model_state_path)))
                range_bins = load_numpy(os.path.join(script_dir, range_bins_path))
                range_error_model_input_format = load_pickle(os.path.join(script_dir, range_model_input_format_path))
                range_model_params = load_pickle(os.path.join(script_dir, range_model_params_path))
                pdet_uncertainty_model = load_pickle(os.path.join(script_dir, pdet_uncertainty_model_path))
                pdet_transition_model = load_pickle(os.path.join(script_dir, pdet_transition_model_path))

                # === Reconstruct Uncertainty Model Functions ===
                pixel_uncertainty_model = generate_pixel_uncertainty_model(
                    bins=20,
                    pregen_state_data=pixel_uncertainty_model_state_data
                )

                range_uncertainty_model = generate_range_uncertainty_model(
                    range_bins=range_bins,
                    pregen_state_data=range_error_model_input_format,
                    pregen_model_params=range_model_params
                )


            except KeyError as e:
                raise ValueError(f"Missing model file path in config: {e}")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not load model file: {e}")

        if generate_detection_uncertainty:
            model_paths = camera_configs.get("uncertainty_model_files", {})

            try:
                detected_range_model_params_path = model_paths["detected_range_model_params_path"]

                print("[Ownship] Loading detected uncertainty models from config paths...")

                detected_range_uncertainty_model_params = load_pickle(os.path.join(script_dir, detected_range_model_params_path))

                if not generate_synthetic_uncertainty:
                    pixel_model_state_path = model_paths["pixel_model_state_path"]
                    pixel_uncertainty_model_state_data = tuple(load_pickle(os.path.join(script_dir, pixel_model_state_path)))

                detection_pixel_uncertainty_model = generate_pixel_uncertainty_model(
                    bins=20,
                    pregen_state_data=pixel_uncertainty_model_state_data
                )
                    
            except KeyError as e:
                raise ValueError(f"Missing model file path in config: {e}")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not load model file: {e}")

        # === Create and Mount Cameras ===
        for cam_cfg in camera_list:
            res = (cam_cfg["resolution"]["width"], cam_cfg["resolution"]["height"])
            fov = (cam_cfg["fov"]["horizontal_deg"], cam_cfg["fov"]["vertical_deg"])
            pos = tuple(cam_cfg["position"])
            orientation = tuple(cam_cfg["orientation_rpy_deg"])

            cam = Camera(res, fov)

            if generate_synthetic_uncertainty:
                cam.add_synthetic_uncertainty_models(
                    pixel_uncertainty_model_input=pixel_uncertainty_model,
                    range_uncertainty_model_input=range_uncertainty_model,
                    range_error_model_input_format_input=range_error_model_input_format,
                    pdet_uncertainty_model_input=pdet_uncertainty_model,
                    pdet_transition_model_input=pdet_transition_model
                )

            if generate_detection_uncertainty:
                cam.add_detection_uncertainty_models(
                    pixel_uncertainty_model_input = detection_pixel_uncertainty_model, detected_range_uncertainty_params = detected_range_uncertainty_model_params
                )

            # Mount camera
            self.add_camera(cam, pos, orientation)
        
    def list_cameras(self):
        for idx, cam in enumerate(self.cameras):
            print(f"Camera {idx + 1}:")
            print(f"  {cam['camera']}")
            print(f"  Position (m): {cam['position_xyz']}")
            print(f"  Orientation (deg): {cam['orientation_rpy_deg']}")
            print()

    def intrinsic_neu_rpy_to_matrix(self, roll, pitch, yaw):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        pitch *= -1 # Make yaw negative since I'm using non-standard NEU instead of NED
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def map_pixel_views_to_global_cartesian_coords(self, cam_id: int, pixel_range_time_array: np.ndarray):
        """
        Converts pixel coordinates and horizontal XY-plane ranges into global 3D positions.

        Args:
            cam_id: Integer, camera index.
            pixel_range_array: Nx3 numpy array where each row is [pixel_x, pixel_y, horizontal_range],
                            where horizontal_range is the horizontal distance in the global XY plane
                            between the camera and the target (ignoring altitude difference).

        Returns:
            Nx3 numpy array of global XYZ positions.
        """
        cam = self.cameras[cam_id]
        cam_obj = cam["camera"]
        pos_local = np.array(cam["position_xyz"])
        rpy_local = np.radians(cam["orientation_rpy_deg"])

        # Camera intrinsics
        fx = cam_obj.pixels_x / (2 * np.tan(np.radians(cam_obj.fov_x_deg / 2)))
        fy = cam_obj.pixels_y / (2 * np.tan(np.radians(cam_obj.fov_y_deg / 2)))
        cx = cam_obj.pixels_x / 2
        cy = cam_obj.pixels_y / 2

        # Unpack pixel coordinates and horizontal ranges
        pixel_x = pixel_range_time_array[:, 0]
        pixel_y = pixel_range_time_array[:, 1]
        horizontal_ranges = pixel_range_time_array[:, 2]
        times = pixel_range_time_array[:, 3]

        # Back-project rays in camera frame
        x_cam = np.ones_like(pixel_x)
        y_cam = (pixel_x - cx) / fx
        z_cam = -(pixel_y - cy) / fy
        rays_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Normalize rays
        rays_cam_normalized = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)

        # Transform rays into global frame direction
        R_cam = self.intrinsic_neu_rpy_to_matrix(*rpy_local)

        #### Project cam rays using interpolated ownship position at the exact time of detection ####
        # Per-row ownship pose at the corresponding time
        N = rays_cam_normalized.shape[0]
        rays_global = np.empty_like(rays_cam_normalized)
        cam_global_pos = np.empty((N, 3), dtype=float)

        for i, t in enumerate(times):
            R_own = self.intrinsic_neu_rpy_to_matrix(*np.radians(self.orientation_rpy_deg(t)))  # ownship→global at time t
            T_own = self.position(t)                                               # ownship origin at time t

            R_total = R_own @ R_cam                                                # camera→global at time t
            rays_global[i] = R_total @ rays_cam_normalized[i]                      # rotate ray i
            cam_global_pos[i] = R_own @ pos_local + T_own                          # camera origin in global

        # Scale rays so their horizontal (XY) length matches the given horizontal range
        horizontal_norms = np.linalg.norm(rays_global[:, :2], axis=1, keepdims=True)
        scaling_factors = horizontal_ranges[:, np.newaxis] / (horizontal_norms + 1e-8)  # Avoid divide by zero
        points_global = cam_global_pos + rays_global * scaling_factors

        return points_global  # Shape: (N, 3)

    # def plot(self, fov_length=2.0):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     cmap = cm.get_cmap('tab10')
    #     legend_elements = []
    #     global_positions = []

    #     R_ownship = self.intrinsic_neu_rpy_to_matrix(*np.radians(self.orientation_rpy_deg))
    #     T_ownship = self.position

    #     for idx, cam in enumerate(self.cameras):
    #         pos_local = cam["position_xyz"]
    #         rpy_local = np.radians(cam["orientation_rpy_deg"])
    #         cam_obj = cam["camera"]
    #         color = cmap(idx % 10)

    #         R_cam = self.intrinsic_neu_rpy_to_matrix(*rpy_local)

    #         # Define FOV corners in camera frame
    #         dx = fov_length
    #         dy = dx * np.tan(np.radians(cam_obj.fov_x_deg / 2))
    #         dz = dx * np.tan(np.radians(cam_obj.fov_y_deg / 2))
    #         corners = np.array([
    #             [0, 0, 0],
    #             [dx, -dy, -dz],
    #             [dx, dy, -dz],
    #             [dx, dy, dz],
    #             [dx, -dy, dz]
    #         ])

    #         # Apply camera and ownship rotation and translation
    #         pos_global = R_ownship @ pos_local + T_ownship
    #         corners_world = (corners @ R_cam.T @ R_ownship.T) + pos_global
    #         global_positions.append(pos_global)

    #         # FOV pyramid faces
    #         faces = [
    #             [corners_world[0], corners_world[1], corners_world[2]],
    #             [corners_world[0], corners_world[2], corners_world[3]],
    #             [corners_world[0], corners_world[3], corners_world[4]],
    #             [corners_world[0], corners_world[4], corners_world[1]],
    #             [corners_world[1], corners_world[2], corners_world[3], corners_world[4]]
    #         ]
    #         poly = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='k')
    #         ax.add_collection3d(poly)

    #         # Draw direction arrow
    #         cam_dir_local = np.array([1, 0, 0])  # forward in camera frame
    #         cam_dir_global = R_ownship @ R_cam @ cam_dir_local
    #         ax.quiver(*pos_global, *cam_dir_global, length=0.5, color=color, normalize=True)
    #         ax.scatter(*pos_global, color=color, s=50)

    #         legend_elements.append((color, f"Camera {idx + 1}"))

    #     # Plot ownship centroid
    #     ax.scatter(*self.position, color='black', s=100, label='Ownship')
        
    #     # Plot ownship orientation (forward direction)
    #     ownship_dir = self.intrinsic_neu_rpy_to_matrix(*np.radians(self.orientation_rpy_deg)) @ np.array([1, 0, 0])
    #     ax.quiver(*self.position, *ownship_dir, length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)
        
    #     # Add 'Ownship' to legend if not already
    #     legend_elements.append(('black', 'Ownship'))
        

    #     # Axis scaling based on all global camera positions
    #     if global_positions:
    #         all_points = np.vstack(global_positions)
    #         max_range = np.ptp(all_points, axis=0).max()
    #         mid = np.mean(all_points, axis=0)
    #         for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
    #             axis([m - max_range / 2, m + max_range / 2])

    #     # Legend
    #     legend_handles = [Line2D([0], [0], marker='o', color=color, label=label,
    #                              markersize=10, linestyle='None') for color, label in legend_elements]
    #     ax.legend(handles=legend_handles, loc='upper right')

    #     ax.invert_yaxis()
    #     ax.set_xlabel("X (forward)")
    #     ax.set_ylabel("Y (right)")
    #     ax.set_zlabel("Z (up)")
    #     ax.set_title("Aircraft Cameras and FOVs")
    #     ax.grid(True)
    #     ax.set_box_aspect([1, 1, 1])
    #     plt.show()