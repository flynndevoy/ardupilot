# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:06:34 2025

@author: adamg
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from typing import *
from matplotlib import cm
from functools import partial
import time
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as R
from Tracker_and_SUM import tracking
from Tracker_and_SUM.prediction_structure import *
import yaml
from Tracker_and_SUM.common_utils import DotDict
from filterpy.kalman import KalmanFilter

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
        

#### OWNSHIP CLASS ####

class Camera:
    def __init__(self, resolution_xy: Tuple[int, int], fov_deg_xy: Tuple[float, float]):
        self.pixels_x, self.pixels_y = resolution_xy
        self.fov_x_deg, self.fov_y_deg = fov_deg_xy
        self.range_error_model_input_format = None

    def __repr__(self):
        return (f"Camera(resolution=({self.pixels_x}, {self.pixels_y}), "
                f"FOV=({self.fov_x_deg} deg, {self.fov_y_deg} deg))")
    
    def add_uncertainty_models(self, pixel_uncertainty_model_input, range_uncertainty_model_input, detected_range_uncertainty_params, range_error_model_input_format_input):
        # Partially apply the resolution and FOV to the functions
        self.pixel_uncertainty_model = partial(
            pixel_uncertainty_model_input,
            new_res=(self.pixels_x, self.pixels_y)[1],
            new_fov=(self.fov_x_deg, self.fov_y_deg)[1]
        )
        self.range_uncertainty_model = partial(
            range_uncertainty_model_input,
            new_res=(self.pixels_x, self.pixels_y)[1],
            new_fov=(self.fov_x_deg, self.fov_y_deg)[1]
        )
        self.detected_range_uncertainty_params = detected_range_uncertainty_params
        self.range_error_model_input_format = range_error_model_input_format_input

    def get_detected_range_uncertainty_params(self, distance):
        AMAZON_CAMERA_RES = [2448, 2048]
        AMAZON_CAMERA_FOV = [67.32, 56.52]
        def rescale_distances_inverse(
            distances,
            fov_new_deg = self.fov_y_deg,
            res_new_px = self.pixels_y,
            fov_old_deg = AMAZON_CAMERA_FOV[1],
            res_old_px = AMAZON_CAMERA_RES[1]
        ):
            # Convert degrees to radians
            fov_old_rad = np.deg2rad(fov_old_deg)
            fov_new_rad = np.deg2rad(fov_new_deg)

            # Compute scaling ratio of pixels spanned
            pixel_ratio = (res_new_px / res_old_px) * (np.tan(fov_old_rad / 2) / np.tan(fov_new_rad / 2))

            if np.isscalar(distances):
                return distances / pixel_ratio
            else:
                return list(np.array(distances) / pixel_ratio)
            
        def rescale_distances(
            distances,
            fov_new_deg = self.fov_y_deg,
            res_new_px = self.pixels_y,
            fov_old_deg = AMAZON_CAMERA_FOV[1],
            res_old_px = AMAZON_CAMERA_RES[1]
        ):
            # Convert degrees to radians
            fov_old_rad = np.deg2rad(fov_old_deg)
            fov_new_rad = np.deg2rad(fov_new_deg)

            # Compute scaling ratio of pixels spanned
            pixel_ratio = (res_new_px / res_old_px) * (np.tan(fov_old_rad / 2) / np.tan(fov_new_rad / 2))
            if np.isscalar(distances):
                return distances * pixel_ratio
            else:
                return list(np.array(distances) * pixel_ratio)
            
        distance_scaled = rescale_distances_inverse(
            distance
        )

        # Loop over bins to find the right one
        for (bin_start, bin_end), params in self.detected_range_uncertainty_params.items():
            if bin_start <= distance_scaled < bin_end:
                mean_inverse_scale, variance_inverse_scale = params['mean'], params['variance']
                new_mean_offset = rescale_distances(
                                mean_inverse_scale,
                            )
                
                range_std = rescale_distances(
                                np.sqrt(variance_inverse_scale),
                            )
                
                range_var = range_std ** 2

                return new_mean_offset, range_var

class Ownship:
    def __init__(self, id):
        self.id = id
        self.cameras = []
        self.position = np.zeros(3)
        self.orientation_rpy_deg = np.zeros(3)

    def reset(self, id):
        self.id = id
        self.position = np.zeros(3)
        self.orientation_rpy_deg = np.zeros(3)

    def set_state(self, position_xyz: Tuple[float, float, float],
                  orientation_rpy_deg: Tuple[float, float, float]):
        self.position = np.array(position_xyz, dtype=float)
        self.orientation_rpy_deg = np.array(orientation_rpy_deg, dtype=float)

    def add_camera(self,
                   camera: Camera,
                   position_xyz: Tuple[float, float, float],
                   orientation_rpy_deg: Tuple[float, float, float]):
        self.cameras.append({
            "camera": camera,
            "position_xyz": np.array(position_xyz),
            "orientation_rpy_deg": np.array(orientation_rpy_deg)
        })

    def list_cameras(self):
        for idx, cam in enumerate(self.cameras):
            print(f"Camera {idx + 1}:")
            print(f"  {cam['camera']}")
            print(f"  Position (m): {cam['position_xyz']}")
            print(f"  Orientation (deg): {cam['orientation_rpy_deg']}")
            print()

    def _rpy_to_matrix(self, roll, pitch, yaw):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                       [0, 1, 0],
                       [np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def map_pixel_views_to_global_cartesian_coords(self, cam_id: int, pixel_range_array: np.ndarray):
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
        pixel_x = pixel_range_array[:, 0]
        pixel_y = pixel_range_array[:, 1]
        horizontal_ranges = pixel_range_array[:, 2]

        # Back-project rays in camera frame
        x_cam = np.ones_like(pixel_x)
        y_cam = (pixel_x - cx) / fx
        z_cam = -(pixel_y - cy) / fy
        rays_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Normalize rays
        rays_cam_normalized = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)

        # Transform rays into global frame direction
        R_cam = self._rpy_to_matrix(*rpy_local)
        R_own = self._rpy_to_matrix(*np.radians(self.orientation_rpy_deg))
        T_own = np.array(self.position)
        cam_global_pos = R_own @ pos_local + T_own
        R_total = R_own @ R_cam

        # Rotate rays into global frame
        rays_global = (R_total @ rays_cam_normalized.T).T  # Shape: (N, 3)

        # Scale rays so their horizontal (XY) length matches the given horizontal range
        horizontal_norms = np.linalg.norm(rays_global[:, :2], axis=1, keepdims=True)
        scaling_factors = horizontal_ranges[:, np.newaxis] / (horizontal_norms + 1e-8)  # Avoid divide by zero
        points_global = cam_global_pos + rays_global * scaling_factors

        return points_global  # Shape: (N, 3)

    def plot(self, fov_length=2.0):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cmap = cm.get_cmap('tab10')
        legend_elements = []
        global_positions = []

        R_ownship = self._rpy_to_matrix(*np.radians(self.orientation_rpy_deg))
        T_ownship = self.position

        for idx, cam in enumerate(self.cameras):
            pos_local = cam["position_xyz"]
            rpy_local = np.radians(cam["orientation_rpy_deg"])
            cam_obj = cam["camera"]
            color = cmap(idx % 10)

            R_cam = self._rpy_to_matrix(*rpy_local)

            # Define FOV corners in camera frame
            dx = fov_length
            dy = dx * np.tan(np.radians(cam_obj.fov_x_deg / 2))
            dz = dx * np.tan(np.radians(cam_obj.fov_y_deg / 2))
            corners = np.array([
                [0, 0, 0],
                [dx, -dy, -dz],
                [dx, dy, -dz],
                [dx, dy, dz],
                [dx, -dy, dz]
            ])

            # Apply camera and ownship rotation and translation
            pos_global = R_ownship @ pos_local + T_ownship
            corners_world = (corners @ R_cam.T @ R_ownship.T) + pos_global
            global_positions.append(pos_global)

            # FOV pyramid faces
            faces = [
                [corners_world[0], corners_world[1], corners_world[2]],
                [corners_world[0], corners_world[2], corners_world[3]],
                [corners_world[0], corners_world[3], corners_world[4]],
                [corners_world[0], corners_world[4], corners_world[1]],
                [corners_world[1], corners_world[2], corners_world[3], corners_world[4]]
            ]
            poly = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='k')
            ax.add_collection3d(poly)

            # Draw direction arrow
            cam_dir_local = np.array([1, 0, 0])  # forward in camera frame
            cam_dir_global = R_ownship @ R_cam @ cam_dir_local
            ax.quiver(*pos_global, *cam_dir_global, length=0.5, color=color, normalize=True)
            ax.scatter(*pos_global, color=color, s=50)

            legend_elements.append((color, f"Camera {idx + 1}"))

        # Plot ownship centroid
        ax.scatter(*self.position, color='black', s=100, label='Ownship')
        
        # Plot ownship orientation (forward direction)
        ownship_dir = self._rpy_to_matrix(*np.radians(self.orientation_rpy_deg)) @ np.array([1, 0, 0])
        ax.quiver(*self.position, *ownship_dir, length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)
        
        # Add 'Ownship' to legend if not already
        legend_elements.append(('black', 'Ownship'))
        

        # Axis scaling based on all global camera positions
        if global_positions:
            all_points = np.vstack(global_positions)
            max_range = np.ptp(all_points, axis=0).max()
            mid = np.mean(all_points, axis=0)
            for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
                axis([m - max_range / 2, m + max_range / 2])

        # Legend
        legend_handles = [Line2D([0], [0], marker='o', color=color, label=label,
                                 markersize=10, linestyle='None') for color, label in legend_elements]
        ax.legend(handles=legend_handles, loc='upper right')

        ax.invert_yaxis()
        ax.set_xlabel("X (forward)")
        ax.set_ylabel("Y (right)")
        ax.set_zlabel("Z (up)")
        ax.set_title("Aircraft Cameras and FOVs")
        ax.grid(True)
        ax.set_box_aspect([1, 1, 1])
        plt.show()

### ENCOUNTER CLASS ####    

class Encounter:
    def __init__(self, ownship: 'Ownship', frame = None, intruders: List['Intruder'] = [], sum_on = True, plot_cartesian_history = False):
        self.frame = frame
        self.ownship = ownship
        self.sum_on = sum_on
        self.intruders = intruders if intruders else []
        self.camera_views = {idx: (x, []) for idx, x in enumerate(ownship.cameras)}
        self.intruder_views = {intr.id: {} for intr in intruders}
        self.encounter_history = {}
        self.plot_cartesian_history = plot_cartesian_history
        self.cartesian_history = {}

        if frame is not None:
            self.create_camera_intruder_views()
            self.update_encounter_history()
            if self.plot_cartesian_history:
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

    def create_camera_intruder_views(self):
        # Determine line of sight intruders
        for intr in self.intruders:
            for cam_idx, cam in enumerate(self.ownship.cameras):
                cam_obj = cam["camera"]
                pos_local = cam["position_xyz"]
                rpy_local = np.radians(cam["orientation_rpy_deg"])
        
                R_cam = self.ownship._rpy_to_matrix(*rpy_local)
                R_own = self.ownship._rpy_to_matrix(*np.radians(self.ownship.orientation_rpy_deg))
                T_own = self.ownship.position
        
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
        self.cartesian_history[self.ownship.id]['position'].append(self.ownship.position)
        self.cartesian_history[self.ownship.id]['orientation'].append(self.ownship.orientation_rpy_deg)
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
    
    def update_encounter_history(self):
        if self.ownship.id not in self.encounter_history:
            # Initialize ownship history
            self.encounter_history[self.ownship.id] = {
                'frame': [],
                'position': [],
                'orientation': []
            }

        # Update ownship state history
        self.encounter_history[self.ownship.id]['frame'].append(self.frame)
        self.encounter_history[self.ownship.id]['position'].append(self.ownship.position)
        self.encounter_history[self.ownship.id]['orientation'].append(self.ownship.orientation_rpy_deg)
        all_intruder_ids = set(self.encounter_history.keys()).union(self.intruder_views.keys())
        all_intruder_ids.discard(self.ownship.id)
        if not all_intruder_ids:
            all_intruder_ids = set()
        
        for intr_id in all_intruder_ids:
            intr_views = self.intruder_views.get(intr_id, {}).values()
            num_detections = len(intr_views)

            if intr_id not in self.encounter_history:
                # Initialize new intruder history
                self.encounter_history[intr_id] = {
                    'frame': [],
                    'cam': [],
                    'pixel_x': [],
                    'pixel_y': [],
                    'range': [],
                    'error_px': [],
                    'error_py': [],
                    'error_range': [],
                    'd1_error_range': [],
                    'd2_error_range': [],
                }

            intr_hist = self.encounter_history[intr_id]

            if num_detections > 0:
                for view in intr_views:
                    for cam_idx, (px, py, rng) in view.items():
                        intr_hist['frame'].append(self.frame)
                        intr_hist['cam'].append(cam_idx)
                        intr_hist['pixel_x'].append(px)
                        intr_hist['pixel_y'].append(py)
                        intr_hist['range'].append(rng)
                        intr_hist['error_px'].append(np.nan)
                        intr_hist['error_py'].append(np.nan)
                        intr_hist['error_range'].append(np.nan)
                        intr_hist['d1_error_range'].append(np.nan)
                        intr_hist['d2_error_range'].append(np.nan)
            else:
                # No detections for this intruder this frame
                intr_hist['frame'].append(self.frame)
                intr_hist['cam'].append(np.nan)
                intr_hist['pixel_x'].append(np.nan)
                intr_hist['pixel_y'].append(np.nan)
                intr_hist['range'].append(np.nan)
                intr_hist['error_px'].append(np.nan)
                intr_hist['error_py'].append(np.nan)
                intr_hist['error_range'].append(np.nan)
                intr_hist['d1_error_range'].append(np.nan)
                intr_hist['d2_error_range'].append(np.nan)

            ##### TRIM FRAMES ########

            # If multiple camera detections for this frame, trim to only single cam
            # TODO: Multiple camera averaging in the future maybe?

            # Find all indices for current frame and where there exists a detection on a camera
            matching_indices = [i for i in range(len(intr_hist['frame'])) if (intr_hist['frame'][i] == self.frame) and not (np.isnan(intr_hist['cam'][i]))]
            # Skip if no detections found
            if not matching_indices:
                continue

            # Sanity Check
            if len(intr_hist['frame']) - 1 not in matching_indices:
                print("LAST INDEX OF FRAME HISTORY NOT IN MATCHING INDICES! ASSUMPTION FAILURE!")
            
            # If there exist multiple detections for this intruder for this frame, trim to only have 1
            if len(matching_indices) > 1:
                # Collect all cameras used in this frame for this intruder
                current_frame_cams = [intr_hist['cam'][i] for i in matching_indices]
                # Set cam for this frame to be last applicable cam if is also in this frame, otherwise use new camera
                cam_idx_for_this_frame = np.nan
                for i in reversed(range(len(intr_hist['frame']))):
                    if intr_hist['frame'][i] < self.frame and not np.isnan(intr_hist['cam'][i]):
                        if intr_hist['cam'][i] in current_frame_cams:
                            cam_idx_for_this_frame = intr_hist['cam'][i]
                        break

                if np.isnan(cam_idx_for_this_frame):
                    # Fallback: choose the first available cam from this frame
                    cam_idx_for_this_frame = current_frame_cams[0]

                # Trim entries in this frame to keep only the selected camera
                indices_to_keep = [
                    i for i in matching_indices if intr_hist['cam'][i] == cam_idx_for_this_frame
                ]

                indices_to_remove = set(matching_indices) - set(indices_to_keep)

                for key in intr_hist:
                    for idx in sorted(indices_to_remove, reverse=True):
                        del intr_hist[key][idx]

    def add_uncertainty_to_intruders(self):
        for intr_idx, intr_hist in self.encounter_history.items():
            if intr_idx == self.ownship.id:
                continue
            
            cam_id_for_this_frame = intr_hist['cam'][-1]
            if np.isnan(cam_id_for_this_frame):
                continue
            
            cam_for_this_frame = self.ownship.cameras[cam_id_for_this_frame]['camera']

            # Sample Pixel Error
            pixel_x_error, pixel_y_error = cam_for_this_frame.pixel_uncertainty_model(gt_dist=intr_hist['range'][-1])
            intr_hist['error_px'][-1] = pixel_x_error
            intr_hist['error_py'][-1] = pixel_y_error

            
            # range_error_input_space_format = {
            #     'history_first_deriv_error': [1, 2, 3],
            #     'history_second_deriv_error': [1],
            #     'history_error': [1, 2, 4],
            #     'history_error_rel': [10],
            #     'history_gt_dist': [0, 5],
            # }

            # Validate format of input data
            range_error_input_space_format = cam_for_this_frame.range_error_model_input_format
            
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
                error_sample_input_array = [gt_range_history, error_history, d1_history, d2_history]

            # Sample range uncertainty for this detection for this frame
            range_error_this, d1_range_error_this, d2_range_error_this = cam_for_this_frame.range_uncertainty_model(input_history = error_sample_input_array)
            intr_hist['error_range'][-1] = range_error_this
            intr_hist['d1_error_range'][-1] = d1_range_error_this
            intr_hist['d2_error_range'][-1] = d2_range_error_this

    def return_camera_views(self):
        return self.camera_views
    
    def detections_this_frame(self, ideal=False):
        detections = {cam_id: [] for cam_id in range(len(self.ownship.cameras))}
        
        if ideal:
            for intr_id in self.intruder_views.keys():
                intr = self.encounter_history[intr_id]
                if intr_id == self.ownship.id:
                    continue
                # Ignore if no detection last frame
                if np.isnan(intr['cam'][-1]):
                    continue
                detection = {
                    'pixel_x': intr['pixel_x'][-1],
                    'pixel_y': intr['pixel_y'][-1],
                    'range': intr['range'][-1],
                    'conf': 1,
                    'w': 100, # TODO
                    'h': 100, # TODO
                    'x_offset': 0, # TODO
                    'y_offset': 0, # TODO
                    'dx': intr['pixel_x'][-1] - intr['pixel_x'][-2] if len(intr['pixel_x']) >= 2 and not np.isnan(intr['pixel_x'][-2]) else np.nan,
                    'dy': intr['pixel_y'][-1] - intr['pixel_y'][-2] if len(intr['pixel_y']) >= 2 and not np.isnan(intr['pixel_y'][-2]) else np.nan,
                    }
                detections[intr['cam'][-1]].append(detection)
        else:
            for intr_id in self.intruder_views.keys():
                intr = self.encounter_history[intr_id]
                if intr_id == self.ownship.id:
                    continue
                # Ignore if no detection last frame
                if np.isnan(intr['cam'][-1]):
                    continue
                detection = {
                    'pixel_x': intr['pixel_x'][-1] + intr['error_px'][-1],
                    'pixel_y': intr['pixel_y'][-1] + intr['error_py'][-1],
                    'range': intr['range'][-1] + intr['error_range'][-1],
                    'conf': 0.8, #TODO
                    'w': 100, # TODO
                    'h': 100, # TODO
                    'x_offset': 0, # TODO
                    'y_offset': 0, # TODO
                    'dx': intr['pixel_x'][-1] - intr['pixel_x'][-2] if len(intr['pixel_x']) >= 2 and not np.isnan(intr['pixel_x'][-2]) else np.nan,
                    'dy': intr['pixel_y'][-1] - intr['pixel_y'][-2] if len(intr['pixel_y']) >= 2 and not np.isnan(intr['pixel_y'][-2]) else np.nan,
                    }
                detections[intr['cam'][-1]].append(detection)

        return detections 

    def plot(self, mode="3d", whole_history = False, fov_length=2.0):
        assert mode in {"3d", "camera_view", "combined"}, "Mode must be '3d', 'camera_view', or 'combined'"
    
        from matplotlib.gridspec import GridSpec
    
        cameras = self.ownship.cameras
        num_cams = len(cameras)
        n_cols = int(np.ceil(np.sqrt(num_cams)))
        n_rows = int(np.ceil(num_cams / n_cols))
    
        if mode == "combined":
            fig = plt.figure(figsize=(6 * n_cols, 6 + 3 * n_rows))
            gs = GridSpec(n_rows + 2, n_cols, figure=fig)  # Reserve upper rows for 3D
            ax3d = fig.add_subplot(gs[:2, :], projection='3d')
        elif mode == "3d":
            fig = plt.figure(figsize=(10, 8))
            ax3d = fig.add_subplot(111, projection='3d')
        elif mode == "camera_view":
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axs = np.array(axs).flatten()
    
        # === 3D Plot ===
        if mode in {"3d", "combined"}:
            cmap = cm.get_cmap('tab10')
            legend_elements = []
            global_positions = []
    
            R_own = self.ownship._rpy_to_matrix(*np.radians(self.ownship.orientation_rpy_deg))
            T_own = self.ownship.position
    
            for idx, cam in enumerate(self.ownship.cameras):
                pos_local = cam["position_xyz"]
                rpy_local = np.radians(cam["orientation_rpy_deg"])
                cam_obj = cam["camera"]
                color = cmap(idx % 10)
    
                R_cam = self.ownship._rpy_to_matrix(*rpy_local)
                pos_global = R_own @ pos_local + T_own
                R_total = R_own @ R_cam
    
                # FOV pyramid
                dx = fov_length
                dy = dx * np.tan(np.radians(cam_obj.fov_x_deg / 2))
                dz = dx * np.tan(np.radians(cam_obj.fov_y_deg / 2))
                corners = np.array([
                    [0, 0, 0],
                    [dx, -dy, -dz],
                    [dx, dy, -dz],
                    [dx, dy, dz],
                    [dx, -dy, dz]
                ])
                corners_world = (corners @ R_cam.T @ R_own.T) + pos_global
                global_positions.append(pos_global)
    
                faces = [
                    [corners_world[0], corners_world[1], corners_world[2]],
                    [corners_world[0], corners_world[2], corners_world[3]],
                    [corners_world[0], corners_world[3], corners_world[4]],
                    [corners_world[0], corners_world[4], corners_world[1]],
                    [corners_world[1], corners_world[2], corners_world[3], corners_world[4]]
                ]
                ax3d.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='k'))
                ax3d.quiver(*pos_global, *(R_total @ [1, 0, 0]), length=0.5, color=color, normalize=True)
                ax3d.scatter(*pos_global, color=color, s=50)
                legend_elements.append((color, f"Camera {idx + 1}"))
    
            # Ownship last frame
            ax3d.scatter(*T_own, color='black', s=100)
            ax3d.quiver(*T_own, *(R_own @ [1, 0, 0]), length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)
            legend_elements.append(('black', 'Ownship'))
    
            intruder_colors = plt.cm.tab10(np.arange(len(self.cartesian_history.keys())))

            if whole_history:
                for idx, obj_hist in self.cartesian_history.items():
                    positions = np.vstack(obj_hist['position'][:-1])
                    orientations = [
                        np.array(o) if o is not None else np.array([np.nan, np.nan, np.nan])
                        for o in obj_hist['orientation'][:-1]
                    ]

                    if len(positions) == 0:
                        continue  # Skip if no history

                    if idx == self.ownship.id:
                        # === Ownship ===
                        ax3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='black', s=100)

                        for pos, orient in zip(positions, orientations):
                            R = self.ownship._rpy_to_matrix(*np.radians(orient))
                            dir_vector = R @ np.array([1, 0, 0])
                            ax3d.quiver(*pos, *dir_vector, length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)

                        legend_elements.append(('black', 'Ownship'))
                    else:
                        # === Intruders ===
                        color = intruder_colors[idx % len(intruder_colors)]

                        ax3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color=color, s=80, marker='X')

                        for pos, orient in zip(positions, orientations):
                            R_intr = self.ownship._rpy_to_matrix(*np.radians(orient))
                            dir_vector = R_intr @ np.array([1, 0, 0])
                            ax3d.quiver(*pos, *dir_vector, length=0.7, color=color, linewidth=2, arrow_length_ratio=0.1, normalize=True)

                        legend_elements.append((color, f"Intruder {idx}"))
            else:
                for i, intr in enumerate(self.intruders):
                    color = intruder_colors[i % 10]
                    ax3d.scatter(*intr.position, color=color, s=80, marker='X', label=f"Intruder {i + 1}")
                    if intr.orientation_rpy_deg is not None:
                        R_intr = self.ownship._rpy_to_matrix(*np.radians(intr.orientation_rpy_deg))
                        ax3d.quiver(*intr.position, *(R_intr @ [1, 0, 0]), length=0.7, color=color, linewidth=2, arrow_length_ratio=0.1)
            
                    legend_elements.append((color, f"Intruder {i + 1}"))

            # LOS rays and red X from camera_views
            for cam_idx, (cam, intr_list) in self.camera_views.items():
                cam_obj = cam["camera"]
                pos_local = cam["position_xyz"]
                rpy_local = np.radians(cam["orientation_rpy_deg"])
                R_cam = self.ownship._rpy_to_matrix(*rpy_local)
                R_total = R_own @ R_cam
                cam_global = R_own @ pos_local + T_own
    
                fx = cam_obj.pixels_x / (2 * np.tan(np.deg2rad(cam_obj.fov_x_deg) / 2))
                fy = cam_obj.pixels_y / (2 * np.tan(np.deg2rad(cam_obj.fov_y_deg) / 2))

                for intr_idx, (pixel_x, pixel_y, depth) in intr_list:
                    y = (pixel_x - cam_obj.pixels_x / 2) * fov_length / fx
                    z = -(pixel_y - cam_obj.pixels_y / 2) * fov_length / fy
                    x = fov_length
                    point_local = np.array([x, y, z])
                    point_global = R_total @ point_local + cam_global
                    intr_this = next(i for i in self.intruders if i.id == intr_idx)
                    intr_pos = intr_this.position
                            
                    if depth < fov_length:
                        ax3d.plot([cam_global[0], point_global[0]], [cam_global[1], point_global[1]], [cam_global[2], point_global[2]], 'r--')
                    else:
                        ax3d.plot([cam_global[0], intr_pos[0]], [cam_global[1], intr_pos[1]], [cam_global[2], intr_pos[2]], 'r--')

                    intr_idx = intr_this.id % min(10, len(intruder_colors))
                    color = intruder_colors[intr_idx]
                    ax3d.scatter(*point_global, color=color, marker='x', s=60, linewidths=2)
    
            if global_positions:
                all_points = np.vstack(global_positions)
                max_range = np.ptp(all_points, axis=0).max()
                mid = np.mean(all_points, axis=0)
                min_range = 1.0  # Minimum size for the bounding box
                max_range = max(np.ptp(all_points, axis=0).max(), min_range)
                zoom_out_factor = 5  # Or more, like 2.0 or 3.0
                for setter, m in zip([ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim], mid):
                    setter([m - zoom_out_factor * max_range / 2, m + zoom_out_factor * max_range / 2])
    
            from matplotlib.lines import Line2D
            seen = set()
            legend_handles = []
            for color, label in legend_elements:
                if label not in seen:
                    style = {'marker': 'o', 'markersize': 10, 'linestyle': 'None'}
                    if label == 'Ownship':
                        style = {'linestyle': '-', 'linewidth': 3}
                    elif label == 'Intruder':
                        style = {'marker': 'X', 'markersize': 10, 'linestyle': 'None'}
                    legend_handles.append(Line2D([0], [0], color=color, label=label, **style))
                    seen.add(label)
    
            ax3d.legend(handles=legend_handles)
            ax3d.invert_yaxis()
            ax3d.set_xlabel("X (forward)")
            ax3d.set_ylabel("Y (right)")
            ax3d.set_zlabel("Z (up)")
            ax3d.set_title("Encounter Visualization")
            ax3d.grid(True)
            ax3d.set_box_aspect([1, 1, 1])
    
        # === 2D Camera View Subplots ===
        if mode in {"camera_view", "combined"}:
            intruder_colors = plt.cm.tab10(np.arange(len(self.encounter_history.keys())))
            if mode == "camera_view":
                axs = np.array([
                    plt.subplot(n_rows, n_cols, i + 1) for i in range(num_cams)
                ])
            else:
                axs = [fig.add_subplot(gs[2 + i // n_cols, i % n_cols]) for i in range(num_cams)]
    
            axs = np.array(axs).flatten()

            if whole_history:
                for idx, cam in enumerate(cameras):
                    cam_obj = cam["camera"]
                    ax = axs[idx]
                    ax.set_title(f'Camera {idx}')
                    ax.set_xlim([0, cam_obj.pixels_x])
                    ax.set_ylim([0, cam_obj.pixels_y])
                    ax.set_aspect('equal')
                    ax.invert_yaxis()
                    ax.set_xlabel("Pixel X")
                    ax.set_ylabel("Pixel Y")
                    ax.grid(True)

                    for intr_id, intr in self.encounter_history.items():
                        if intr_id == self.ownship.id:
                            continue
                        for frame_idx in range(len(intr['frame'])):
                            if intr['cam'][frame_idx] == idx:
                                color = intruder_colors[intr_id]
                                ax.scatter(intr['pixel_x'][frame_idx], intr['pixel_y'][frame_idx], color=color, s=20)
            else:
                for idx, cam in enumerate(cameras):
                    cam_obj = cam["camera"]
                    ax = axs[idx]
                    ax.set_title(f"Camera {idx + 1}")
                    ax.set_xlim(0, cam_obj.pixels_x)
                    ax.set_ylim(0, cam_obj.pixels_y)
                    ax.set_aspect('equal')
                    ax.invert_yaxis()
                    ax.set_xlabel("Pixel X")
                    ax.set_ylabel("Pixel Y")
                    ax.grid(True)
        
                    if idx in self.camera_views:
                        _, intruder_list = self.camera_views[idx]
                        for intr_idx, (px, py, _) in intruder_list:
                            intr_this = next(i for i in self.intruders if i.id == intr_idx)
                            intr_idx = intr_this.id % min(10, len(intruder_colors))
                            color = intruder_colors[intr_idx]
                            ax.plot(px, py, marker='o', color=color, markersize=6, label=f"Intruder {intr_idx + 1}")
    
            for j in range(num_cams, len(axs)):
                fig.delaxes(axs[j])
    
        plt.tight_layout()
        plt.show()

    def plot_perspective(self, mode="combined", fov_length=2.0):
        assert mode in {"3d", "camera_view", "combined"}, "Mode must be '3d', 'camera_view', or 'combined'"
    
        from matplotlib.gridspec import GridSpec
    
        cameras = self.ownship.cameras
        num_cams = len(cameras)
        n_cols = int(np.ceil(np.sqrt(num_cams)))
        n_rows = int(np.ceil(num_cams / n_cols))
    
        if mode == "combined":
            fig = plt.figure(figsize=(6 * n_cols, 6 + 3 * n_rows))
            gs = GridSpec(n_rows + 2, n_cols, figure=fig)  # Reserve upper rows for 3D
            ax3d = fig.add_subplot(gs[:2, :], projection='3d')
        elif mode == "3d":
            fig = plt.figure(figsize=(10, 8))
            ax3d = fig.add_subplot(111, projection='3d')
        elif mode == "camera_view":
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axs = np.array(axs).flatten()
    
        # === 3D Plot ===
        if mode in {"3d", "combined"}:
            cmap = cm.get_cmap('tab10')
            legend_elements = []
            global_positions = []
    
            R_own = self.ownship._rpy_to_matrix(*np.radians(self.ownship.orientation_rpy_deg))
            T_own = self.ownship.position
    
            for idx, cam in enumerate(self.ownship.cameras):
                pos_local = cam["position_xyz"]
                rpy_local = np.radians(cam["orientation_rpy_deg"])
                cam_obj = cam["camera"]
                color = cmap(idx % 10)
    
                R_cam = self.ownship._rpy_to_matrix(*rpy_local)
                pos_global = R_own @ pos_local + T_own
                R_total = R_own @ R_cam
    
                # FOV pyramid
                dx = fov_length
                dy = dx * np.tan(np.radians(cam_obj.fov_x_deg / 2))
                dz = dx * np.tan(np.radians(cam_obj.fov_y_deg / 2))
                corners = np.array([
                    [0, 0, 0],
                    [dx, -dy, -dz],
                    [dx, dy, -dz],
                    [dx, dy, dz],
                    [dx, -dy, dz]
                ])
                corners_world = (corners @ R_cam.T @ R_own.T) + pos_global
                global_positions.append(pos_global)
    
                faces = [
                    [corners_world[0], corners_world[1], corners_world[2]],
                    [corners_world[0], corners_world[2], corners_world[3]],
                    [corners_world[0], corners_world[3], corners_world[4]],
                    [corners_world[0], corners_world[4], corners_world[1]],
                    [corners_world[1], corners_world[2], corners_world[3], corners_world[4]]
                ]
                ax3d.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='k'))
                ax3d.quiver(*pos_global, *(R_total @ [1, 0, 0]), length=0.5, color=color, normalize=True)
                ax3d.scatter(*pos_global, color=color, s=50)
                legend_elements.append((color, f"Camera {idx + 1}"))
    
            # Ownship + intruders
            ax3d.scatter(*T_own, color='black', s=100)
            ax3d.quiver(*T_own, *(R_own @ [1, 0, 0]), length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)
            legend_elements.append(('black', 'Ownship'))

            # Extract x-values of intruders to normalize
            x_vals_intr = np.array([intr.position[0] for intr in self.intruders])
            norm = mcolors.Normalize(vmin=x_vals_intr.min(), vmax=x_vals_intr.max())
            cmap = cm.get_cmap('coolwarm')

            for i, intr in enumerate(self.intruders):
                x = intr.position[0]
                color = cmap(norm(x))
                ax3d.scatter(*intr.position, color=color, s=80, marker='X')
    
            if global_positions:
                all_points = np.vstack(global_positions)
                max_range = np.ptp(all_points, axis=0).max()
                mid = np.mean(all_points, axis=0)
                min_range = 1.0  # Minimum size for the bounding box
                max_range = max(np.ptp(all_points, axis=0).max(), min_range)
                zoom_out_factor = 10  # Or more, like 2.0 or 3.0
                for setter, m in zip([ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim], mid):
                    setter([m - zoom_out_factor * max_range / 2, m + zoom_out_factor * max_range / 2])
    
            seen = set()
            legend_handles = []
            for color, label in legend_elements:
                if label not in seen:
                    style = {'marker': 'o', 'markersize': 10, 'linestyle': 'None'}
                    if label == 'Ownship':
                        style = {'linestyle': '-', 'linewidth': 3}
                    elif label == 'Intruder':
                        style = {'marker': 'X', 'markersize': 10, 'linestyle': 'None'}
                    legend_handles.append(Line2D([0], [0], color=color, label=label, **style))
                    seen.add(label)

            ax3d.legend(handles=legend_handles)
            ax3d.invert_yaxis()
            ax3d.set_xlabel("X (forward)")
            ax3d.set_ylabel("Y (right)")
            ax3d.set_zlabel("Z (up)")
            ax3d.set_title("Encounter Visualization")
            ax3d.grid(True)
            ax3d.set_box_aspect([1, 1, 1])
    
        # === 2D Camera View Subplots ===
        if mode in {"camera_view", "combined"}:
            intruder_colors = plt.cm.tab10(np.arange(len(self.intruders)))
            if mode == "camera_view":
                axs = np.array([
                    plt.subplot(n_rows, n_cols, i + 1) for i in range(num_cams)
                ])
            else:
                axs = [fig.add_subplot(gs[2 + i // n_cols, i % n_cols]) for i in range(num_cams)]
    
            axs = np.array(axs).flatten()
            for idx, cam in enumerate(cameras):
                cam_obj = cam["camera"]
                ax = axs[idx]
                ax.set_title(f"Camera {idx + 1}")
                ax.set_xlim(0, cam_obj.pixels_x)
                ax.set_ylim(0, cam_obj.pixels_y)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.set_xlabel("Pixel X")
                ax.set_ylabel("Pixel Y")
                ax.grid(True)
    
                if idx in self.camera_views:
                    _, intruder_list = self.camera_views[idx]
                    for intr_idx, (px, py, _) in intruder_list:
                        intr = self.intruders[intr_idx]
                        x = intr.position[0]
                        color = cmap(norm(x))
                        ax.plot(px, py, marker='o', color=color, markersize=6)
    
            for j in range(num_cams, len(axs)):
                fig.delaxes(axs[j])
    
        plt.tight_layout()
        plt.show()

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

    def pixels_to_ground(self, cam_pos, cam_ori_rpy_deg, cam_obj, pixels):
        """
        Given pixel coordinates (array), camera position/orientation, and camera specs,
        return 3D points where each pixel's viewing ray intersects the ground plane (z=0).

        Args:
            cam_pos: (3,) array-like, camera position in world frame
            cam_ori_rpy_deg: (3,) tuple/list/array, roll-pitch-yaw in degrees
            fx, fy: focal lengths in pixels
            cx, cy: principal point (image center) in pixels
            pixels: Nx2 numpy array or list of (px, py) pixel coordinates

        Returns:
            ground_points: (N, 3) array of world coordinates (x, y, z=0) for each input pixel
                        or np.nan rows for rays that never intersect z=0 (e.g., pointing upward)
        """

        # Camera intrinsics
        fx = cam_obj.pixels_x / (2 * np.tan(np.radians(cam_obj.fov_x_deg / 2)))
        fy = cam_obj.pixels_y / (2 * np.tan(np.radians(cam_obj.fov_y_deg / 2)))

        cx = cam_obj.pixels_x / 2
        cy = cam_obj.pixels_y / 2

        pixels = np.array(pixels)  # Ensure numpy array of shape (N, 2)
        px = pixels[:, 0]
        py = pixels[:, 1]

        # Convert pixel coordinates to ray directions in camera frame
        x = np.ones_like(px)
        y = (px - cx) / fx
        z = -(py - cy) / fy
        rays_cam = np.stack([x, y, z], axis=1)  # Shape (N, 3)

        # Build rotation matrix (from camera to world frame)
        roll, pitch, yaw = np.radians(cam_ori_rpy_deg)
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
        R = Rz @ Ry @ Rx

        # Rotate rays to world frame
        rays_world = rays_cam @ R.T  # Shape (N, 3)

        # Camera origin
        cam_pos = np.array(cam_pos)

        # Solve for intersection with ground plane z=0 for each ray
        t = -cam_pos[2] / rays_world[:, 2]

        # If ray points upward (ray_world[2] <= 0), no ground intersection
        invalid = (rays_world[:, 2] == 0) | (t < 0)
        ground_points = cam_pos + rays_world * t[:, np.newaxis]

        # Set invalid ones to NaN
        ground_points[invalid] = np.nan

        return ground_points

    def compute_horizon_line(self, cam_pos, R_total, cam_obj, radius=10000, num_points=100):

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
    
        if plot:
            num_cams = len(self.ownship.cameras)
            n_cols = int(np.ceil(np.sqrt(num_cams)))
            n_rows = int(np.ceil(num_cams / n_cols))
            fig = plt.figure(figsize=(6 * n_cols, 6 + 3 * n_rows))
            gs = GridSpec(n_rows + 2, n_cols, figure=fig)  # Reserve upper rows for 3D
            ax3d = fig.add_subplot(gs[:2, :], projection='3d')
            cmap = cm.get_cmap('tab10')
            legend_elements = []
            global_positions = []
            horizon_lines = {}
    
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
    
            R_own = self.ownship._rpy_to_matrix(*np.radians(self.ownship.orientation_rpy_deg))
            T_own = self.ownship.position

            R_cam = self.ownship._rpy_to_matrix(*rpy_local)
            pos_global = R_own @ pos_local + T_own
            R_total = R_own @ R_cam

            if plot:
                horizon_lines[idx] = self.compute_horizon_line(pos_global, R_total, cam_obj)

            ground_coords[idx] = self.pixels_to_ground(pos_global, R.from_matrix(R_total).as_euler('xyz', degrees=True), cam_obj, input_pixels)

            ##################################
            R_own_prev = self.ownship._rpy_to_matrix(*np.radians(self.encounter_history[self.ownship.id]['orientation'][-2]))
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

            ###################################
            if plot:
                # FOV pyramid
                dx = fov_length
                dy = dx * np.tan(np.radians(cam_obj.fov_x_deg / 2))
                dz = dx * np.tan(np.radians(cam_obj.fov_y_deg / 2))
                corners = np.array([
                    [0, 0, 0],
                    [dx, -dy, -dz],
                    [dx, dy, -dz],
                    [dx, dy, dz],
                    [dx, -dy, dz]
                ])
                corners_world = (corners @ R_cam.T @ R_own.T) + pos_global
                global_positions.append(pos_global)
    
                faces = [
                    [corners_world[0], corners_world[1], corners_world[2]],
                    [corners_world[0], corners_world[2], corners_world[3]],
                    [corners_world[0], corners_world[3], corners_world[4]],
                    [corners_world[0], corners_world[4], corners_world[1]],
                    [corners_world[1], corners_world[2], corners_world[3], corners_world[4]]
                ]

                color = cmap(idx % 10)

                ax3d.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='k'))
                ax3d.quiver(*pos_global, *(R_total @ [1, 0, 0]), length=0.5, color=color, normalize=True)
                ax3d.scatter(*pos_global, color=color, s=50)
                legend_elements.append((color, f"Camera {idx + 1}"))

        if plot:
            # Ownship
            ax3d.scatter(*T_own, color='black', s=100)
            ax3d.quiver(*T_own, *(R_own @ [1, 0, 0]), length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)
            legend_elements.append(('black', 'Ownship'))

            for cam, coord_list in ground_coords.items():
                for coord in coord_list:
                    if not np.any(np.isnan(coord)):
                        ax3d.scatter(*coord, color='red', s=20, marker='o')

            # Intruders
            intruder_colors = plt.cm.tab10(np.arange(len(self.intruders)))

            for i, intr in enumerate(self.intruders):
                color = intruder_colors[i % 10]
                ax3d.scatter(*intr.position, color=color, s=80, marker='X', label=f"Intruder {i + 1}")
                if intr.orientation_rpy_deg is not None:
                    R_intr = self.ownship._rpy_to_matrix(*np.radians(intr.orientation_rpy_deg))
                    ax3d.quiver(*intr.position, *(R_intr @ [1, 0, 0]), length=0.7, color=color, linewidth=2, arrow_length_ratio=0.1)
            
                legend_elements.append((color, f"Intruder {i + 1}"))

            if global_positions:
                all_points = np.vstack(global_positions)
                max_range = np.ptp(all_points, axis=0).max()
                mid = np.mean(all_points, axis=0)
                min_range = 1.0  # Minimum size for the bounding box
                max_range = max(np.ptp(all_points, axis=0).max(), min_range)
                zoom_out_factor = 10  # Or more, like 2.0 or 3.0
                for setter, m in zip([ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim], mid):
                    setter([m - zoom_out_factor * max_range / 2, m + zoom_out_factor * max_range / 2])
    
            seen = set()
            legend_handles = []
            for color, label in legend_elements:
                if label not in seen:
                    style = {'marker': 'o', 'markersize': 10, 'linestyle': 'None'}
                    if label == 'Ownship':
                        style = {'linestyle': '-', 'linewidth': 3}
                    elif label == 'Intruder':
                        style = {'marker': 'X', 'markersize': 10, 'linestyle': 'None'}
                    legend_handles.append(Line2D([0], [0], color=color, label=label, **style))
                    seen.add(label)

            ax3d.legend(handles=legend_handles)
            ax3d.invert_yaxis()
            ax3d.set_xlabel("X (forward)")
            ax3d.set_ylabel("Y (right)")
            ax3d.set_zlabel("Z (up)")
            ax3d.set_title("Encounter Visualization")
            ax3d.grid(True)
            ax3d.set_box_aspect([1, 1, 1])
    
            # === 2D Camera View Subplots ===
            axs = [fig.add_subplot(gs[2 + i // n_cols, i % n_cols]) for i in range(num_cams)]
        
            axs = np.array(axs).flatten()
            for idx, cam in enumerate(self.ownship.cameras):
                cam_obj = cam["camera"]
                ax = axs[idx]
                ax.set_title(f"Camera {idx + 1}")
                ax.set_xlim(0, cam_obj.pixels_x)
                ax.set_ylim(0, cam_obj.pixels_y)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.set_xlabel("Pixel X")
                ax.set_ylabel("Pixel Y")
                ax.grid(True)

                ax.plot(*zip(*horizon_lines[idx]), linestyle='--', color='orange')

                if idx in self.camera_views:
                    ground_coord_dict = frame_pair_ground_pixels[idx]
                    for px, py in ground_coord_dict['current_frame']:
                        ax.plot(px, py, marker='o', color='red', markersize=6)
                    for px, py in ground_coord_dict['prev_frame']:
                        ax.plot(px, py, marker='x', color='green', markersize=6)
                    if affine_trans_matrices[idx] is not None:
                        for px, py in ground_coord_dict['prev_frame']:
                            px_new, py_new = (affine_trans_matrices[idx] @ np.array([px, py, 1]))[:2]
                            ax.plot(px_new, py_new, marker='x', color='blue', markersize=6)

                if idx in self.camera_views:
                    _, intruder_list = self.camera_views[idx]
                    for intr_idx, (px, py, _) in intruder_list:
                        intr_this = next(i for i in self.intruders if i.id == intr_idx)
                        intr_idx = intr_this.id % min(10, len(intruder_colors))
                        color = intruder_colors[intr_idx]
                        ax.plot(px, py, marker='o', color=color, markersize=6, label=f"Intruder {intr_idx + 1}")
    
            for j in range(num_cams, len(axs)):
                fig.delaxes(axs[j])
        
            plt.tight_layout()
            plt.show()   

        return affine_trans_matrices     

### TRACKER CLASS ####   

class Tracker:
    def __init__(self, ownship: 'Ownship', script_dir = "."):
        self.ownship = ownship
        self.cfg = self.load_config_data("config", script_dir + "/config")
        self.detection_tracking_params = DotDict(self.cfg["DETECTION_AND_TRACKING_PARAMS"])
        self.trackers = {cam_id: tracking.SimpleOffsetTracker(
            min_track_size = self.detection_tracking_params.MIN_TRACK_LENGTH,
            threshold_to_find = self.detection_tracking_params.THRESHOLD_TO_FIND,
            threshold_to_continue = self.detection_tracking_params.THRESHOLD_TO_CONTINUE,
            threshold_distance = self.detection_tracking_params.MAX_COST,
            min_distance = self.detection_tracking_params.MAX_DISTANCE) for cam_id in range(len(ownship.cameras))}

        self.track_history = {cam_id: {} for cam_id in range(len(self.ownship.cameras))}
        self.range_kalman_filters = {cam_id: {} for cam_id in range(len(self.ownship.cameras))}  # track_id → KalmanFilter for 1d range smoothing
        self.xyz_kalman_filters = {cam_id: {} for cam_id in range(len(self.ownship.cameras))}  # track_id → KalmanFilter for 3D position/velocity smoothing

    def load_config_data(self, config_file: str, config_dir='config') -> dict:
        with open(f"{config_dir}/{config_file}.yaml") as f:
            cfg: dict = yaml.load(f, Loader=yaml.FullLoader)

        print(f"Loaded config file: {config_dir}/{config_file}.yaml")

        return cfg

    def update_tracks(self, time, transform_matrices = None, detections_this_frame=None, frame=None):
        new_track_items = {}
        if not detections_this_frame:
            #TODO: Add coasting logic here
            return None
        else:
            for cam_id, detections in detections_this_frame.items():
                this_cam_detections = []
                if transform_matrices:
                    AFMat_this = transform_matrices[cam_id]
                else:
                    AFMat_this = None
                for detection in detections:
                    if detection['conf'] >= self.trackers[cam_id].threshold_to_continue:
                        
                        this_cam_detections.append(DetectedItem(
                            cx=detection['pixel_x'], # TODO: I REMOVED THE OFFSET TERM?????
                            cy=detection['pixel_y'], # TODO: I REMOVED THE OFFSET TERM?????
                            w=detection['w'],
                            h=detection['h'],
                            distance=detection['range'],
                            above_horizon=0.0,   # None
                            confidence=detection['conf'],
                            primary_conf=1,   # None
                            track_id=-1,
                            dx=detection['dx'],
                            dy=detection['dy'],
                            item_id='',
                            cls_idx = 0,   # None
                            cam_id = cam_id
                        ))

                filtered_track_items = self.trackers[cam_id].process_frame_detections(detections=this_cam_detections, transform=AFMat_this)

                if filtered_track_items:

                    for it in filtered_track_items:
                        new_track_items[f"cam_{cam_id}_trk_{it.track_id}"] =  dict(
                                time=time,
                                cx=it.cx,
                                cy=it.cy,
                                w=it.w,
                                h=it.h,
                                conf=it.confidence,
                                primary_conf = it.primary_conf,
                                offset=[0.0, 0.0],
                                distance = it.distance,
                                above_horizon = it.above_horizon,
                                cls_idx=it.cls_idx,
                                dx=it.dx,
                                dy=it.dy,
                                abs_cartesian_x_y_z = self.ownship.map_pixel_views_to_global_cartesian_coords(cam_id, np.array([[it.cx, it.cy, it.distance]]))[0]
                            )    

                    # Post_Process: Apply Kalman Smoothing and Calcaulted Heading and Velocity
                    for track_id, item in new_track_items.items():
                        ###################################################
                        ############# Range Smoothing #####################
                        ###################################################
                        raw_range = item['distance']
                        range_mean_offset, range_var = self.ownship.cameras[cam_id]['camera'].get_detected_range_uncertainty_params(raw_range)
                        raw_range -= range_mean_offset

                        # Initialize Range Kalman filter for this track if it doesn't exist
                        if track_id not in self.range_kalman_filters[cam_id]:
                            dt = 0.05
                            self.range_kalman_filters[cam_id][track_id] = self.create_range_kalman_filter(raw_range, dt, item['time'])
                        else:
                            dt = item['time'] - self.range_kalman_filters[cam_id][track_id].prev_timestamp

                        kf_range = self.range_kalman_filters[cam_id][track_id]

                        # Update timestep for range kalman filter
                        kf_range.F = np.array([
                                                [1, dt],
                                                [0, 1]
                                            ])

                        # Input variance of distance into Kalman filter
                        kf_range.R = np.array([[range_var]])

                        kf_range.predict()
                        kf_range.update(np.array([raw_range]))

                        # Update prev_timestep for range kalman filter
                        kf_range.prev_timestamp = item['time']

                        smoothed_range = kf_range.x[0, 0]

                        item['distance_smoothed'] = smoothed_range
                        item['distance_variance'] = range_var

                        # Recalculate smoothed 3D position using original cx, cy but smoothed range
                        pixel_pos = np.array([[item['cx'], item['cy'], smoothed_range]])
                        smoothed_xyz = self.ownship.map_pixel_views_to_global_cartesian_coords(cam_id, pixel_pos)

                        item['abs_cartesian_x_y_z_smoothed'] = smoothed_xyz.flatten()

                        ###################################################################
                        ########### XYZ smoothing + Velocity + Heading Estimation #########
                        ###################################################################

                        # Initialize XZY Kalman filter for this track if it doesn't exist
                        if track_id not in self.xyz_kalman_filters[cam_id]:
                            dt = 0.05
                            self.xyz_kalman_filters[cam_id][track_id] = self.create_xyz_kalman_filter(item['abs_cartesian_x_y_z_smoothed'], dt, item['time'])
                        else:
                            dt = item['time'] - self.xyz_kalman_filters[cam_id][track_id].prev_timestamp

                        kf_xyz = self.xyz_kalman_filters[cam_id][track_id]

                        # Update timestep for range kalman filter
                        kf_xyz.F = np.array([
                                                [1, 0, 0, dt, 0, 0],
                                                [0, 1, 0, 0, dt, 0],
                                                [0, 0, 1, 0, 0, dt],
                                                [0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1]
                                            ])

                        kf_xyz.predict()
                        kf_xyz.update(smoothed_xyz.flatten())

                        # Update prev_timestep for range kalman filter
                        kf_xyz.prev_timestamp = item['time']

                        # Extract Intruder State
                        x, y, z, vx, vy, vz = kf_xyz.x[:, 0]

                        # Save Smoothed State
                        item['velocity_x_y_z_smoothed'] = np.array([vx, vy, vz])
                        #item['abs_cartesian_x_y_z_smoothed'] = np.array([x, y, z])

                        # Compute and saves smoothed track heading
                        trk_rad = np.arctan2(vy, vx)
                        trk_deg = (np.degrees(trk_rad) + 360) % 360
                        item['trk_smoothed'] = trk_deg

                        # Save to track history
                        if track_id not in self.track_history[cam_id]:
                            self.track_history[cam_id][track_id] = []
                        self.track_history[cam_id][track_id].append(item)

        return new_track_items
    
    def create_range_kalman_filter(self, initial_range, dt, initial_timestamp = 0):
        kf = KalmanFilter(dim_x=2, dim_z=1)

        # State: [range, range_rate]
        kf.F = np.array([
            [1, dt],
            [0, 1]
        ])
        kf.H = np.array([[1, 0]])

        kf.P *= 800
        kf.R *= 3000   # Measurement noise (adjustable)
        kf.Q *= 2 * 2 * 0.0001 * 0.05**2  # Process noise (adjustable)

        kf.x[0, 0] = initial_range
        kf.x[1, 0] = 0  # Assume zero initial rate of change

        kf.prev_timestamp = initial_timestamp

        return kf
    
    def create_xyz_kalman_filter(self, initial_xyz, dt, initial_timestamp = 0):
        kf = KalmanFilter(dim_x=6, dim_z=3)  # [x, y, z, vx, vy, vz]

        # State transition model
        kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Measurement model: we observe x, y, z directly
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Initial state: position from measurement, velocity assumed 0
        kf.x[:3, 0] = initial_xyz
        kf.x[3:, 0] = 0.0

        # Covariances (adjust as needed)
        kf.P *= 500
        kf.R *= 3000     # Measurement noise
        kf.Q *= 0.0001 * 0.05**2  # Process noise

        kf.prev_timestamp = initial_timestamp

        return kf

# Example usage
if __name__ == "__main__":
    '''
    # Camera spec: 61° horizontal FOV, 50° vertical FOV
    cam_fov = (61.0, 50.0)
    cam_res = (2448, 2048)
    
    # Create 6 identical cameras
    cams = [Camera(cam_res, cam_fov) for _ in range(6)]
    
    # Ownship setup
    ownship = Ownship()
    ownship.set_state((10.0, 20.0, -5.0), (0.0, 0.0, 45.0))  # position + orientation
    
    # Mount cameras around a full 360° with slight overlaps
    for i, cam in enumerate(cams):
        azimuth_deg = i * 60  # 0, 60, 120, ..., 300
        position = (0.0, 0.0, 0.0)  # All at same origin relative to ownship
        orientation = (0.0, 0.0, azimuth_deg)  # Facing outward in yaw
        ownship.add_camera(cam, position, orientation)

    ownship.plot()
    
    intr1 = Intruder(1, (12.0, 22.0, -6.0))  # No heading
    intr2 = Intruder(2, (11.0, 21.5, -5.0), orientation_rpy_deg=(0.0, 0.0, 135.0))  # With heading
    intr3 = Intruder(3, (15.0, 25.5, -5.0), orientation_rpy_deg=(0.0, 0.0, 135.0))  # With heading
    
    enc = Encounter(ownship, [intr1, intr2, intr3])
    enc.create_camera_intruder_views()
    print(enc.return_camera_views())
    enc.plot(mode="3d")           # Just 3D
    enc.plot(mode="camera_view") # Just 2D subplots
    enc.plot(mode="combined")    # Stacked layout
    
    '''

    # === Setup cameras and ownship ===
    cam_fov = (61.0, 50.0)
    cam_res = (2448, 2048)
    cams = [Camera(cam_res, cam_fov) for _ in range(6)]
    
    ownship = Ownship('OS1')
    ownship.set_state((0.0, 0.0, 1.0), (0.0, -15.0, 0.0))
    own_pos = np.array(ownship.position)
    
    intersecting_rays = []

    for i, cam in enumerate(cams):
        azimuth_deg = i * 60
        ownship.add_camera(cam, (0.0, 0.0, 0.0), (0.0, 0.0, azimuth_deg))
        

    # === Intruder circling setup ===
    radius = 5.0
    z_alt = 0.0
    angle = -10 + 3.5 + 180/6
    x = radius * np.cos(np.radians(angle))
    y = radius * np.sin(np.radians(angle))
    intruder = Intruder(idx = 1, position_xyz=(x, y, z_alt))
    intruder2 = Intruder(idx = 2, position_xyz=(x, y, z_alt + 1))
    intruder3 = Intruder(idx = 3, position_xyz=(x, y, z_alt - 1 + 2 * np.sin(10* np.radians(angle))))
    
    # === Animate ===
    num_frames = 2
    angular_speed = 10 * 5 / 100  # full circle in num_frames steps
    
    # plt.ion()
    ownship_states = [((0.0, 0.0, 1.0), (0.0, -15.0, 0.0)),((0.0, 0.0, 1.0), (0.0, -15.0, 0.0))]

    enc = Encounter(ownship, sum_on = False, plot_cartesian_history = True)
    tracker = Tracker(ownship)
    enc.update_encounter(0, ownship = ownship, intruders = [intruder, intruder2, intruder3])
    raw_detections = enc.detections_this_frame(ideal=True)
    AFMats = enc.generate_affine_transform_matrices(plot = False)
    tracker.update_tracks(ownship, AFMats, raw_detections)
    for t in range(1, num_frames):
        print(t)
        ownship.set_state(ownship_states[1][0], ownship_states[1][1])
        # Circular intruder motion
        angle += angular_speed
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))

        intruder.update_state(position_xyz = (x, y, z_alt),
                  orientation_rpy_deg = (0.0, 0.0, 0.0))
        intruder2.update_state(position_xyz = (x, y, z_alt + 1),
                  orientation_rpy_deg = (0.0, 0.0, 0.0))
        intruder3.update_state(position_xyz = (x, y, z_alt - 1 + 2 * np.sin(10* np.radians(angle))),
                  orientation_rpy_deg = (0.0, 0.0, 0.0))
    
        # Create encounter + plot
        enc.update_encounter(t, ownship = ownship, intruders = [intruder, intruder2, intruder3])
        enc.plot(mode = 'combined', whole_history=False)
        
        # plt.close()
        #enc.plot(mode="combined")
        
        #plt.pause(0.5)
        # start_time = time.time()
        AFMats = enc.generate_affine_transform_matrices(plot = False)
        raw_detections = enc.detections_this_frame(ideal=True)

        tracker.update_tracks(ownship, AFMats, raw_detections)

    #enc.plot(mode = 'combined', whole_history=True)

    print(tracker.track_history)

    def get_track_id_color_map(data):
        all_track_ids = set()
        for cam_tracks in data.values():
            for frame_items in cam_tracks:
                for item in frame_items:
                    all_track_ids.add(item['track_id'])
        all_track_ids = sorted(list(all_track_ids))
        cmap = cm.get_cmap('tab20', len(all_track_ids))
        track_id_to_color = {track_id: cmap(i) for i, track_id in enumerate(all_track_ids)}
        return track_id_to_color, all_track_ids

    def plot_3d_tracks(data):
        track_id_to_color, all_track_ids = get_track_id_color_map(data)

        fig = plt.figure(figsize=(8, 6))
        ax3d = fig.add_subplot(111, projection='3d')

        num_cams = len(enc.ownship.cameras)
        cmap = cm.get_cmap('tab10')
        legend_elements = []
        global_positions = []
        horizon_lines = {}

        for idx, cam in enumerate(enc.ownship.cameras):
            # Camera Positioning Specs
            pos_local = cam["position_xyz"]
            rpy_local = np.radians(cam["orientation_rpy_deg"])
            
            cam_obj = cam["camera"]
    
            R_own = enc.ownship._rpy_to_matrix(*np.radians(enc.ownship.orientation_rpy_deg))
            T_own = enc.ownship.position

            R_cam = enc.ownship._rpy_to_matrix(*rpy_local)
            pos_global = R_own @ pos_local + T_own
            R_total = R_own @ R_cam
            
            ###################################
            # FOV pyramid
            dx = 2
            dy = dx * np.tan(np.radians(cam_obj.fov_x_deg / 2))
            dz = dx * np.tan(np.radians(cam_obj.fov_y_deg / 2))
            corners = np.array([
                [0, 0, 0],
                [dx, -dy, -dz],
                [dx, dy, -dz],
                [dx, dy, dz],
                [dx, -dy, dz]
            ])
            corners_world = (corners @ R_cam.T @ R_own.T) + pos_global
            global_positions.append(pos_global)

            faces = [
                [corners_world[0], corners_world[1], corners_world[2]],
                [corners_world[0], corners_world[2], corners_world[3]],
                [corners_world[0], corners_world[3], corners_world[4]],
                [corners_world[0], corners_world[4], corners_world[1]],
                [corners_world[1], corners_world[2], corners_world[3], corners_world[4]]
            ]

            color = cmap(idx % 10)

            ax3d.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='k'))
            ax3d.quiver(*pos_global, *(R_total @ [1, 0, 0]), length=0.5, color=color, normalize=True)
            ax3d.scatter(*pos_global, color=color, s=50)
            legend_elements.append((color, f"Camera {idx + 1}"))

        # Ownship
        ax3d.scatter(*T_own, color='black', s=100)
        ax3d.quiver(*T_own, *(R_own @ [1, 0, 0]), length=1.0, color='black', linewidth=3, arrow_length_ratio=0.1, normalize=True)
        legend_elements.append(('black', 'Ownship'))

        for cam_tracks in data.values():
            for frame_items in cam_tracks:
                for item in frame_items:
                    xyz = item.get('abs_cartesian_x_y_z_smoothed')
                    if xyz is not None:
                        x, y, z = xyz.flatten()
                        color = track_id_to_color.get(item['track_id'], 'black')
                        ax3d.scatter(x, y, z, color=color, s=20)

        ax3d.set_title('3D Cartesian Track Plot (X-Y-Z)')
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.grid(True)

        # Equal aspect ratio
        x_limits = ax3d.get_xlim3d()
        y_limits = ax3d.get_ylim3d()
        z_limits = ax3d.get_zlim3d()

        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]
        max_range = max(x_range, y_range, z_range)

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax3d.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax3d.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax3d.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=track_id_to_color[tid],
                    label=f'Track {tid}', markersize=8)
            for tid in all_track_ids
        ]
        ax3d.legend(handles=handles, loc='upper right')

        plt.tight_layout()

    def plot_2d_camera_views(data):
        track_id_to_color, all_track_ids = get_track_id_color_map(data)

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()

        for idx in range(6):
            ax = axes[idx]
            if idx in data:
                frames = data[idx]
                for frame_items in frames:
                    for item in frame_items:
                        cx, cy = item['cx'], item['cy']
                        track_id = item['track_id']
                        color = track_id_to_color.get(track_id, 'black')
                        ax.scatter(cx, cy, color=color, s=20)

            ax.set_title(f'Camera {idx}')
            ax.set_xlabel('cx')
            ax.set_ylabel('cy')
            ax.set_xlim([0, 2448])
            ax.set_ylim([0, 2048])
            ax.invert_yaxis()
            ax.grid(True)

        # Global legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=track_id_to_color[tid],
                    label=f'Track {tid}', markersize=8)
            for tid in all_track_ids
        ]
        fig.legend(handles=handles, loc='upper center', ncol=max(1, len(all_track_ids) // 2))

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_3d_tracks(tracker.track_history)
    plot_2d_camera_views(tracker.track_history)
    plt.show()
    # print(time.time() - start_time)

    # plt.ioff()
    # plt.close()

##########

# # Initialise Ownship
# ownship_id = "Ownship_1"
# ownship = Ownship(ownship_id)

# # Define Camera Parameters
# cam_fovs = [
#     (61.76, 48.26), # Front-Left
#     (68.4,  53.96), # Side-Left
#     (76.4,  61.04), # Rear-Left
#     (61.76, 48.26), # Front-Right
#     (68.4,  53.96), # Side-Right
#     (76.4,  61.04) # Rear-Right
# ]

# cam_res_new = (9344, 7000)

# cam_positions = [(0.0, 0.0, 0.0),
#                  (0.0, 0.0, 0.0),
#                  (0.0, 0.0, 0.0),
#                  (0.0, 0.0, 0.0),
#                  (0.0, 0.0, 0.0),
#                  (0.0, 0.0, 0.0)]

# cam_orientations = [(0.0, 0.0, 0.0),
#                     (0.0, 0.0, 60.0),
#                     (0.0, 0.0, 120.0),
#                     (0.0, 0.0, 180.0),
#                     (0.0, 0.0, 240.0),
#                     (0.0, 0.0, 300.0)]

# # Create Dictionary of Cameras
# cameras = {
#             i: 
#                 {
#                     'res': cam_res_new,
#                     'fov': cam_fovs[i],
#                     'pos': cam_positions[i],
#                     'orientation': cam_orientations[i],
#                     'cam': Camera(cam_res_new, cam_fovs[i])
#                 }
#             for i in range(6)
#         }

# # Initialise Camera Uncertainty Models with Camera Parameters and add to Ownship
# for camera in cameras.values():
#     ownship.add_camera(camera['cam'], camera['pos'], camera['orientation'])

# enc_this = Encounter(ownship, sum_on = False)
# enc_this.plot(mode="combined")
