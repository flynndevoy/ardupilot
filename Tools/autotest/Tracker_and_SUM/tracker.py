# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:06:34 2025

@author: adamg
"""

import numpy as np
from typing import *
from Tracker_and_SUM import tracking
from Tracker_and_SUM.prediction_structure_v2 import *
import yaml
from Tracker_and_SUM.common_utils import DotDict
import navpy
from filterpy.kalman import KalmanFilter
from Tracker_and_SUM.ownship_camera_classes import *
from Tracker_and_SUM.sum_functions import *
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
from matplotlib.patches import Wedge
from scipy.spatial.distance import cdist
import copy
import os
import time
from scipy.stats import chi2
from collections import OrderedDict
# from scipy.spatial.distance import mahalanobis

### TRACKER CLASS ####   
class Tracker:
    def __init__(self, ownship_id = "OS", script_dir = "."):
        self.ownship = Ownship(ownship_id)
        self.camera_cfg = self.load_camera_config_data(config_dir=script_dir)
        self.ownship.mount_cameras_from_config_file(camera_configs=self.camera_cfg, script_dir=script_dir)
        self.cartesian_centered_aligned = [
            self.camera_cfg['daidalus_input_coord_frame']['cartesian'],
            self.camera_cfg['daidalus_input_coord_frame']['centered'],
            self.camera_cfg['daidalus_input_coord_frame']['aligned']
            ]
        self.home_lat_lon = None
        self.trackers = {cam_id: tracking.SimpleOffsetTracker(
            min_track_size = 10,
            threshold_to_find = 0.3,
            threshold_to_continue = 0.2,
            threshold_distance = 200,
            min_distance = 20000,
            max_frames_offset = 3) for cam_id in range(len(self.ownship.cameras))}

        self.track_history = {}
        self.track_history_length = 100
        self.xyz_kalman_filters = {"EO": {}, "ADSB": {}, "Ownship": {}}
        self.pos_cov_boost = 0
        self.iso_decomp_guidance_hist = {}
        self.coasting_start_thresh_EO = 0.2 # sec (how long missing detections before coasting start and coasting update frequency)
        self.coasting_start_thresh_ADSB = 5.0 # sec (how long missing ADSB before coasting start and coasting update frequency)
        self.track_establishment_age = 10.0 # sec (min age of track before coasting permitted)
        self.track_termination_sec = [2.5, 5.0, 5.0, 10.0] # Sec of missed telemetry before track terminated [before establishment time EO, after establishment time EO, before establishment time ADSB, after establishment time ADSB]
        self.enforce_single_EO_intruder = True

        # For Plotting kalman filter updates
        self.plot = False
        self.plot_ADSB_instead_of_eo = False
        self.fused_xy_fig = None
        self.plot_save_path = None
        self.plot_last_saved_second = None
        self.plot_update_count = 0
        self.fused_xy_ax = None
        self.fused_xy_path = None
        self.fused_xy_point = None
        self.intr_gt_path = None
        self.intr_xy_gt_point = None
        self.fused_xy_uncertainty = None
        self.measured_xy_uncertainty = None
        self.fused_xy_history = []
        self.intr_xy_history = []
        self.min_xy_separation = [np.inf, np.inf]
        self.prev_cam_id_set = None  # in __init__ or first call
        self.track_color_idx = 0
        self.track_colors = plt.get_cmap('tab10').colors  # or any other colormap
        self.speed_history = []
        self.accel_history = []
        self.accel2_history = []
        self.time_history = []
        self.gt_speed_history = []
        self.speed_ax = None
        self.accel_ax = None
        self.speed_est_line = None
        self.speed_gt_line = None
        self.accel_est_line = None
        self.debug_param_history = []
        self.debug_params_lines = []
        self.fused_vel_path_gt = None
        self.daidalus_constituent_build_count = 0
        self.daidalus_constituent_build_total_s = 0.0
        self.daidalus_constituent_build_max_s = 0.0
        self.daidalus_timing_file_initialized = False

    def save_tracker_plot(self, path, dpi=150):
        if self.fused_xy_fig is None:
            return
        self.fused_xy_fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=False)

    def tracker_plot_frame_path(self, path, second):
        suffix = f'_t{second:04d}s.png'
        if path.endswith('_tracker_spheres.png'):
            return path.replace('_tracker_spheres.png', f'_tracker_spheres{suffix}')
        if path.endswith('.png'):
            return path[:-4] + suffix
        return path + suffix

    def append_daidalus_timing(self, fused_time, generated_count, last_ms, avg_ms, max_ms):
        if self.plot_save_path:
            timing_path = os.path.join(os.path.dirname(self.plot_save_path), 'timing.md')
        else:
            timing_path = 'timing.md'

        if not self.daidalus_timing_file_initialized:
            with open(timing_path, 'w', encoding='utf-8') as f:
                f.write('# DAIDALUS Constituent Timing\n\n')
                f.write('| build | sim_time_s | generated | last_ms | avg_ms | max_ms |\n')
                f.write('|---:|---:|---:|---:|---:|---:|\n')
            self.daidalus_timing_file_initialized = True

        with open(timing_path, 'a', encoding='utf-8') as f:
            f.write(
                f'| {self.daidalus_constituent_build_count} '
                f'| {fused_time:.3f} '
                f'| {generated_count} '
                f'| {last_ms:.3f} '
                f'| {avg_ms:.3f} '
                f'| {max_ms:.3f} |\n'
            )

    def load_config_data(self, config_file: str, config_dir='config') -> dict:
        with open(f"{config_dir}/{config_file}.yaml") as f:
            cfg: dict = yaml.load(f, Loader=yaml.FullLoader)

        print(f"Loaded config file: {config_dir}/{config_file}.yaml")

        return cfg
    
    def load_camera_config_data(self, config_file="cameras_config", config_dir='.'):
        # === Load YAML Config ===
        config_path = os.path.join(config_dir, config_file + ".yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        camera_configs = config["configs"]

        return camera_configs
    
    #####################################
    ###### Update Tracker Helpers #######
    #####################################

    def nic_to_containment_radius(self, nic: int) -> float:
        """
        Convert ADS-B NIC (Navigation Integrity Category) to containment radius in metres.
        Based on RTCA DO-260B / ICAO Annex 10 definitions.
        
        Args:
            nic (int): NIC value (0–13)
        
        Returns:
            float | None: Containment radius in metres, or None if undefined.
        """
        nm_to_m = 1852.0  # nautical mile to metre

        # Lookup table in NM (from DO-260B NIC table)
        nic_to_nm = {
            0: 20.0,        # >20 NM, essentially undefined, defaulted to 20 here as the fallback. Can be tuned
            1: 20.0,
            2: 8.0,
            3: 4.0,
            4: 2.0,
            5: 1.0,
            6: 0.6,
            7: 0.2,
            8: 0.1,
            9: 0.04,
            10: 0.013,
            11: 0.004,
        }

        nm_value = nic_to_nm.get(nic, None)
        if nm_value is None:
            return None
        return nm_value * nm_to_m
    
    def nacp_to_horizontal_variance(self, nacp: int, fallback_nacp = 6) -> float:
        """
        Convert ADS-B NACp (Navigation Accuracy Category - Position)
        to a 1-sigma variance (metres^2) for horizontal position error.

        Uses 95% EPU bounds from RTCA DO-260B.

        Args:
            nacp (int): NACp value (0–11)

        Returns:
            float | None: Horizontal variance (sigma^2), or None if undefined.
        """

        # 95% error bounds (EPU95) in metres
        nacp_to_epu95_m = {
            0: 18520.0,     # ≥ 10 NM
            1: 18520.0,
            2: 7408.0,      # 4 NM
            3: 3704.0,      # 2 NM
            4: 1852.0,      # 1 NM
            5: 926.0,       # 0.5 NM
            6: 555.6,       # 0.3 NM
            7: 185.2,       # 0.1 NM
            8: 92.6,        # 0.05 NM
            9: 30.0,        # EPU < 30 m
            10: 10.0,       # <10 m
            11: 3.0,        # <3 m
        }

        epu95 = nacp_to_epu95_m.get(nacp, nacp_to_epu95_m[fallback_nacp])

        # Convert 95% 2D radius to 1-sigma std deviation
        sigma = epu95 / 2.4477

        # Return variance
        return sigma ** 2


    def compute_damping_measurement_jitter(
            self, x_prev_2, x_prev, x_curr, P_prev_2, P_prev, P_curr, dt,
            a_allowed = 3.0, conf_level=0.99, sigmoid_upper = 0.5, slope = 3
        ):
        """
        Computes damping based on the Mahalanobis norm of acceleration relative to position covariance.
        Damping reaches 1.0 when the Mahalanobis norm exceeds the threshold for `conf_level` significance.
        
        Parameters:
            x_prev_2: Position at t-2
            x_prev: Position at t-1
            x_curr: Position at t
            P_curr: Position covariance at time t (3x3)
            dt: Time step (seconds)
            conf_level: Confidence level threshold (default: 0.99)

        Returns:
            damping: float in [0, 1]
        """
        # === Stack measurements
        x_stack = np.hstack([x_prev_2, x_prev, x_curr])  # shape (3D,)
        P_stack = np.block([
            [P_prev_2,              np.zeros_like(P_prev_2), np.zeros_like(P_prev_2)],
            [np.zeros_like(P_prev), P_prev,                  np.zeros_like(P_prev)],
            [np.zeros_like(P_curr), np.zeros_like(P_curr),   P_curr]
        ])

        D = x_curr.shape[0]

        # === Acceleration operator
        coeffs = np.array([1, -2, 1]) / dt**2
        L = np.kron(coeffs, np.eye(D))  # shape (D, 3D)

        # === Compute raw acceleration and its covariance
        a = L @ x_stack
        P_a = L @ P_stack @ L.T

        # === Subtract allowed baseline acceleration in direction of a
        a_norm = np.linalg.norm(a)
        if a_norm > 1e-6 and a_allowed > 0:
            a_dir = a / a_norm
            a_excess = a - a_allowed * a_dir
            # Clamp to zero if result flips sign (i.e. within allowed)
            if np.dot(a_excess, a) < 0:
                a_excess = np.zeros_like(a)
        else:
            a_excess = a.copy()

        # === Mahalanobis norm squared
        try:
            inv_P_a = np.linalg.inv(P_a)
        except np.linalg.LinAlgError:
            inv_P_a = np.linalg.pinv(P_a)

        a_maha_sq = float(a_excess.T @ inv_P_a @ a_excess)

        # === Step 6: Normalize squared Mahalanobis
        chi2_thresh = chi2.ppf(conf_level, df=D)
        normalized = a_maha_sq / chi2_thresh

        damping = 1 - 1.0 / (1 + normalized)

        return damping, a_excess

    def acceleration_cov_boost(self, a, scale=1.0, damping=0.0):
        """
        Returns a covariance boost matrix aligned with acceleration direction.

        Parameters:
            a: Acceleration vector (shape [D])
            a_maha_sq: Mahalanobis norm squared of acceleration (scalar)
            P_a: Acceleration covariance matrix (shape [D, D])
            scale: User-defined global scale (e.g., vel_boost_scale)
            damping: Damping factor in [0, 1], higher means more suppression

        Returns:
            cov_boost: (D x D) matrix
        """
        a_norm = np.linalg.norm(a)
        if a_norm < 1e-6:
            return np.zeros((3, 3))

        a_dir = a / a_norm
        boost_magnitude = scale * damping * a_norm ** 2
        cov_boost = boost_magnitude * np.outer(a_dir, a_dir)
        return cov_boost 

    def modify_velocity_cov_for_daidalus(self,
            history, daidalus_cov, measurement_cov, dt, N=10,
            cos_center=0.95, mag_center=0.1,
            shrink_target=0.05, min_scale=0.05,
            directional=True,
            shift_boost_scale=0.1
        ):
            """
            Continuously shrinks velocity covariance based on alignment and magnitude agreement,
            and applies a directional stretch based on the most recent velocity shift.

            Parameters:
                history: list of dicts with smoothed positions and velocities
                daidalus_cov: 6x6 Kalman covariance matrix
                dt: time delta between measurements
                N: number of history steps to use
                cos_center: threshold cosine similarity (alignment)
                mag_center: threshold magnitude ratio (agreement)
                shrink_target: how low the shrink factor can go (e.g., 0.3)
                min_scale: minimum eigenvalue in the velocity covariance
                directional: whether to shrink only along mean motion direction
                shift_boost_scale: scaling factor for velocity jump covariance boost
            """
            if len(history) < max(N + 1, 3):  # Need at least 3 for velocity shift
                return daidalus_cov

            P_vel = copy.deepcopy(daidalus_cov[3:6, 3:6])

            # === Define directional boost helper ===
            def directional_boost_vel(mean_delta, max_norm=50):
                norm = np.linalg.norm(mean_delta)
                if norm > max_norm:
                    mean_delta = mean_delta / norm * max_norm
                    norm = max_norm
                if norm < 1e-6:
                    return np.zeros((3, 3))
                dir_unit = mean_delta / norm
                return norm ** 2 * np.outer(dir_unit, dir_unit)  # directional variance
            
            def directional_boost_pos(velocity_proj: float, max_norm: float = 500.0, scale: float = 10.0, cos_sims = []) -> float:
                """
                Computes a scalar boost based on the projected velocity component.
                """
                norm = np.abs(velocity_proj)

                # === Get largest cosine similarity deviation from 1 ===
                if len(cos_sims) > 0:
                    velocity = history[-1]['velocity_x_y_z_smoothed']

                    # === Step 1: Compute dominant eigenvector of measurement and daidalus covariance ===
                    eigvals, eigvecs = np.linalg.eigh(measurement_cov)
                    dominant_dir = eigvecs[:, -1]  # Eigenvector with largest eigenvalue

                    # eigvals, _ = np.linalg.eigh(daidalus_cov_orig[0:3, 0:3])
                    dominant_val = np.sqrt(eigvals[-1])

                    # === Step 2: Normalize velocity ===
                    v_norm = np.linalg.norm(velocity)
                    if v_norm > 1e-6:
                        v_unit = velocity / v_norm
                        cos_theta = np.clip(np.dot(v_unit, dominant_dir), -1.0, 1.0)
                        if abs(cos_theta) < abs(np.cos(np.radians(10))):
                            perp_factor = 1.0 - cos_theta**2  # sin(theta)
                        else:
                            perp_factor = 0.0
                    else:
                        perp_factor = 0.0
                        
                    max_cos_dev = max([1.0 - abs(cs) for cs in cos_sims])
                else:
                    max_cos_dev = 0.0
                    perp_factor = 0.0
                    dominant_val = 0.0

                # === Convert to scaling factor ===
                cosine_boost_factor = (dominant_val * max_cos_dev * perp_factor)  # Tunable scaling function

                clipped = np.clip(norm + cosine_boost_factor, 0, max_norm)

                return scale * clipped ** 2

            # # Extract history
            N_pos = 10
            pos_seq = np.array([h['abs_cartesian_x_y_z_smoothed'] for h in history[-(N_pos+1):]])

            # === Compute velocity shift stretch using empirical covariance ===
            if len(pos_seq) >= N:
                # === Medium-term velocity shift (acceleration) ===
                recent_pos_med = pos_seq[-N:]  # shape (N, 3)
                acc_med = np.abs(np.diff(np.diff(recent_pos_med, axis=0) / dt, axis=0))  # (N-2, 3)
                mean_acc_med = np.mean(acc_med, axis=0)

                # === Short-term velocity shift (acceleration) ===
                recent_pos_short = pos_seq[-3:]
                acc_short = np.abs(np.diff(np.diff(recent_pos_short, axis=0) / dt, axis=0))  # (1, 3)
                mean_acc_short = np.mean(acc_short, axis=0)

                # === Boosts ===
                boost_vel_med = directional_boost_vel(mean_acc_med, max_norm=50)
                boost_vel_short = directional_boost_vel(mean_acc_short, max_norm=25)

                daidalus_cov[3:6, 3:6] += shift_boost_scale * (boost_vel_med + boost_vel_short)

            # === Apply shrink ===
            if directional:
                # === Extract history excluding coasting frames ===
                non_coasting_history = [h for h in history if not h['coasting']]

                # Need at least N+1 for pos diff, and N for velocity
                if len(non_coasting_history) < N + 1:
                    return daidalus_cov  # Not enough data to proceed

                # Slice latest N+1 non-coasting states
                non_coast_hist = non_coasting_history[-(N + 1):]

                # Extract position and velocity sequences
                pos_seq = np.array([h['abs_cartesian_x_y_z_smoothed'] for h in non_coast_hist])      # (N+1, 3)
                vel_kf_seq = np.array([h['velocity_x_y_z_smoothed'] for h in non_coast_hist[:-1]])   # (N, 3)

                # Compute empirical velocity
                vel_emp_seq = (pos_seq[1:] - pos_seq[:-1]) / dt                                      # (N, 3)

                # Align and compute delta-v
                delta_v_array = vel_emp_seq - vel_kf_seq            

                # === Covariance of deltas between empirical and KF velocity ===
                P_delta = np.cov(delta_v_array, rowvar=False)

                # === Eigendecompose ===
                eigvals, eigvecs = np.linalg.eigh(P_delta)

                # === Convert delta variance to directional shrink factors ===
                # More shrink in stable (low error variance) directions
                shrink_factors = 1 / (1 + eigvals)

                # Rescale into [shrink_target, 1.0]
                shrink_factors = shrink_target + (1 - shrink_factors) * (1.0 - shrink_target)

                # === Apply directional shrink ===
                P_vel_shrunk = np.zeros((3, 3))
                for i in range(3):
                    e_i = eigvecs[:, i:i+1]
                    proj = e_i @ e_i.T
                    P_vel_shrunk += shrink_factors[i] * (proj @ P_vel @ proj)

                # Compute cosine similarities and for position variance calcs below
                cos_sims = []
                for v_emp, v_kf in zip(vel_emp_seq, vel_kf_seq):
                    norm_emp = np.linalg.norm(v_emp)
                    norm_kf = np.linalg.norm(v_kf)
                    if norm_emp > 1e-3 and norm_kf > 1e-3:
                        cos_sim = np.dot(v_emp, v_kf) / (norm_emp * norm_kf)
                    else:
                        cos_sim = 1.0
                    cos_sims.append(cos_sim)
            else:
                # === Extract history excluding coasting frames ===
                non_coasting_history = [h for h in history if not h['coasting']]

                # Need at least N+1 for pos diff, and N for velocity
                if len(non_coasting_history) < N + 1:
                    return daidalus_cov  # Not enough data to proceed

                # Slice latest N+1 non-coasting states
                non_coast_hist = non_coasting_history[-(N + 1):]

                # Extract position and velocity sequences
                pos_seq = np.array([h['abs_cartesian_x_y_z_smoothed'] for h in non_coast_hist])      # (N+1, 3)
                vel_kf_seq = np.array([h['velocity_x_y_z_smoothed'] for h in non_coast_hist[:-1]])   # (N, 3)

                # Compute empirical velocity
                vel_emp_seq = (pos_seq[1:] - pos_seq[:-1]) / dt                                      # (N, 3)

                # Compute cosine similarities and magnitude ratios
                cos_sims, mag_ratios = [], []
                for v_emp, v_kf in zip(vel_emp_seq, vel_kf_seq):
                    norm_emp = np.linalg.norm(v_emp)
                    norm_kf = np.linalg.norm(v_kf)
                    if norm_emp > 1e-3 and norm_kf > 1e-3:
                        cos_sim = np.dot(v_emp, v_kf) / (norm_emp * norm_kf)
                        mag_ratio = np.abs(norm_emp - norm_kf) / norm_emp
                    else:
                        cos_sim = 1.0
                        mag_ratio = 0.0
                    cos_sims.append(cos_sim)
                    mag_ratios.append(mag_ratio)

                mean_cos = np.mean(cos_sims)
                mean_mag_ratio = np.mean(mag_ratios)

                # === Custom-tuned shrink weights ===
                norm_cos = (1 - mean_cos) / (1 - cos_center + 1e-6)
                alignment_weight = 1 / (1 + 3 * norm_cos)  # Tuned to match your plot

                norm_mag = mean_mag_ratio / (mag_center + 1e-6)
                magnitude_weight = 1 / (1 + norm_mag)

                shrink_weight = alignment_weight * magnitude_weight
                effective_shrink = 1.0 - shrink_weight * (1.0 - shrink_target)

                print(f"[Shrink Debug] Cos: {mean_cos:.3f}, Mag: {mean_mag_ratio:.3f}, "
                    f"AlignW: {alignment_weight:.3f}, MagW: {magnitude_weight:.3f}, "
                    f"ShrinkW: {shrink_weight:.3f}, EffShrink: {effective_shrink:.3f}")
            
                P_vel_shrunk = effective_shrink * P_vel

            # Ensure eigenvalues are not too small
            eigvals = np.linalg.eigvalsh(P_vel_shrunk)
            if np.any(eigvals < min_scale):
                P_vel_shrunk += (min_scale - np.min(eigvals) + 1e-6) * np.eye(3)

            daidalus_cov[3:6, 3:6] = P_vel_shrunk

            if len(pos_seq) >= N_pos:
                # === Extract position covariance and dominant direction ===
                cov_pos = daidalus_cov[0:3, 0:3]
                eigvals, eigvecs = np.linalg.eigh(cov_pos)
                dominant_dir = eigvecs[:, -1]  # Eigenvector of largest eigenvalue

                # === Medium-term position shift (velocity magnitude/direction changes) ===
                recent_pos_med = pos_seq  # shape (N, 3)
                pos_deltas_med = np.abs(np.diff(recent_pos_med, axis=0))  # (N-1, 3)
                mean_vel_med = np.mean(pos_deltas_med, axis=0)

                # Project onto dominant direction
                proj_med = np.dot(mean_vel_med, dominant_dir)
                boost_med_mag = directional_boost_pos(proj_med, max_norm=100, scale = 20, cos_sims = cos_sims)

                # === Short-term shift ===
                recent_pos_short = pos_seq[-2:]
                pos_deltas_short = np.abs(np.diff(recent_pos_short, axis=0))
                mean_vel_short = np.mean(pos_deltas_short, axis=0)

                proj_short = np.dot(mean_vel_short, dominant_dir)
                boost_short_mag = directional_boost_pos(proj_short, max_norm=100, scale = 8)

                # === Apply directional covariance stretch along dominant_dir ===
                # eigvals, eigvecs = np.linalg.eigh(measurement_cov)
                # dominant_val = eigvals[-1]
                this_cov_boost = max(0.8 ** 2 * self.pos_cov_boost, min(150 ** 2, boost_med_mag + boost_short_mag))#max(0.8 ** 2 * self.pos_cov_boost, min([300 ** 2, 1.05 ** 2 * self.pos_cov_boost, boost_med_mag + boost_short_mag]))
                directional_cov_boost = this_cov_boost * np.outer(dominant_dir, dominant_dir)
                self.pos_cov_boost = this_cov_boost
                
                # === Apply total boost ===
                daidalus_cov[0:3, 0:3] += directional_cov_boost
            return daidalus_cov

    def iterate_tracker_state(self, track_id, track_source, coasting, fused_time, cam_ids, fused_mean = None, fused_cov = None, new_init_histories = {}, retroactive_update = False):
        damping_abs_x, a_abs_x = 0.0, 0.0
        past_measurement = False
        # === Initialize or update Kalman filter ===
        
        if track_id not in self.xyz_kalman_filters[track_source]:
            if coasting:
                print("Error! Passing Coasting = True for uninitialised Track!")
                return {}

            # Initialise guidance history for this track
            self.iso_decomp_guidance_hist.setdefault(
                    track_id,
                    {
                        'iters_since_guidance': 0,
                        'last_relaxed_span': 0.0,
                        'num_relaxations': 0,
                        'iters_at_max_relaxation': 0
                    }
                )

            print(f"Initialising New Track: {track_id}")

            # Initialise tracker and seed with initialisation history
            if new_init_histories:
                print("Init History Detected")
                self.propagate_init_history(new_init_histories, track_id, track_source)
                print("History Propagation Complete. Now Processing Latest Detection.")
                return self.iterate_tracker_state(track_id, track_source, coasting, fused_time, cam_ids, fused_mean, fused_cov)
            # If no init histories (e.g. ADSB), or when recursively running iterate_tracker_state inside propagate_init_history, the first detection is passed without an init hisory so this line is triggered
            else:
                print("No Init History Detected")
                dt = 0.1  # Initial timestep
                self.xyz_kalman_filters[track_source][track_id] = self.create_kalman_filter(track_source, fused_mean, fused_cov, dt, fused_time)
                kf_xyz = self.xyz_kalman_filters[track_source][track_id]
        else:
            # print(f"Propagating Track: {track_id}")
            if not retroactive_update:
                # === Get kalman filter and history for this track ===
                kf_xyz = self.xyz_kalman_filters[track_source][track_id]
                this_track_history = self.track_history[track_id]

                # === Check recency of new measurement ===
                #############################################################################################
                #### CASE 1: Ignore if this measurement has been superceded by a more recent measurement ####
                #############################################################################################
                if (fused_time <= kf_xyz.last_measured_time) or (coasting and fused_time <= kf_xyz.prev_timestamp):
                    return {}
                
                #########################################################
                #### CASE 2: Update normally with latest measurement ####
                #########################################################
                if fused_time > kf_xyz.prev_timestamp:
                    # === Calcaulte dt for this timestemp ===
                    dt = fused_time - kf_xyz.prev_timestamp

                    # === Update Kalman matrices and timestep before update ===
                    self.update_F_Q_matrices(kf_xyz, dt, coasting, fused_time, track_source)

                    # === Kalman predict step ===
                    kf_xyz.predict()

                ##############################################################################################################
                #### CASE 3: Retroactively update Kalman filter if this measurement is in the past but is the most recent ####
                ##############################################################################################################
                else:
                    past_measurement = True
                    # Bookmark current timestamp since this will be overwritten in the recursive update
                    current_timestamp = kf_xyz.prev_timestamp

                    print(f"New measurement detected in previous coasting period ({current_timestamp - fused_time}s ago) - retroatively updating")

                    # This recursive update sets the Kalman filter state and history state to the state right after this most recent update
                    self.iterate_tracker_state(track_id, track_source, coasting, fused_time, cam_ids, fused_mean, fused_cov, retroactive_update=True)

                    # === Get updated kalman filter after recursive update ===
                    kf_xyz = self.xyz_kalman_filters[track_source][track_id]

                    # Now just coast from the last measurement time to the current timestamp

                    # === Calculate coasting time from most recent measurement to current timestamp ===
                    coasting_time = current_timestamp - fused_time

                    if coasting_time > 0:
                        coasting_steps = max(1, int(coasting_time // 0.25)) # Using 0.25s coast timestep (tuneable)

                        # === Break coasting period up into number of timesteps ===
                        dt = coasting_time / coasting_steps

                        # === Coast from most recent measurement to current timestamp ===
                        for time_step in range(1, coasting_steps + 1):
                            # New fused time is the new time after each added timestep
                            new_timestamp = fused_time + time_step * dt
                            # === Update Kalman matrices and timestep before update ===
                            self.update_F_Q_matrices(kf_xyz, dt, True, new_timestamp, track_source)
                            # === Kalman predict step for each coasting timestep ===
                            kf_xyz.predict()

                        # === Update Kalman Filter Timestamp after Predict Step(s) ===
                        kf_xyz.prev_timestamp = current_timestamp
                    else:
                        # Edge case: If new measurement is at exactly the same time as last coast update just read off last dt
                        dt = kf_xyz.dt
            else:
                ############################################################################################
                ##### Retroactive Update Step 1: Coast from last measurement to the current measurement ####
                ############################################################################################
                # === Reset kalman filter and history for this track to state at last measurement ===
                kf_xyz = copy.deepcopy(self.xyz_kalman_filters[track_source][track_id].state_at_last_measured_time)
                this_track_history = copy.deepcopy(kf_xyz.track_history_at_last_measured_time)
                self.xyz_kalman_filters[track_source][track_id] = kf_xyz
                self.track_history[track_id] = this_track_history
                
                # === Calcualte coasting time from last measurement to this most recent measurement ===
                coasting_time = fused_time - kf_xyz.last_measured_time
                coasting_steps = max(1, int(coasting_time // 0.25)) # Using 0.25s coast timestep (tuneable)

                # === Break coasting period up into number of timesteps ===
                dt = coasting_time / coasting_steps

                # === Coast from last measurement to this most recent measurement ===
                for time_step in range(1, coasting_steps + 1):
                    # === Kalman predict step for each coasting timestep ===
                    # New fused time is the new time after each added timestep
                    new_timestamp = kf_xyz.last_measured_time + time_step * dt
                    # === Update Kalman matrices and timestep before update ===
                    self.update_F_Q_matrices(kf_xyz, dt, True, new_timestamp, track_source)
                    kf_xyz.predict()
                
                # === Reset coasting status for subsequent measurement update once coasting complete ===
                # TODO: Check if the update step uses these matrices of they're only used in predict, in whcih case this line can be removed
                self.update_F_Q_matrices(kf_xyz, dt, coasting, fused_time, track_source)
                
            # === Apply shock damping to most recent measurement (EO Only) ===
            if not coasting and not past_measurement:
                if track_source == "EO":
                    if (
                        len(this_track_history) > 2
                    ):
                        if (
                            this_track_history[-1]['abs_cartesian_x_y_z_cov'] is not None and 
                            this_track_history[-2]['abs_cartesian_x_y_z_cov'] is not None
                        ):
                            x_prev_2 = this_track_history[-2]['abs_cartesian_x_y_z'].copy()
                            P_prev_2 = copy.deepcopy(this_track_history[-2]['abs_cartesian_x_y_z_cov'])
                            x_prev = this_track_history[-1]['abs_cartesian_x_y_z'].copy()
                            P_prev = copy.deepcopy(this_track_history[-1]['abs_cartesian_x_y_z_cov'])

                            x_curr = fused_mean.copy()
                            P_curr = copy.deepcopy(fused_cov)

                            damping_abs_x, a_abs_x = self.compute_damping_measurement_jitter(
                                x_prev_2, x_prev, x_curr, P_prev_2, P_prev, P_curr, dt, conf_level=0.9, sigmoid_upper = 1)

                    if fused_time - kf_xyz.track_init_timestamp < 10:
                        jitter_inflation_factor_measurement = 0.05
                        measurement_jitter_boost = self.acceleration_cov_boost(a_abs_x, scale=jitter_inflation_factor_measurement, damping=damping_abs_x)
                        fused_cov_new = copy.deepcopy(fused_cov) + measurement_jitter_boost# * malh_resid_scalar ** 2 * innovation_inflation_factor
                    else:
                        jitter_inflation_factor_measurement = 0.05
                        measurement_jitter_boost = self.acceleration_cov_boost(a_abs_x, scale=jitter_inflation_factor_measurement, damping=damping_abs_x)
                        
                        # === Compute range direction from ownship to intruder ===
                        [x_own, y_own, z_own] = self.ownship.position(ownship_time = fused_time).flatten()
                        [x_intr, y_intr, z_intr] = kf_xyz.x[0:3].reshape(3,).copy()
                        range_vec = np.array([x_intr - x_own, y_intr - y_own, z_intr - z_own])
                        range_norm = np.linalg.norm(range_vec)
                        range_unit = range_vec / range_norm if range_norm > 1e-6 else np.array([1.0, 0.0, 0.0])  # fallback

                        # === Innovation residual in direction of range vector ===
                        fused_pos = fused_mean.reshape(3,).copy()
                        y_eh = fused_pos - np.array([x_intr, y_intr, z_intr])
                        var_along_range = range_unit @ copy.deepcopy(fused_cov)[:3, :3] @ range_unit
                        resid_scalar = range_unit @ y_eh
                        malh_resid_scalar = resid_scalar / var_along_range              
                        innovation_inflation_factor = 800000
                        fused_cov_new = copy.deepcopy(fused_cov) + measurement_jitter_boost * max(1.0, abs(malh_resid_scalar) ** 2 * innovation_inflation_factor)
                else:
                    fused_cov_new = fused_cov
                kf_xyz.R = fused_cov_new
                kf_xyz.update(fused_mean)

            # === Post-Update Acceleration Capping (EO Only) ===
            if track_source == "EO":
                # Pull previous state
                v_prev = np.asarray(self.track_history[track_id][-1]['velocity_x_y_z_smoothed'], dtype=float)
                x_prev = np.asarray(self.track_history[track_id][-1]['abs_cartesian_x_y_z_smoothed'], dtype=float)

                # Current state from filter
                x_curr = np.asarray(kf_xyz.x[0:3].reshape(3,), dtype=float)
                v_now  = np.asarray(kf_xyz.x[3:6].reshape(3,), dtype=float)

                # Use ONE clamped dt everywhere
                dt_s = float(max(dt, 1e-6))

                # Raw accel estimate over this interval
                a = (v_now - v_prev) / dt_s

                # === Direction from intruder to ownship (for "away" check) ===
                range_vec  = np.asarray(self.ownship.position(ownship_time=fused_time).flatten(), dtype=float) - x_curr
                range_norm = float(np.linalg.norm(range_vec))
                if range_norm < 1e-6:
                    range_unit = np.array([1.0, 0.0, 0.0], dtype=float)  # safe fallback
                else:
                    range_unit = range_vec / range_norm

                if fused_time - kf_xyz.track_init_timestamp < 10:
                    # --- Forbid "away" accel, then isotropically clamp magnitude ---
                    AWAY_ACC_CAP = 0.0   # m/s^2 allowed in the away (-range_unit) direction
                    ACC_CAP      = 20.0  # isotropic cap on |a|

                    # Step 1: zero any "away" component (a_toward < 0 means away)
                    a_toward = float(np.dot(a, range_unit))
                    a_toward_clamped = max(a_toward, -AWAY_ACC_CAP)   # with 0.0 => no away accel
                    a_no_away = a + (a_toward_clamped - a_toward) * range_unit

                    # Step 2: isotropic cap
                    a_norm = float(np.linalg.norm(a_no_away))
                    a_clamped = a_no_away * (ACC_CAP / a_norm) if a_norm > ACC_CAP else a_no_away

                    # Update velocity & position with clamped accel
                    v_now_clamped = v_prev + a_clamped * dt_s
                    kf_xyz.x[3:6] = v_now_clamped.reshape(3, 1)
                else:
                    # --- Directional clamping in a body-like frame (fwd/lat/vert) ---
                    # Build orthonormal frame
                    vnorm = float(np.linalg.norm(v_prev))
                    v_hat = v_prev / vnorm if vnorm > 1e-3 else np.array([1.0, 0.0, 0.0], dtype=float)

                    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
                    lateral = np.cross(z_axis, v_hat)
                    lat_n = float(np.linalg.norm(lateral))
                    if lat_n < 1e-3:
                        lateral = np.array([0.0, 1.0, 0.0], dtype=float)
                    else:
                        lateral /= lat_n
                    vertical = np.cross(v_hat, lateral)
                    ver_n = float(np.linalg.norm(vertical))
                    if ver_n > 1e-9:
                        vertical /= ver_n

                    # Components of a
                    a_fwd = float(np.dot(a, v_hat))
                    a_lat = float(np.dot(a, lateral))
                    a_ver = float(np.dot(a, vertical))

                    # Directional caps
                    a_fwd_c = float(np.clip(a_fwd, -20.0, 20.0))
                    a_lat_c = float(np.clip(a_lat, -10.0, 10.0))
                    a_ver_c = float(np.clip(a_ver, -10.0, 10.0))

                    # Recompose clamped accel
                    a_clamped = a_fwd_c * v_hat + a_lat_c * lateral + a_ver_c * vertical

                    # Update velocity (position update optional here)
                    v_now_clamped = v_prev + a_clamped * dt_s
                    kf_xyz.x[3:6] = v_now_clamped.reshape(3, 1)

                    # If you want position consistency here too, uncomment (this position clamping seems to break it):
                    # x_curr_clamped = x_prev + v_prev * dt_s + 0.5 * a_clamped * (dt_s ** 2)
                    # kf_xyz.x[0:3] = x_curr_clamped.reshape(3, 1)

            # === Update Kalman Filter Timestamp after Predict Step(s) ===
            # Note, intermediate timestamps in retroactive coasting not used anywhere so only final timestamp saved
            kf_xyz.prev_timestamp = fused_time

        # === Extract smoothed state ===
        x, y, z, vx, vy, vz = kf_xyz.x[:6, 0]

        if track_source == "EO":
            daidalus_uncertainty_factor = 2.5
        else:
            daidalus_uncertainty_factor = 1.0

        daidalus_cov = copy.deepcopy(kf_xyz.P)[0:6, 0:6] * daidalus_uncertainty_factor

        if track_source == "EO" and not coasting:
            if track_id in self.track_history:
                history = self.track_history[track_id]

                daidalus_cov = self.modify_velocity_cov_for_daidalus(
                    history, daidalus_cov, fused_cov.copy(), dt
                    )

        std_xyz = np.sqrt(np.diag(daidalus_cov[0:3, 0:3]))
        std_vxyz = np.sqrt(np.diag(daidalus_cov[3:6, 3:6]))

        # === Extract signed std from YX covariances ===
        cov_yx = daidalus_cov[1, 0]
        std_yx_signed = np.sign(cov_yx) * np.sqrt(np.abs(cov_yx))

        cov_vyx = daidalus_cov[4, 3]
        std_vyx_signed = np.sign(cov_vyx) * np.sqrt(np.abs(cov_vyx))

        # === Convert Relative Cartesian to Lat Lon Coords
        # Edge Case Handler: Somehow have intruder detections before home lat lon set by first ownship position
        if self.home_lat_lon is None:
            return {}
       
        intr_lat, intr_lon, _ = navpy.ned2lla(
            [x, y, -z], self.home_lat_lon[0], self.home_lat_lon[1], 0)
 
        fused_item = {
            'time': fused_time,
            'lat': intr_lat,
            'lon': intr_lon,
            'abs_cartesian_x_y_z': [np.nan, np.nan, np.nan] if coasting else fused_mean.copy(),
            'abs_cartesian_x_y_z_cov': None if coasting else copy.deepcopy(fused_cov),
            'inflated_2_sigma_bounding_cov': copy.deepcopy(daidalus_cov),
            'abs_cartesian_x_y_z_smoothed': np.array([x, y, z]),
            'velocity_x_y_z_smoothed': np.array([vx, vy, vz]),
            'uncertainty_cartesian_x_y_z_smoothed': std_xyz,
            'uncertainty_velocity_x_y_z_smoothed': std_vxyz,
            'uncertainty_cartesian_yx_std_smoothed': std_yx_signed,
            'uncertainty_velocity_yx_std_smoothed': std_vyx_signed,
            'trk_smoothed': (np.degrees(np.arctan2(vy, vx)) + 360) % 360,
            'cam_ids_contributed': cam_ids,
            'coasting': coasting,
            'track_source': track_source
        }

        # If we want to plot the initialisation history, uncomment this
        # if track_source == 'EO':
        #     x, y, z = fused_item['abs_cartesian_x_y_z_smoothed']
        #     vx, vy, vz = fused_item['velocity_x_y_z_smoothed']
        #     kf_xyz = self.xyz_kalman_filters[track_id]
        #     std_xyz = fused_item['uncertainty_cartesian_x_y_z_smoothed']
        #     std_vxyz = fused_item['uncertainty_velocity_x_y_z_smoothed']
        #     cor_yx = fused_item['uncertainty_cartesian_yx_std_smoothed']
        #     cor_vyx = fused_item['uncertainty_velocity_yx_std_smoothed']

        #     # === Append latest point to history ===
        #     self.fused_xy_history.append((x, y, std_xyz[0], std_xyz[1]))

        # === Update history ===
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(fused_item)
        if len(self.track_history[track_id]) > self.track_history_length:
            del self.track_history[track_id][:-self.track_history_length]  # remove all but the last 100

        # === Update Kalman filter with last measurement state params ===
        if not coasting:
            kf_xyz.last_measured_time = fused_time 
            kf_xyz.state_at_last_measured_time = None
            kf_xyz.state_at_last_measured_time = copy.deepcopy(kf_xyz)
            kf_xyz.track_history_at_last_measured_time = copy.deepcopy(self.track_history[track_id])

        return fused_item

    def update_tracks_v3(self, ownship_state, detection_summaries, adsb_intruders = [], intruder_state = None, daidalus_sugg_guidance = None, use_internal_ownship_tracker = True):
        output_tracks = {}
        terminated_track_ids = []

        ownship_time = ownship_state["time"]
        
        if use_internal_ownship_tracker:
            ################################################
            ########## OWNSHIP KALMAN FILTER ###############
            ################################################
            track_source = 'Ownship'
            track_id = 'Ownship'

            # Calculate ownship velocity components
            trk_rad = np.deg2rad(ownship_state["trk"])
            vx_n = ownship_state["ground_speed"] * np.cos(trk_rad)  # North component (m/s)
            vy_e = ownship_state["ground_speed"] * np.sin(trk_rad)  # East component (m/s)                # Up component (m/s)

            ownship_pos_vel = np.array([ownship_state["north_m"], ownship_state["east_m"], ownship_state["alt"],
                                    vx_n, vy_e, ownship_state["vertical_speed"]])
            
            pos_std, vel_std = 15.0, 5.0
            ownship_cov = np.diag([pos_std**2, pos_std**2, pos_std**2, vel_std**2, vel_std**2, vel_std**2])# TODO: Replace with correct Ownship uncertainty logic

            ownship_tracker_output = self.iterate_tracker_state(track_id, track_source, False, ownship_time, None, ownship_pos_vel, ownship_cov)

            if ownship_tracker_output:
                self.ownship.set_state(
                    tuple(ownship_tracker_output["abs_cartesian_x_y_z_smoothed"]),
                    (ownship_state['rl'], ownship_state['pt'], ownship_tracker_output["trk_smoothed"]), #TODO: Do proper attitude, right now just using heading as yaw
                    ownship_time
                    )
                
                # Standardise ownship output structure
                processed_ownship_tracker_output = self.process_tracked_ownship_state(ownship_tracker_output)

                # Add standardised ownship in return object
                output_tracks["Ownship"] = processed_ownship_tracker_output

        else:
            ################################################
            ########## DIRECT OWNSHIP INPUTS ###############
            ################################################
            if self.ownship.set_state(
                    tuple([ownship_state["north_m"], ownship_state["east_m"], ownship_state["alt"]]),
                    (ownship_state['rl'], ownship_state['pt'], ownship_state["yw"]), #TODO: Do proper attitude, right now just using heading as yaw
                    ownship_time
                ):

                # Add direct ownship input in return object IF ownship state has changed (set state returned True)
                output_tracks["Ownship"] = ownship_state

        ################################################
        ######## EO DETECTION KALMAN FILTERS ###########
        ################################################
        for queue_idx, detection_summary in enumerate(detection_summaries):
            new_track_items = {}
            new_init_histories = {}
            for cam_id in range(len(detection_summary['frame_id'])):
                if not detection_summary['detections'][cam_id]:
                    continue
                
                time_this = float(detection_summary["frame_id"][cam_id].split("_")[1])

                AFMat_this = detection_summary['predicted_transforms'][cam_id]
                this_cam_detections = detection_summary['detections'][cam_id]

                filtered_track_items = self.trackers[cam_id].process_frame_detections(detections=this_cam_detections, transform=AFMat_this, cam_obj = self.ownship.cameras[cam_id]['camera'], detection_time=time_this)

                if filtered_track_items:
                    for it in filtered_track_items:
                        new_track_items[f"cam_{cam_id}_trk_{it.track_id}"] =  dict(
                                cam_id=cam_id,
                                time=it.time,
                                cx=it.abs_cx,
                                cy=it.abs_cy,
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
                                init_history=it.init_history,
                                abs_cartesian_x_y_z = self.ownship.map_pixel_views_to_global_cartesian_coords(cam_id, np.array([[it.abs_cx, it.abs_cy, it.distance, it.time]]))[0]
                            )  
                        
                        # Extract initialisation history for any new detections
                        if it.init_history:
                            # print(it.init_history)
                            # times = []
                            # dists = []

                            # for it in it.init_history:
                            #     # Main detection
                            #     if it.time is not None and it.distance is not None:
                            #         times.append(float(it.time))
                            #         dists.append(float(it.distance))

                            #     # Its init history (list of DetectedItem-like objects)
                            #     if it.init_history:
                            #         for h in it.init_history:
                            #             if h.time is not None and h.distance is not None:
                            #                 times.append(float(h.time))
                            #                 dists.append(float(h.distance))

                            # # Optional: sort by time in case they're out of order
                            # pairs = sorted(zip(times, dists), key=lambda x: x[0])
                            # times, dists = zip(*pairs) if pairs else ([], [])

                            # # Plot
                            # plt.figure()
                            # plt.plot(times, dists, marker='o')
                            # plt.xlabel("Time (s)")
                            # plt.ylabel("Distance (m)")
                            # plt.title("Detection + Init History Distance vs Time")
                            # plt.grid(True)
                            # plt.show()
                            for init_hist in it.init_history:
                                time_det = init_hist.time
                                new_init_histories.setdefault(time_det, []).append(init_hist)  

            # Combine measurements that are within epsilon time delta of eachother
            epsilon = 1e-3
            merged_histories = OrderedDict()

            # Step 1: Sort timestamps
            for time in sorted(new_init_histories):
                assigned = False
                for existing_time in merged_histories:
                    if abs(time - existing_time) < epsilon:
                        merged_histories[existing_time].extend(new_init_histories[time])
                        assigned = True
                        break
                if not assigned:
                    merged_histories[time] = list(new_init_histories[time]) 
            
            new_init_histories = merged_histories

            track_source = 'EO'

            if self.enforce_single_EO_intruder:

                track_id = 1  # fixed ID

                if new_track_items:
                    means = []
                    covs = []
                    cam_ids = []
                    timestamps = []
                    # detection_confidences = []
                    # horizon_statuses = []
                    coasting = False

                    for item in new_track_items.values():
                        cam_id = item['cam_id']
                        raw_range = item['distance']
                        range_bias, range_var, pixel_var = self.ownship.cameras[cam_id]['camera'].get_detected_uncertainty_params(raw_range)
                        pixel_std = np.sqrt(pixel_var)
                        range_std = np.sqrt(range_var)

                        mean_global_adjusted, cov_global = self.calculate_xyz_mean_and_covariance(
                            pixel_std, pixel_std, range_std, range_bias,
                            item['cx'], item['cy'], raw_range, cam_id, item['time']
                        )

                        # If all uncertainties are zero, fallback to small epsilon covariance
                        if np.allclose(cov_global, 0):
                            cov_global = np.eye(3) * 1e-6

                        means.append(mean_global_adjusted)
                        covs.append(cov_global)
                        cam_ids.append(cam_id)
                        timestamps.append(item['time'])
                        # detection_confidences.append(item['conf'])
                        # horizon_statuses.append(item['above_horizon'])

                    if len(means) == 1:
                        fused_cov = covs[0]
                        fused_mean = means[0]
                        fused_time = timestamps[0]
                    else:
                        # === Gaussian Fusion ===
                        precisions = []
                        for C in covs:
                            if np.allclose(C, 0):
                                C = np.eye(3) * 1e-6  # fallback for perfect measurements
                            precisions.append(inv(C))
                        precision_sum = sum(precisions)

                        fused_cov = inv(precision_sum)
                        fused_mean = fused_cov @ sum(P @ m for P, m in zip(precisions, means))
                        fused_time = np.mean(timestamps)
                else:
                    # === If no new detections but track also exists, Coasting ===
                    if track_id in self.track_history:
                        # === If no new detections but track also exists, Coasting ===
                        if len(self.track_history[track_id]) > 2:
                            # Only coast on last telemetry in the queue since ownship has the same timestamp for each item in this queue
                            # (high frequency of thread iteration means that in practice, ownship lag will be very small compared to any intruder in the same queue batch)
                            if queue_idx != len(detection_summaries) - 1:
                                continue

                            fused_time = ownship_state['time']
                            this_track_kf = self.xyz_kalman_filters[track_source][track_id]

                            # Skip if not past coasting threshold (This means that it will perform a coast update every => coasting_start_thresh seconds)
                            if fused_time - this_track_kf.prev_timestamp < self.coasting_start_thresh_EO:
                                continue

                            # Track age
                            this_track_age = fused_time - this_track_kf.track_init_timestamp

                            # Determine if track established and return termination threshold
                            if this_track_age <= self.track_establishment_age:
                                this_track_termination_sec = self.track_termination_sec[0]
                            else:
                                this_track_termination_sec = self.track_termination_sec[1]

                            # Terminate Track if termination threshold reached (if last measurement over track_termination_sec seconds ago)
                            if fused_time - this_track_kf.last_measured_time > this_track_termination_sec:
                                del self.xyz_kalman_filters[track_source][track_id]
                                del self.track_history[track_id]
                                del self.iso_decomp_guidance_hist[track_id]
                                terminated_track_ids.append(track_id)
                                print(f"EO TRACK {track_id} TERMINATED! MISSING TELEMETRY FOR {this_track_termination_sec}s")
                                continue

                            # Skip if track too new for coasting but not yet terminated (if track younger than track_establishment_age seconds)
                            if this_track_age <= self.track_establishment_age:
                                continue

                            # Otherwise, enter coasting mode
                            print("Coasting EO Intruder")
                            coasting = True
                            cam_ids = ['Coasted']
                            fused_mean = None
                            fused_cov = None

                        else:
                            # === If insufficient history, do nothing ===
                            continue   
                    else:
                        # === If no existing track and no detections, do nothing ===
                        continue         

                # Update fused item for this detection (will be superceded by next detection if queue had length > 1)
                eo_intruder_track_this = self.iterate_tracker_state(track_id, track_source, coasting, fused_time, cam_ids, fused_mean, fused_cov, new_init_histories)
                # This ensures that if there are multiple detections in the queue and a later detection returns an error, it retains the earlier detection track output
                if eo_intruder_track_this:
                    output_tracks[track_id] = eo_intruder_track_this
            
            else:
                # Iterate through all EO track Ids, including new and existing tracks
                for track_id in set(new_track_items.keys()) | set(self.xyz_kalman_filters[track_source].keys()):
                    if track_id in new_track_items:
                        item = new_track_items[track_id]
                        
                        coasting = False

                        cam_id = item['cam_id']
                        raw_range = item['distance']
                        range_bias, range_var, pixel_var = self.ownship.cameras[cam_id]['camera'].get_detected_uncertainty_params(raw_range)
                        pixel_std = np.sqrt(pixel_var)
                        range_std = np.sqrt(range_var)

                        mean_global_adjusted, cov_global = self.calculate_xyz_mean_and_covariance(
                            pixel_std, pixel_std, range_std, range_bias,
                            item['cx'], item['cy'], raw_range, cam_id, item['time']
                        )

                        fused_cov = cov_global
                        fused_mean = mean_global_adjusted
                        fused_time = item['time']
                        cam_ids = [cam_id]
                    else:
                        # === If no new detections but track also exists, Coasting ===
                        if len(self.track_history[track_id]) > 2:
                            # Only coast on last telemetry in the queue since ownship has the same timestamp for each item in this queue
                            # (high frequency of thread iteration means that in practice, ownship lag will be very small compared to any intruder in the same queue batch)
                            if queue_idx != len(detection_summaries) - 1:
                                continue

                            fused_time = ownship_state['time']
                            this_track_kf = self.xyz_kalman_filters[track_source][track_id]

                            # Skip if not past coasting threshold (This means that it will perform a coast update every => coasting_start_thresh seconds)
                            if fused_time - this_track_kf.prev_timestamp < self.coasting_start_thresh_EO:
                                continue

                            # Track age
                            this_track_age = fused_time - this_track_kf.track_init_timestamp

                            # Determine if track established and return termination threshold
                            if this_track_age <= self.track_establishment_age:
                                this_track_termination_sec = self.track_termination_sec[0]
                            else:
                                this_track_termination_sec = self.track_termination_sec[1]

                            # Terminate Track if termination threshold reached (if last measurement over track_termination_sec seconds ago)
                            if fused_time - this_track_kf.last_measured_time > this_track_termination_sec:
                                del self.xyz_kalman_filters[track_source][track_id]
                                del self.track_history[track_id]
                                del self.iso_decomp_guidance_hist[track_id]
                                terminated_track_ids.append(track_id)
                                print(f"EO TRACK {track_id} TERMINATED! MISSING TELEMETRY FOR {this_track_termination_sec}s")
                                continue

                            # Skip if track too new for coasting but not yet terminated (if track younger than track_establishment_age seconds)
                            if this_track_age <= self.track_establishment_age:
                                continue

                            # Otherwise, enter coasting mode
                            print("Coasting EO Intruder")
                            coasting = True
                            cam_ids = ['Coasted']
                            fused_mean = None
                            fused_cov = None

                        else:
                            # === If insufficient history, do nothing ===
                            continue            

                    # Update fused item for this detection (will be superceded by next detection if queue had length > 1)
                    eo_intruder_track_this = self.iterate_tracker_state(track_id, track_source, coasting, fused_time, cam_ids, fused_mean, fused_cov, new_init_histories)
                    # This ensures that if there are multiple detections in the queue and a later detection returns an error, it retains the earlier detection track output
                    if eo_intruder_track_this:
                        output_tracks[track_id] = eo_intruder_track_this

        #################################################
        ######## ADSB INTRUDER KALMAN FILTERS ###########
        #################################################

        # Map new ADS-B inputs by ID
        id_key = 'callsign'
        track_source = 'ADSB'
        inputs_by_id = {item[id_key]: item for item in adsb_intruders if id_key in item}

        # Existing ADSB KF ids
        existing_ids = set(self.xyz_kalman_filters[track_source].keys())

        # Iterate through all ADSB IDs, including new and existing
        all_ids = existing_ids | set(inputs_by_id.keys())
        for track_id in all_ids:          
            if track_id in inputs_by_id:
                coasting = False
                this_intr = inputs_by_id[track_id]
                intr_coord = np.array([this_intr["north_m"], this_intr["east_m"], this_intr["alt"]])
                nac_p = this_intr["NACp"]

                # Calculate ownship velocity components
                trk_rad = np.deg2rad(this_intr["trk"])
                vx_n = this_intr["ground_speed"] * np.cos(trk_rad)  # North component (m/s)
                vy_e = this_intr["ground_speed"] * np.sin(trk_rad)  # East component (m/s)                # Up component (m/s)

                adsb_pos_vel = np.array([this_intr["north_m"], this_intr["east_m"], this_intr["alt"],
                                        vx_n, vy_e, this_intr["vertical_speed"]])

                pos_var, vel_var = self.nacp_to_horizontal_variance(nacp = nac_p), 5.0 ** 2
                adsb_cov = np.diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var])

                fused_time = this_intr["time"]
                cam_ids = ['ADSB']
            else:
                fused_time = ownship_state['time']
                this_track_kf = self.xyz_kalman_filters[track_source][track_id]

                # Skip if not past coasting threshold (This means that it will perform a coast update every => coasting_start_thresh seconds)
                if fused_time - this_track_kf.prev_timestamp < self.coasting_start_thresh_ADSB:
                    continue

                # Track age
                this_track_age = fused_time - this_track_kf.track_init_timestamp

                # Determine if track established and return termination threshold
                if this_track_age <= self.track_establishment_age:
                    this_track_termination_sec = self.track_termination_sec[2]
                else:
                    this_track_termination_sec = self.track_termination_sec[3]

                # Terminate Track if termination threshold reached (if last measurement over track_termination_sec seconds ago)
                if fused_time - this_track_kf.last_measured_time > this_track_termination_sec:
                    del self.xyz_kalman_filters[track_source][track_id]
                    del self.track_history[track_id]
                    del self.iso_decomp_guidance_hist[track_id]
                    terminated_track_ids.append(track_id)
                    print(f"ADSB TRACK {track_id} TERMINATED! MISSING TELEMETRY FOR {this_track_termination_sec}s")
                    continue

                # Skip if track too new for coasting but not yet terminated (if track younger than track_establishment_age seconds)
                if this_track_age <= self.track_establishment_age:
                    continue

                print("Coasting ADSB Intruder")
                coasting = True
                adsb_pos_vel = None
                adsb_cov = None
                cam_ids = ['ADSB Coasted']

            adsb_tracker_output = self.iterate_tracker_state(track_id, track_source, coasting, fused_time, cam_ids, adsb_pos_vel, adsb_cov)

            if adsb_tracker_output:
                output_tracks[track_id] = adsb_tracker_output
         
        # Create tracker output object 
        tracker_output = output_tracks, {track_id: hist_this[-1] for track_id, hist_this in self.track_history.items() if track_id != "Ownship"}, terminated_track_ids

        # If no new output tracks, return before plotting
        if not output_tracks:
            return tracker_output

        # =========================
        # BEGIN: Faster plotting block (drop-in replacement)
        # =========================
        if self.plot_ADSB_instead_of_eo:
            track_id = 'INTRUDER'
            if track_id not in output_tracks:
                return tracker_output
            intruder_state = adsb_intruders[-1]
            fused_mean = intr_coord
            cam_ids = ["ADSB"]
            track_source = "ADSB"
        else:
            plot_ownship = False
            if plot_ownship:
                track_source = 'Ownship'
                track_id = "Ownship"
                fused_time = ownship_time
                cam_ids = ["Ownship"]
                coasting = False
                intruder_state = ownship_state
            else:
                track_source = "EO"
                if self.enforce_single_EO_intruder:
                    track_id = 1
                else:
                    if self.xyz_kalman_filters[track_source].keys():
                        track_id = list(self.xyz_kalman_filters[track_source].keys())[0]
                    else:
                        return tracker_output

        if self.plot and track_id in self.xyz_kalman_filters[track_source]:
            if track_id in output_tracks:
                fused_item = output_tracks[track_id]
            else:
                return tracker_output
            x, y, z = fused_item['abs_cartesian_x_y_z_smoothed']
            vx, vy, vz = fused_item['velocity_x_y_z_smoothed']
            kf_xyz = self.xyz_kalman_filters[track_source][track_id]
            std_xyz = fused_item['uncertainty_cartesian_x_y_z_smoothed']
            std_vxyz = fused_item['uncertainty_velocity_x_y_z_smoothed']
            cor_yx = fused_item['uncertainty_cartesian_yx_std_smoothed']
            cor_vyx = fused_item['uncertainty_velocity_yx_std_smoothed']

            # === Append latest point to history ===
            self.fused_xy_history.append((x, y, std_xyz[0], std_xyz[1]))
            if intruder_state:
                self.intr_xy_history.append((intruder_state['north_m'], intruder_state['east_m']))
            ownship_xyz = self.ownship.position(ownship_time=fused_time).flatten()

            current_cam_id_set = set(cam_ids)
            stds_plot = 2

            if daidalus_sugg_guidance:
                try:
                    self.debug_param_history.append(daidalus_sugg_guidance)
                except Exception:
                    self.debug_param_history.append([0, 0, 0, 0])
            else:
                self.debug_param_history.append([(x + 540) % 360 - 180 for x in self.ownship.orientation_rpy_deg()])

            # ------------- helpers (fast culling) -------------
            def _line_set_data_culled(line, xarr, yarr, ax, margin=0.0, max_points=4000):
                """Cull polyline to current axes limits (with margin) and optional downsample."""
                if xarr is None or yarr is None or len(xarr) == 0:
                    line.set_data([], [])
                    return
                xarr = np.asarray(xarr); yarr = np.asarray(yarr)
                x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
                m = (xarr >= x0 - margin) & (xarr <= x1 + margin) & (yarr >= y0 - margin) & (yarr <= y1 + margin)
                if not np.any(m):
                    line.set_data([], [])
                    return
                xv, yv = xarr[m], yarr[m]
                n = xv.size
                if n > max_points:
                    idx = np.linspace(0, n - 1, max_points).astype(int)
                    xv, yv = xv[idx], yv[idx]
                line.set_data(xv, yv)

            def _cov_ellipse_params(var_x, var_y, cov_xy, sigma_scale=1.0):
                """Return (width, height, angle_deg) for 2x2 covariance ellipse scaled by sigma_scale."""
                C = np.array([[var_x, cov_xy],
                            [cov_xy, var_y]], dtype=float)
                vals, vecs = np.linalg.eigh(C)
                order = vals.argsort()[::-1]
                vals = vals[order]; vecs = vecs[:, order]
                angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                width = 2.0 * sigma_scale * np.sqrt(max(vals[0], 0.0))
                height = 2.0 * sigma_scale * np.sqrt(max(vals[1], 0.0))
                return width, height, angle

            def _velocity_tip_polygon(ax, cx, cy, cov2x2, sigma_scale=1.0, npts=20):
                """
                Build (xs, ys) for the velocity-tip ellipse polygon centered at (cx,cy),
                culling points outside view (so we don't push invisible verts to the backend).
                """
                vals, vecs = np.linalg.eigh(cov2x2)
                order = vals.argsort()[::-1]
                vals = vals[order]; vecs = vecs[:, order]
                shape = np.diag(np.sqrt(np.maximum(vals, 0.0)) * sigma_scale)

                angs = np.linspace(0, 2*np.pi, npts, endpoint=True)
                unit = np.stack((np.cos(angs), np.sin(angs)), axis=0)  # (2, n)
                xy = vecs @ shape @ unit
                xs = cx + xy[0]; ys = cy + xy[1]

                # Cull to axes box (+small pad to keep the curve connected near edges)
                pad = 1e-9
                x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
                mask = (xs >= x0 - pad) & (xs <= x1 + pad) & (ys >= y0 - pad) & (ys <= y1 + pad)
                if not np.any(mask):
                    return np.array([]), np.array([])
                return xs[mask], ys[mask]

            # === Create or update persistent XY plot ===
            if self.fused_xy_fig is None:
                self.fused_xy_fig = plt.figure(figsize=(14, 6.5))
                gs = self.fused_xy_fig.add_gridspec(nrows=1, ncols=2, wspace=0.22)

                self.fused_xy_ax  = self.fused_xy_fig.add_subplot(gs[0, 0])
                self.fused_vel_ax = self.fused_xy_fig.add_subplot(gs[0, 1])

                # --- XY Position Plot ---
                self.fused_xy_ax.set_title("Fused Track in XY Plane")
                self.fused_xy_ax.set_xlabel("North (m)")
                self.fused_xy_ax.set_ylabel("East (m)")
                self.fused_xy_ax.grid(True)
                self.fused_xy_ax.axis("equal")

                # Ownship-to-fused-intruder line, useful for checking uncertainty orientation.
                x_own, y_own = ownship_xyz[0], ownship_xyz[1]
                self.ownship_line, = self.fused_xy_ax.plot([x_own, x], [y_own, y],
                                                        color='black', linestyle=':', linewidth=1.4,
                                                        zorder=1, label='Ownship-Intruder LOS')
                self.ownship_dot,  = self.fused_xy_ax.plot([x_own], [y_own],
                                                        marker='o', color='black', markersize=7, zorder=4, label='Ownship')

                # Paths + current points
                xy = np.array(self.fused_xy_history)
                if intruder_state:
                    xy_intr = np.array(self.intr_xy_history)
                self.fused_xy_path, = self.fused_xy_ax.plot(xy[:, 0], xy[:, 1], color=self.track_colors[0], label='Track')
                self.fused_xy_point = self.fused_xy_ax.plot(x, y, 'ro', label='Current')[0]

                # Measurement point if available
                try:
                    self.fused_xy_point_measured = self.fused_xy_ax.plot(fused_mean[0], fused_mean[1], 'rx',
                                                                        label='Measurement')[0]
                except Exception:
                    self.fused_xy_point_measured = self.fused_xy_ax.plot(np.nan, np.nan, 'rx',
                                                                        label='Measurement')[0]
                if intruder_state:
                    self.intr_xy_gt_point = self.fused_xy_ax.plot(xy_intr[-1][0], xy_intr[-1][1],
                                                                marker='x', color='darkblue', markersize=10,
                                                                label='Intruder Ground Truth')[0]
                    self.intr_gt_path, = self.fused_xy_ax.plot(xy_intr[:, 0], xy_intr[:, 1], color='blue',
                                                            label='Intruder Ground Truth')

                # Uncertainty ellipse (Kalman) @ (x,y)
                cov_xy = np.sign(cor_yx) * (np.abs(cor_yx) ** 2)
                w, h, ang = _cov_ellipse_params(stds_plot*std_xyz[0]**1*1, stds_plot*std_xyz[1]**1*1, cov_xy, sigma_scale=stds_plot)
                # The helper already multiplies, but keep consistent:
                width, height, angle = _cov_ellipse_params(std_xyz[0]**2, std_xyz[1]**2, cov_xy, sigma_scale=stds_plot)

                ell = Ellipse((x, y), width, height, angle=angle,
                            edgecolor='green', facecolor='none', lw=1.5, label=f'Kalman - {stds_plot}σ')
                ell.set_clip_on(True)  # ensure clipping to axes
                self.fused_xy_ax.add_patch(ell)
                self.fused_xy_uncertainty = ell

                # Measurement ellipse (only if not coasting)
                if not coasting:
                    vals_m, vecs_m = np.linalg.eigh(kf_xyz.R[0:2, 0:2])
                    order_m = vals_m.argsort()[::-1]
                    vals_m = vals_m[order_m]; vecs_m = vecs_m[:, order_m]
                    angle_m = np.degrees(np.arctan2(vecs_m[1, 0], vecs_m[0, 0]))
                    width_m, height_m = stds_plot * 2 * np.sqrt(vals_m)
                    ell_m = Ellipse((x, y), width_m, height_m, angle=angle_m,
                                    edgecolor='red', facecolor='none', lw=1.5, label=f'Measured - {stds_plot}σ', zorder=0.5)
                    ell_m.set_clip_on(True)
                    self.fused_xy_ax.add_patch(ell_m)
                    self.measured_xy_uncertainty = ell_m
                else:
                    # placeholder so attribute exists
                    dummy = Ellipse((0, 0), 0, 0, fill=False, visible=False)
                    self.fused_xy_ax.add_patch(dummy)
                    self.measured_xy_uncertainty = dummy

                padding = 400
                x_latest, y_latest = xy[-1, 0], xy[-1, 1]
                self.fused_xy_ax.set_xlim(x_latest - padding, x_latest + padding)
                self.fused_xy_ax.set_ylim(y_latest - padding, y_latest + padding)
                self.fused_xy_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, framealpha=0.85)

                # --- Velocity subplot ---
                self.fused_vel_ax.set_title("Velocity in VX-VY Plane")
                self.fused_vel_ax.set_xlabel("VX (m/s)")
                self.fused_vel_ax.set_ylabel("VY (m/s)")
                self.fused_vel_ax.grid(True)
                self.fused_vel_ax.axis("equal")

                self.fused_vel_history = [(vx, vy, std_vxyz[0], std_vxyz[1])]
                self.fused_vel_path, = self.fused_vel_ax.plot([vx], [vy], color='blue', label='Velocity Track')
                if intruder_state:
                    self.fused_vel_path_gt, = self.fused_vel_ax.plot(vx, vy, 'o', color='orange', label='Ground Truth')
                self.fused_vel_point = self.fused_vel_ax.plot(vx, vy, 'ro', label='Current')[0]

                circle_100 = Circle((0, 0), radius=100, edgecolor='gray', facecolor='none',
                                    linestyle='--', linewidth=1.0, label='100 m/s radius')
                circle_100.set_clip_on(True)
                self.fused_vel_ax.add_patch(circle_100)

                # Velocity covariance ellipse (as patch in velocity axes)
                cov_vxy = np.sign(cor_vyx) * (np.abs(cor_vyx) ** 2)
                width_v, height_v, angle_v = _cov_ellipse_params(std_vxyz[0]**2, std_vxyz[1]**2, cov_vxy, sigma_scale=stds_plot)
                ell_v = Ellipse((vx, vy), width_v, height_v, angle=angle_v,
                                edgecolor='orange', facecolor='none', lw=1.5, label=f'{stds_plot}σ')
                ell_v.set_clip_on(True)
                self.fused_vel_ax.add_patch(ell_v)
                self.fused_vel_uncertainty = ell_v

                padding_vel = 40
                self.fused_vel_ax.set_xlim(vx - padding_vel, vx + padding_vel)
                self.fused_vel_ax.set_ylim(vy - padding_vel, vy + padding_vel)
                self.fused_vel_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, framealpha=0.85)

                self.fused_xy_fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.18, wspace=0.22)
                plt.pause(0.01)

            else:
                # === Update existing artists ===

                # 2D line from ownship to fused intruder estimate.
                x_own,  y_own  = ownship_xyz[0], ownship_xyz[1]
                self.ownship_line.set_data([x_own, x], [y_own, y])
                self.ownship_dot.set_data([x_own], [y_own])

                # Handle camera set change (legend throttled)
                if current_cam_id_set != self.prev_cam_id_set:
                    color = self.track_colors[self.track_color_idx % len(self.track_colors)]
                    label = f'Cameras: {sorted(current_cam_id_set)}'
                    self.fused_xy_path, = self.fused_xy_ax.plot([x], [y], color=color, label=label)
                    self.track_color_idx += 1
                    self.prev_cam_id_set = current_cam_id_set

                    handles, labels = self.fused_xy_ax.get_legend_handles_labels()
                    cam_entries = [(h, l) for h, l in zip(handles, labels) if l.startswith("Cameras")]
                    cam_entries = cam_entries[-4:]
                    other_entries = [(h, l) for h, l in zip(handles, labels) if not l.startswith("Cameras")]
                    self.fused_xy_ax.legend(
                        handles=[h for h, _ in other_entries + cam_entries],
                        labels=[l for _, l in other_entries + cam_entries],
                        loc='upper center', bbox_to_anchor=(0.5, -0.16),
                        ncol=2, fontsize=8, framealpha=0.85
                    )
                else:
                    old_x, old_y = self.fused_xy_path.get_data()
                    self.fused_xy_path.set_data(np.append(old_x, x), np.append(old_y, y))

                # Update path and current points
                xy = np.array(self.fused_xy_history)
                if intruder_state:
                    xy_intr = np.array(self.intr_xy_history)
                self.fused_xy_point.set_data([x], [y])
                try:
                    self.fused_xy_point_measured.set_data([fused_mean[0]], [fused_mean[1]])
                except Exception:
                    self.fused_xy_point_measured.set_data([np.nan], [np.nan])
                if intruder_state:
                    self.intr_xy_gt_point.set_data([xy_intr[-1][0]], [xy_intr[-1][1]])

                # Update XY axes limits around latest
                padding = 400
                x_latest, y_latest = xy[-1, 0], xy[-1, 1]
                self.fused_xy_ax.set_xlim(x_latest - padding, x_latest + padding)
                self.fused_xy_ax.set_ylim(y_latest - padding, y_latest + padding)

                # Cull long polylines to the visible window
                _line_set_data_culled(self.fused_xy_path, xy[:, 0], xy[:, 1], self.fused_xy_ax, margin=50.0, max_points=1000)
                if intruder_state:
                    _line_set_data_culled(self.intr_gt_path, xy_intr[:, 0], xy_intr[:, 1], self.fused_xy_ax, margin=50.0, max_points=1000)

                # --- Update uncertainty ellipse (XY) ---
                cov_xy = np.sign(cor_yx) * (np.abs(cor_yx) ** 2)
                width, height, angle = _cov_ellipse_params(std_xyz[0]**2, std_xyz[1]**2, cov_xy, sigma_scale=stds_plot)
                self.fused_xy_uncertainty.set_width(width)
                self.fused_xy_uncertainty.set_height(height)
                self.fused_xy_uncertainty.angle = angle
                self.fused_xy_uncertainty.set_center((x, y))
                self.fused_xy_uncertainty.set_clip_on(True)

                try:
                    if not coasting:
                        vals_m, vecs_m = np.linalg.eigh(kf_xyz.R[0:2, 0:2])
                        order_m = vals_m.argsort()[::-1]
                        vals_m = vals_m[order_m]; vecs_m = vecs_m[:, order_m]
                        angle_m = np.degrees(np.arctan2(vecs_m[1, 0], vecs_m[0, 0]))
                        width_m, height_m = stds_plot * 2 * np.sqrt(vals_m)
                        self.measured_xy_uncertainty.set_width(width_m)
                        self.measured_xy_uncertainty.set_height(height_m)
                        self.measured_xy_uncertainty.angle = angle_m
                        self.measured_xy_uncertainty.set_center((x, y))
                        self.measured_xy_uncertainty.set_visible(True)
                    else:
                        self.measured_xy_uncertainty.set_visible(False)
                except:
                    self.measured_xy_uncertainty.set_visible(False)

                # --- Velocity vectors and tip-ellipse drawn on XY axes ---
                vel_mag = np.hypot(vx, vy)
                # Remove previous arrows if they exist
                if hasattr(self, "velocity_arrows"):
                    for arrow in self.velocity_arrows:
                        arrow.remove()

                arrow_est = self.fused_xy_ax.arrow(
                    x, y, vx, vy, head_width=10, head_length=15, fc='purple', ec='purple', zorder=3,
                    label='Estimated Velocity'
                )
                if intruder_state:
                    gs = intruder_state['ground_speed']
                    trk_deg = intruder_state['trk']
                    trk_rad = np.radians(trk_deg)
                    gt_dx = gs * np.cos(trk_rad); gt_dy = gs * np.sin(trk_rad)
                    arrow_gt = self.fused_xy_ax.arrow(
                        x, y, gt_dx, gt_dy, head_width=10, head_length=15, fc='green', ec='green', zorder=3,
                        label='Ground Truth Velocity'
                    )
                    self.velocity_arrows = [arrow_est, arrow_gt]
                else:
                    self.velocity_arrows = [arrow_est]

                # Velocity-tip ellipse polygon in XY space (centered at x+vx, y+vy)
                cov_vxy = np.sign(cor_vyx) * (np.abs(cor_vyx) ** 2)
                cov2 = np.array([[std_vxyz[0]**2, cov_vxy],
                                [cov_vxy,         std_vxyz[1]**2]], dtype=float)
                cx_v, cy_v = x + vx, y + vy
                xs_poly, ys_poly = _velocity_tip_polygon(self.fused_xy_ax, cx_v, cy_v, cov2, sigma_scale=stds_plot, npts=20)
                if hasattr(self, "velocity_tip_ellipse"):
                    if xs_poly.size >= 3:
                        self.velocity_tip_ellipse.set_data(xs_poly, ys_poly)
                        self.velocity_tip_ellipse.set_visible(True)
                    else:
                        self.velocity_tip_ellipse.set_visible(False)
                else:
                    if xs_poly.size >= 3:
                        self.velocity_tip_ellipse, = self.fused_xy_ax.plot(
                            xs_poly, ys_poly, color='blue', linestyle='--',
                            label='Velocity Likelihood Ellipse', zorder=3
                        )
                    else:
                        self.velocity_tip_ellipse, = self.fused_xy_ax.plot([], [], color='blue', linestyle='--',
                                                                            label='Velocity Likelihood Ellipse', zorder=3)
                        self.velocity_tip_ellipse.set_visible(False)

                # === Velocity subplot updates ===
                self.fused_vel_history.append((vx, vy, std_vxyz[0], std_vxyz[1]))
                vxy = np.array(self.fused_vel_history)
                self.fused_vel_path.set_data(vxy[-100:, 0], vxy[-100:, 1])
                if intruder_state:
                    self.fused_vel_path_gt.set_data([gt_dx], [gt_dy])
                self.fused_vel_point.set_data([vx], [vy])

                # Update velocity covariance ellipse (as patch)
                width_v, height_v, angle_v = _cov_ellipse_params(std_vxyz[0]**2, std_vxyz[1]**2, cov_vxy, sigma_scale=stds_plot)
                self.fused_vel_uncertainty.set_width(width_v)
                self.fused_vel_uncertainty.set_height(height_v)
                self.fused_vel_uncertainty.angle = angle_v
                self.fused_vel_uncertainty.set_center((vx, vy))

                # Set padded bounds around latest velocity
                padding_vel = 40
                self.fused_vel_ax.set_xlim(vx - padding_vel, vx + padding_vel)
                self.fused_vel_ax.set_ylim(vy - padding_vel, vy + padding_vel)

                # === Time series (speed / debug) ===
                # Compute acceleration magnitude (optional smoothing)
                if len(self.track_history[track_id]) >= 10:
                    v_prev = self.track_history[track_id][-10]['velocity_x_y_z_smoothed']
                    v_now = kf_xyz.x[3:6].reshape(3,)
                    dt = 0.1
                    a_norm = np.linalg.norm((v_now - v_prev) / dt)
                    self.accel_history.append(a_norm)
                    acc_min = np.min(self.accel_history[-10:])
                    acc_max = np.max(self.accel_history[-10:])
                else:
                    a_norm = np.nan; acc_min = np.nan; acc_max = np.nan

                self.accel2_history.append(np.nan)
                self.speed_history.append(vel_mag)
                self.time_history.append(fused_time)

                vel_min = np.min(self.speed_history)
                vel_max = np.max(self.speed_history)

                # Separation Distance. Prefer ground-truth/reference intruder_state when
                # available; otherwise use the fused tracker estimate already being plotted.
                sep_dist = np.nan
                ver_sep = np.nan
                if intruder_state:
                    intruder_xyz_for_sep = np.array([
                        intruder_state['north_m'],
                        intruder_state['east_m'],
                        intruder_state.get('alt', z),
                    ], dtype=float)
                else:
                    intruder_xyz_for_sep = np.array([x, y, z], dtype=float)

                if intruder_xyz_for_sep.size >= 3:
                    sep_dist = np.linalg.norm(intruder_xyz_for_sep[:2] - np.array(ownship_xyz[:2], dtype=float))
                    ver_sep = abs(intruder_xyz_for_sep[2] - ownship_xyz[2])
                    self.min_xy_separation = [min(self.min_xy_separation[0], sep_dist),
                                            min(self.min_xy_separation[1], ver_sep)]

                sep_text = f"{sep_dist:.0f}" if np.isfinite(sep_dist) else "n/a"
                min_sep_text = f"{self.min_xy_separation[0]:.0f}" if np.isfinite(self.min_xy_separation[0]) else "n/a"
                ver_text = f"{ver_sep:.0f}" if np.isfinite(ver_sep) else "n/a"
                min_ver_text = f"{self.min_xy_separation[1]:.0f}" if np.isfinite(self.min_xy_separation[1]) else "n/a"

                velocity_text = (
                    f"Time: {fused_time:.1f} s\n"
                    f"Speed: {vel_mag:.1f} m/s ({vel_min:.1f}, {vel_max:.1f})\n"
                    f"Accel: {a_norm:.1f} m/s² ({acc_min:.1f}, {acc_max:.1f})\n"
                    f"Hor Separation: {sep_text} m ({min_sep_text})\n"
                    f"Ver Separation: {ver_text} m ({min_ver_text})"
                )

                if not hasattr(self, "velocity_label"):
                    self.velocity_label = self.fused_xy_ax.text(
                        0.02, 0.98, velocity_text, transform=self.fused_xy_ax.transAxes,
                        va='top', ha='left', fontsize=10, bbox=dict(facecolor='white', edgecolor='gray')
                    )
                else:
                    self.velocity_label.set_text(velocity_text)

                if intruder_state:
                    self.gt_speed_history.append(gs)

                self.fused_xy_fig.canvas.draw()
                self.fused_xy_fig.canvas.flush_events()
                self.plot_update_count += 1
                if self.plot_save_path and np.isfinite(fused_time):
                    plot_second = int(np.floor(fused_time))
                    if plot_second >= 0 and plot_second != self.plot_last_saved_second:
                        self.save_tracker_plot(self.tracker_plot_frame_path(self.plot_save_path, plot_second), dpi=150)
                        self.plot_last_saved_second = plot_second
                plt.pause(0.01)
        # =========================
        # END: Faster plotting block
        # =========================

        return tracker_output

    def calculate_xyz_mean_and_covariance(
            self,
            del_x,
            del_y,
            del_d,
            bias_d,
            pixel_x,
            pixel_y,
            range_input,
            cam_id,
            time,
            #TODO: Make ownship and camera position + orientation uncertainty propagation higher order then first term
            # if required. Will need to input the position and orientation of the camera and ownship if it becomes >= second order.
            cam_pos_cov=None,   # 3x3, camera position covariance in ownship frame [m^2]
            cam_rpy_cov=None,   # 3x3, camera RPY covariance [rad^2]
            own_pos_cov=None,   # 3x3, ownship position covariance in global frame [m^2]
            own_rpy_cov=None    # 3x3, ownship RPY covariance [rad^2]
        ):

        ######################################################################
        ##### Intruder XYZ Position Mean in Global Frame with Range Bias #####
        ######################################################################

        adjusted_range = range_input - bias_d
        mean_global_adjusted = self.ownship.map_pixel_views_to_global_cartesian_coords(cam_id, np.array([[pixel_x, pixel_y, adjusted_range, time]]))[0]

        ############################################################
        ##### Intruder XYZ Position Covariance in Camera Frame #####
        ############################################################
         
        #TODO: Replace this with more principled logic
        # Add error buffer right now to account for kalman filter and affine transform induced errors
        del_x += 50# + range_input * (100/3000)
        del_y += 50# + range_input * (100/3000)
        cam = self.ownship.cameras[cam_id]
        cam_obj = cam['camera']
        
        k_x = (2 / cam_obj.pixels_x) * np.tan(np.radians(cam_obj.fov_x_deg)/2)
        k_y = (2 / cam_obj.pixels_y) * np.tan(np.radians(cam_obj.fov_y_deg)/2)
        
        X = pixel_x - (cam_obj.pixels_x / 2)
        # Flip Y since pixel coord frame has Y increasing down the image
        Y = -(pixel_y - (cam_obj.pixels_y / 2))
        d = range_input
        
        ###### Uncertainty in x (Cartesian) #######
        
        x_X = -(d * k_x ** 2 * X) / ((k_x ** 2 * X ** 2 + 1) ** (3/2))
        x_XX = -(d * k_x ** 2 * (1 - 2 * k_x ** 2 * X ** 2)) / ((k_x ** 2 * X ** 2 + 1) ** (5/2))
        x_d = 1 / ((k_x ** 2 * X ** 2 + 1) ** 0.5)
        x_Xd = -(k_x ** 2 * X) / ((k_x ** 2 * X ** 2 + 1) ** (3/2))
        
        var_x = x_X ** 2 * del_x ** 2 + x_d ** 2 * del_d ** 2 + 0.5 * (x_XX ** 2 * del_x ** 4) + x_Xd ** 2 * del_x ** 2 * del_d ** 2
        
        ###### Uncertainty in y (Cartesian) #######
        
        y_X = (d * k_x) / ((k_x ** 2 * X ** 2 + 1) ** (3/2))
        y_XX = (3 * d * k_x ** 5 * X ** 3) / ((k_x ** 2 * X ** 2 + 1) ** (5/2)) - (3 * d * k_x ** 3 * X) / ((k_x ** 2 * X ** 2 + 1) ** (3/2))
        y_d = k_x * X / ((k_x ** 2 * X ** 2 + 1) ** 0.5)
        y_Xd = k_x / ((k_x ** 2 * X ** 2 + 1) ** (3/2))
        
        var_y = y_X ** 2 * del_x ** 2 + y_d ** 2 * del_d ** 2 + 0.5 * (y_XX ** 2 * del_x ** 4) + y_Xd ** 2 * del_x ** 2 * del_d ** 2
        
        ###### Uncertainty in z (Cartesian) #######

        z_X = -(d * k_x ** 2 * k_y * X * Y) / (k_x ** 2 * X ** 2 + 1) ** (3/2)
        z_Y = (k_y * d) / (k_x ** 2 * X ** 2 + 1) ** 0.5
        z_d = (k_y * Y) / (k_x ** 2 * X ** 2 + 1) ** 0.5
        z_XX = -(d * k_x ** 2 * k_y * Y * (1 - 2 * k_x ** 2 * X ** 2)) / (k_x ** 2 * X ** 2 + 1) ** (5/2)
        z_XY = -(k_y * k_x ** 2 * d * X) / (k_x ** 2 * X ** 2 + 1) ** (3/2)
        z_Xd = -(k_y * k_x ** 2 * X * Y) / (k_x ** 2 * X ** 2 + 1) ** (3/2)
        z_Yd = k_y / (k_x ** 2 * X ** 2 + 1) ** 0.5
        
        var_z = (z_X ** 2 * del_x ** 2 + z_Y ** 2 * del_y ** 2 + z_d ** 2 * del_d ** 2
                + 0.5 * (z_XX ** 2 * del_x ** 4)
                + z_XY ** 2 * del_x ** 2 * del_y ** 2 + z_Xd ** 2 * del_x ** 2 * del_d ** 2 + z_Yd ** 2 * del_y ** 2 * del_d ** 2)
        
        # --- Covariances ---
        cov_xy = x_X * y_X * del_x ** 2 + x_d * y_d * del_d ** 2 + 0.5 * x_XX * y_XX * del_x ** 4 + x_Xd * y_Xd * del_x ** 2 * del_d ** 2

        cov_xz = x_X * z_X * del_x ** 2 + x_d * z_d * del_d ** 2 + 0.5 * x_XX * z_XX * del_x ** 4 + x_Xd * z_Xd * del_x ** 2 * del_d ** 2

        cov_yz = y_X * z_X * del_x ** 2 + y_d * z_d * del_d ** 2 + 0.5 * y_XX * z_XX * del_x ** 4 + y_Xd * z_Xd * del_x ** 2 * del_d ** 2

        cov_local = np.array([
            [var_x, cov_xy, cov_xz],
            [cov_xy, var_y, cov_yz],
            [cov_xz, cov_yz, var_z]
        ])
    
        ########################################################################
        ##### Transform Intruder XYZ Position Covariance into Global Frame #####
        ########################################################################

        # --- Base rotation (no orientation uncertainty) ---

        # Get orientation of camera and ownship in their own respective reference frames
        rpy_local = np.radians(cam["orientation_rpy_deg"])
        rpy_ownship = np.radians(self.ownship.orientation_rpy_deg(ownship_time=time))

        # Get transformation from camera local to global
        R_cam = self.ownship.intrinsic_neu_rpy_to_matrix(*rpy_local)
        R_own = self.ownship.intrinsic_neu_rpy_to_matrix(*rpy_ownship)
        R_total = R_own @ R_cam

        # Transform covariance from camera local to global
        cov_global = R_total @ cov_local @ R_total.T

        ##############################################################################################################
        ##### Additional Covariance Contributions Due to Ownship + Intruder Position and Orientation Uncertainty #####
        ##############################################################################################################

        # If no pose covariances given, just return what we already have
        if all(c is None for c in (cam_pos_cov, cam_rpy_cov, own_pos_cov, own_rpy_cov)):
            return mean_global_adjusted, cov_global

        # Precompute geometry used by both mean and Jacobians
        # Recreate the same ray used in map_pixel_views_to_global_cartesian_coords
        fx = cam_obj.pixels_x / (2 * np.tan(np.radians(cam_obj.fov_x_deg / 2)))
        fy = cam_obj.pixels_y / (2 * np.tan(np.radians(cam_obj.fov_y_deg / 2)))
        cx = cam_obj.pixels_x / 2
        cy = cam_obj.pixels_y / 2

        x_cam = 1.0
        y_cam = (pixel_x - cx) / fx
        z_cam = -(pixel_y - cy) / fy
        ray_cam = np.array([x_cam, y_cam, z_cam])
        ray_cam_norm = ray_cam / np.linalg.norm(ray_cam)

        cam_pos_local = np.array(cam["position_xyz"])      # camera in ownship frame
        own_pos_global = self.ownship.position(ownship_time=time)       # ownship in global frame

        def compute_point(cam_rpy, own_rpy, cam_pos, own_pos_g):
            """
            Compute global point for given camera/ownship pose.

            cam_pos      : camera position in ownship frame (3,)
            own_pos_g    : ownship position in global frame (3,)
            adjusted_range  : horizontal distance in the global XY plane
                        between the camera and the target (same definition
                        as horizontal_ranges in map_pixel_views_to_global_cartesian_coords)
            """
            # Same rotations as in the main mapping
            R_cam_ = self.ownship.intrinsic_neu_rpy_to_matrix(*cam_rpy)
            R_own_ = self.ownship.intrinsic_neu_rpy_to_matrix(*own_rpy)
            R_tot_ = R_own_ @ R_cam_  # camera → global

            # Global ray direction (already normalized in camera frame)
            ray_global = R_tot_ @ ray_cam_norm

            # Camera origin in global frame
            cam_global_pos_ = R_own_ @ cam_pos + own_pos_g

            # Scale so that the *horizontal* (XY) component has length = horiz_range
            horiz_norm = np.linalg.norm(ray_global[:2])
            s = adjusted_range / (horiz_norm + 1e-8)  # avoid div by zero

            # Final global point
            return cam_global_pos_ + ray_global * s

        # ---- Jacobians wrt camera and ownship RPY (central finite differences) ----
        eps_r = 1e-4  # radians; central diff usually tolerates a slightly larger step than 1e-6

        J_cam_rpy = np.zeros((3, 3))
        if cam_rpy_cov is not None:
            for j in range(3):  # roll, pitch, yaw
                dtheta = np.zeros(3)
                dtheta[j] = eps_r

                # f(theta + eps)
                p_plus = compute_point(rpy_local + dtheta, rpy_ownship, cam_pos_local, own_pos_global)
                # f(theta - eps)
                p_minus = compute_point(rpy_local - dtheta, rpy_ownship, cam_pos_local, own_pos_global)

                # central difference: (f(+eps) - f(-eps)) / (2*eps)
                J_cam_rpy[:, j] = (p_plus - p_minus) / (2.0 * eps_r)

        J_own_rpy = np.zeros((3, 3))
        if own_rpy_cov is not None:
            for j in range(3):  # roll, pitch, yaw
                dtheta = np.zeros(3)
                dtheta[j] = eps_r

                # f(theta + eps)
                p_plus = compute_point(rpy_local, rpy_ownship + dtheta, cam_pos_local, own_pos_global)
                # f(theta - eps)
                p_minus = compute_point(rpy_local, rpy_ownship - dtheta, cam_pos_local, own_pos_global)

                # central difference: (f(+eps) - f(-eps)) / (2*eps)
                J_own_rpy[:, j] = (p_plus - p_minus) / (2.0 * eps_r)

        # ---- Build extra covariance from pose uncertainties ----
        extra_cov = np.zeros((3, 3))

        if cam_rpy_cov is not None:
            extra_cov += J_cam_rpy @ cam_rpy_cov @ J_cam_rpy.T

        if own_rpy_cov is not None:
            extra_cov += J_own_rpy @ own_rpy_cov @ J_own_rpy.T

        # Position covariances are just additive, but camera must be transformed into global coords first

        if cam_pos_cov is not None:
            extra_cov += R_own @ cam_pos_cov @ R_own.T

        if own_pos_cov is not None:
            extra_cov += own_pos_cov

        cov_global += extra_cov

        return mean_global_adjusted, cov_global

    def process_ownship_state(self, ownship_mavlink_position, ownship_mavlink_attitude):
        if self.home_lat_lon is None:
            self.home_lat_lon = (ownship_mavlink_position['lat_deg'], ownship_mavlink_position['lon_deg'])
        
        os_neu = navpy.lla2ned(
            ownship_mavlink_position['lat_deg'],
            ownship_mavlink_position['lon_deg'],
            -ownship_mavlink_position['alt_msl_m'],
            self.home_lat_lon[0],
            self.home_lat_lon[1],
            0)

        ownship_state_processed = {
            "lat": ownship_mavlink_position['lat_deg'],
            "lon": ownship_mavlink_position['lon_deg'],
            "alt": ownship_mavlink_position['alt_msl_m'],
            "trk": ownship_mavlink_position['hdg_deg'],
            "north_m": os_neu[0],
            "east_m": os_neu[1],
            "ground_speed": np.sqrt(ownship_mavlink_position['ground_vx_m_s']**2 + ownship_mavlink_position['ground_vy_m_s']**2),
            "vertical_speed": ownship_mavlink_position['ground_vz_m_s'],
            "time": ownship_mavlink_position['timestamp'],
            "rl": ownship_mavlink_attitude['roll_deg'], # SUM (Roll)
            "pt": ownship_mavlink_attitude['pitch_deg'], # SUM (Pitch)
            "yw": ownship_mavlink_attitude['yaw_deg'], # SUM (Yaw)
            "wind_dir": 0,
            "wind_spd": 0
        }

        # self.ownship.set_state(
        #     (os_neu[0], os_neu[1], ownship_state_processed["alt"]), 
        #     (ownship_state_processed["rl"], ownship_state_processed["pt"], ownship_state_processed["yw"]),
        #     ownship_state_processed["time"]
        #     )

        return ownship_state_processed
    
    def process_tracked_ownship_state(self, tracked_ownship_state):
        time = tracked_ownship_state['time']
        ground_speed = np.sqrt(tracked_ownship_state['velocity_x_y_z_smoothed'][0]**2 + tracked_ownship_state['velocity_x_y_z_smoothed'][1]**2)
        roll_deg, pitch_deg, yaw_deg = self.ownship.orientation_rpy_deg(ownship_time = time)

        tracked_ownship_state_processed = {
            "lat": tracked_ownship_state['lat'],
            "lon": tracked_ownship_state['lon'],
            "alt": tracked_ownship_state['abs_cartesian_x_y_z_smoothed'][2],
            "trk": tracked_ownship_state['trk_smoothed'],
            "north_m": tracked_ownship_state['abs_cartesian_x_y_z_smoothed'][0],
            "east_m": tracked_ownship_state['abs_cartesian_x_y_z_smoothed'][1],
            "ground_speed": ground_speed,
            "vertical_speed": tracked_ownship_state['velocity_x_y_z_smoothed'][2],
            "time": time,
            "rl": roll_deg, # SUM (Roll)
            "pt": pitch_deg, # SUM (Pitch)
            "yw": yaw_deg, # SUM (Yaw)
            "wind_dir": 0,
            "wind_spd": 0
        }

        return tracked_ownship_state_processed
    
    def process_adsb_intruder_states(self, adsb_intruders, ownship_key = "own"):
        if self.home_lat_lon is None:
            return []
        
        processed_adsb_intruders = []

        for adsb_id, adsb_intr in adsb_intruders.items():
            # Exclude ownship from adsb intruders
            if adsb_id == ownship_key:
                continue

            adsb_neu = navpy.lla2ned(
                adsb_intr['int_lat'],
                adsb_intr['int_lon'],
                -adsb_intr['int_alt_m'],
                self.home_lat_lon[0],
                self.home_lat_lon[1],
                0)
            
            processed = {"lat": adsb_intr['int_lat'],
                "lon": adsb_intr['int_lon'],
                "alt": adsb_intr['int_alt_m'],
                "trk": adsb_intr['int_trk_deg'],
                "north_m": adsb_neu[0],
                "east_m": adsb_neu[1],
                "ground_speed": adsb_intr['int_gs_mps'],
                "vertical_speed": adsb_intr['int_vs_mps'],
                "type": "ADSB",
                "callsign": adsb_intr['int_callsign'],
                "id": 'Placeholder',
                "time": adsb_intr['time'],
                "NIC": adsb_intr['NIC'],
                "NACp": adsb_intr['NACp'],
                "SIL": adsb_intr['SIL'],
                "SDA": adsb_intr['SDA']
            }

            processed_adsb_intruders.append(processed)

        return processed_adsb_intruders

    def _update_constituent_layers(self, intruder_id, intr_cf,
                    intr_la_no, intr_lo_ea,
                    pos_centers_w, pos_covs_iso3,
                    vel_centers_w, vel_covs_iso3,
                    pos_cov3=None, vel_cov3=None,
                    stds_plot=2):
        """Draw/update constituent ellipses on the existing plots using your axes:
        self.fused_xy_ax and self.fused_vel_ax."""

        # Create storage dicts if missing
        if not hasattr(self, '_constituent_pos_patches'):
            self._constituent_pos_patches = {}  # intruder_id -> list[Patch]
        if not hasattr(self, '_constituent_vel_patches'):
            self._constituent_vel_patches = {}
        if not hasattr(self, '_constituent_parent_pos_patches'):
            self._constituent_parent_pos_patches = {}
        if not hasattr(self, '_constituent_parent_vel_patches'):
            self._constituent_parent_vel_patches = {}

        def _ellipse_from_cov2(cov2, center, edgecolor, linestyle, label):
            vals, vecs = np.linalg.eigh(np.asarray(cov2, dtype=float))
            order = vals.argsort()[::-1]
            vals = np.maximum(vals[order], 0.0)
            vecs = vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width, height = 2.0 * stds_plot * np.sqrt(vals)
            return Ellipse(center, width, height, angle=angle,
                           edgecolor=edgecolor, facecolor='none', lw=1.2,
                           linestyle=linestyle, alpha=0.85, label=label)

        # ---- Position (XY) on your existing fused_xy_ax ----
        if (hasattr(self, 'fused_xy_ax') and self.fused_xy_ax is not None
                and isinstance(intr_cf, str) and intr_cf.startswith('NEU')):
            # Remove old
            for p in self._constituent_pos_patches.get(intruder_id, []):
                try: p.remove()
                except Exception: pass
            for p in self._constituent_parent_pos_patches.get(intruder_id, []):
                try: p.remove()
                except Exception: pass
            pos_patches = []
            parent_pos_patches = []

            base_x = float(intr_la_no)
            base_y = float(intr_lo_ea)

            if pos_cov3 is not None:
                e_parent = _ellipse_from_cov2(
                    np.asarray(pos_cov3, dtype=float)[0:2, 0:2],
                    (base_x, base_y),
                    edgecolor='darkred',
                    linestyle='--',
                    label=f'Parent ellipsoid projection - {stds_plot} sigma',
                )
                e_parent.set_zorder(0.35)
                e_parent.set_clip_on(True)
                self.fused_xy_ax.add_patch(e_parent)
                parent_pos_patches.append(e_parent)

            for pc, cov_pos3 in zip(pos_centers_w, pos_covs_iso3):
                cx = base_x + float(pc[0])
                cy = base_y + float(pc[1])
                std_pos = float(np.sqrt(max(cov_pos3[0, 0], 0.0)))   # isotropic -> same on x/y
                diam = 2.0 * stds_plot * std_pos                     # Ellipse takes diameters

                e = Ellipse((cx, cy), diam, diam, angle=0.0,
                            edgecolor='teal', facecolor='none', lw=1.0,
                            alpha=0.6)
                e.set_zorder(0.4)          # behind main tracks
                e.set_clip_on(True)        # clip to current axes view
                self.fused_xy_ax.add_patch(e)
                pos_patches.append(e)

            self._constituent_pos_patches[intruder_id] = pos_patches
            self._constituent_parent_pos_patches[intruder_id] = parent_pos_patches

        # ---- Velocity (VX-VY) on your existing fused_vel_ax ----
        if hasattr(self, 'fused_vel_ax') and self.fused_vel_ax is not None:
            # Remove old
            for p in self._constituent_vel_patches.get(intruder_id, []):
                try: p.remove()
                except Exception: pass
            for p in self._constituent_parent_vel_patches.get(intruder_id, []):
                try: p.remove()
                except Exception: pass
            vel_patches = []
            parent_vel_patches = []

            if vel_cov3 is not None and len(vel_centers_w) > 0:
                base_vx = float(vel_centers_w[0][0])
                base_vy = float(vel_centers_w[0][1])
                e_parent = _ellipse_from_cov2(
                    np.asarray(vel_cov3, dtype=float)[0:2, 0:2],
                    (base_vx, base_vy),
                    edgecolor='darkorange',
                    linestyle='--',
                    label=f'Parent velocity ellipsoid projection - {stds_plot} sigma',
                )
                e_parent.set_zorder(0.35)
                e_parent.set_clip_on(True)
                self.fused_vel_ax.add_patch(e_parent)
                parent_vel_patches.append(e_parent)

            # Velocity ellipses are centered at (mean vx, mean vy) + constituent offsets
            # Caller should pass centers/covs for velocity constituents.
            for vc, cov_vel3 in zip(vel_centers_w, vel_covs_iso3):
                cx = float(vc[0])
                cy = float(vc[1])
                std_vel = float(np.sqrt(max(cov_vel3[0, 0], 0.0)))
                diam_v = 2.0 * stds_plot * std_vel

                e = Ellipse((cx, cy), diam_v, diam_v, angle=0.0,
                            edgecolor='slateblue', facecolor='none', lw=1.0,
                            alpha=0.6)
                e.set_zorder(0.4)
                e.set_clip_on(True)
                self.fused_vel_ax.add_patch(e)
                vel_patches.append(e)

            self._constituent_vel_patches[intruder_id] = vel_patches
            self._constituent_parent_vel_patches[intruder_id] = parent_vel_patches

    def create_daidalus_input(self, ownship_state, intruder_states):
        
        if self.cartesian_centered_aligned[0]:
            coord_frame = "NEU"
            if self.cartesian_centered_aligned[1]:
                la_no = 0.0
                lo_ea = 0.0
                if self.cartesian_centered_aligned[2]:
                    coord_frame = "NEU_aligned"
            else:
                la_no = ownship_state['north_m']
                lo_ea = ownship_state['east_m']  
        else:
            la_no = ownship_state['lat']
            lo_ea = ownship_state['lon']
            coord_frame = "LLA"

        ownship_state_input = {
            "la_no": la_no,
            "lo_ea": lo_ea,
            "a": ownship_state["alt"],
            "t": ownship_state["trk"],
            "gs": ownship_state['ground_speed'],
            "vs": ownship_state['vertical_speed'],
            "ti": ownship_state['time'],
            "cf": coord_frame
        }
        
        intruder_states_input = []

        for intr_id, intruder_state in intruder_states.items():
            if self.cartesian_centered_aligned[0]:
                coord_frame = "NEU"
                if self.cartesian_centered_aligned[1]:
                    if self.cartesian_centered_aligned[2]:
                        intruder_rel = self.intruder_rel_x_y_z(ownship_state, intruder_state, heading_ref_deg = ownship_state["trk"])
                    else:
                        intruder_rel = self.intruder_rel_x_y_z(ownship_state, intruder_state, heading_ref_deg = 0)

                    la_no = intruder_rel['x']
                    lo_ea = intruder_rel['y']
                    trk = intruder_rel['t']

                    std_x, std_y, std_z = 0.0, 0.0, 0.0
                    std_vx, std_vy, std_vz = 0.0, 0.0, 0.0
                    std_yx = 0.0
                    std_vyx = 0.0
                else:
                    la_no = intruder_state['abs_cartesian_x_y_z_smoothed'][0]
                    lo_ea = intruder_state['abs_cartesian_x_y_z_smoothed'][1]
                    trk = intruder_state['trk_smoothed']

                    std_x, std_y, std_z = intruder_state['uncertainty_cartesian_x_y_z_smoothed']
                    std_vx, std_vy, std_vz = intruder_state['uncertainty_velocity_x_y_z_smoothed']
                    std_yx = intruder_state['uncertainty_cartesian_yx_std_smoothed']
                    std_vyx = intruder_state['uncertainty_velocity_yx_std_smoothed']
            else:
                la_no = intruder_state['lat']
                lo_ea = intruder_state['lon']
                trk = intruder_state['trk_smoothed']
                coord_frame = "LLA"
 
                std_x, std_y, std_z = 0.0, 0.0, 0.0
                std_vx, std_vy, std_vz = 0.0, 0.0, 0.
                std_yx = 0.0
                std_vyx = 0.0

            intruder = {"la_no": la_no,
                "lo_ea": lo_ea,
                "a": intruder_state['abs_cartesian_x_y_z_smoothed'][2],
                "t": trk,
                "gs": np.sqrt(
                    intruder_state['velocity_x_y_z_smoothed'][0] ** 2
                    + intruder_state['velocity_x_y_z_smoothed'][1] ** 2
                    ),
                "vs": intruder_state['velocity_x_y_z_smoothed'][2],
                "c": "INTRUDER",
                "i": intr_id,
                "ty": "DAA",
                "ti": intruder_state['time'],
                "cf": coord_frame,
                'd_x': std_x,
                'd_y': std_y,
                'd_z': std_z,
                'd_vx': std_vx,
                'd_vy': std_vy,
                'd_vz': std_vz,
                'd_yx': std_yx,
                'd_vyx': std_vyx
                #"range_det": intruder_state['distance_variance']
            }
            
            if intruder_state['track_source'] == "EO":
                isotropic_decomposition = True
            else:
                isotropic_decomposition = False

            if not isotropic_decomposition:
                intruder_states_input.append(intruder)
            else:
                ### Define Constants ###
                ISO_RELAX_ITER_THRESH = 10 # Time of no guidance to trigger isotropic docmposition relaxation
                ISO_RELAX_FACTOR = 1.01 # Growth rate of desired span during relaxation transition
                SPAN_THRESH_DEG = 1 # (Unrelaxed) Desired max angle of isotropic decomposition position ellipse approximation
                RATIO_THRESH = 0.8 # (Unrelaxed) Min ratio of bounding sphere span to min span of actual ellipse to default to spherical approximation
                VEL_RELAX_ITER_THRESH = 10 # Duration of sustained max relaxation before velocity also relaxed to bounding sphere

                # =========================
                # Generate Isotropic Component Cover
                # =========================
                def ellipsoid_params_from_cov(cov, n_std=2.0):
                    """
                    Return semi-axes (largest first) and rotation matrix R (columns = eigenvectors).
                    Supports 2x2 or 3x3 covariance matrices.
                    """
                    cov = np.asarray(cov)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    order = eigvals.argsort()[::-1]
                    lam = np.maximum(eigvals[order], 0.0)
                    R = eigvecs[:, order]
                    axes = n_std * np.sqrt(lam)
                    return axes, R  # axes length = dim, R shape = (dim, dim)

                def centers_along_major(a, n, method="cheb"):
                    """
                    Interior centers xs ∈ (-a, a). 'cheb' places more points near the ends (smoother radii).
                    """
                    if n <= 1:
                        return np.array([0.0], dtype=float)
                    if method == "cheb":
                        i = np.arange(1, n + 1)
                        xs = a * np.cos(np.pi * i / (n + 1))  # Chebyshev interior nodes
                        xs.sort()
                    else:
                        xs = np.linspace(-a, a, n + 2)[1:-1]
                    return xs.astype(float)

                def cross_section_radius(a, m, s):
                    """Max cross-section radius at abscissa s along major axis (2D: m=b; 3D: m=max(b,c))."""
                    t = 1.0 - (s / a) ** 2
                    return m * np.sqrt(np.clip(t, 0.0, 1.0))

                def switch_abscissa_from_ball(r, x, a, m, direction):
                    """
                    Solve for s in: r^2 = (s - x)^2 + m^2 * (1 - (s/a)^2).
                    Closed-form quadratic; choose root on requested side of x.
                    """
                    eps = 1e-12
                    k = 1.0 - (m * m) / (a * a)  # 1 - (m/a)^2
                    if abs(k) < 1e-12:
                        # Nearly spherical cross-section
                        if abs(x) < eps:
                            s = 0.0
                        else:
                            s = (a * a + x * x - r * r) / (2.0 * x)
                    else:
                        disc = (m * m / (a * a)) * x * x + k * (r * r - m * m)
                        if disc < 0.0:
                            disc = 0.0
                        sqrt_disc = np.sqrt(disc)
                        s = (x + sqrt_disc) / k if direction == "right" else (x - sqrt_disc) / k

                    # Keep s on the correct side of x and within [-a, a]
                    if direction == "right" and s <= x:
                        s = x + 1e-9
                    if direction == "left" and s >= x:
                        s = x - 1e-9
                    return float(np.clip(s, -a, a))

                def propagate_from_left(xs, a, m):
                    """Left→right propagation with exact left end-cap tangency."""
                    n = len(xs)
                    r = np.zeros(n, dtype=float)
                    r[0] = a + xs[0]  # tangent at x = -a
                    for i in range(n - 1):
                        s = switch_abscissa_from_ball(r[i], xs[i], a, m, direction="right")
                        y = cross_section_radius(a, m, s)
                        r[i + 1] = np.sqrt((s - xs[i + 1]) ** 2 + y * y)
                    return r

                def propagate_from_right(xs, a, m):
                    """Right→left propagation with exact right end-cap tangency."""
                    n = len(xs)
                    r = np.zeros(n, dtype=float)
                    r[-1] = a - xs[-1]  # tangent at x = +a
                    for i in range(n - 1, 0, -1):
                        s = switch_abscissa_from_ball(r[i], xs[i], a, m, direction="left")
                        y = cross_section_radius(a, m, s)
                        r[i - 1] = np.sqrt((s - xs[i - 1]) ** 2 + y * y)
                    return r

                def enforce_monotone_growth(xs, r):
                    """
                    Minimal isotonic pass: non-decreasing radii toward center from both ends.
                    Keeps exact end tangency.
                    """
                    r = r.copy()
                    k = int(np.argmin(np.abs(xs)))  # index nearest to x=0

                    # Left → center
                    for i in range(0, k):
                        if r[i + 1] < r[i]:
                            r[i + 1] = r[i]

                    # Right → center
                    for i in range(len(xs) - 1, k, -1):
                        if r[i - 1] < r[i]:
                            r[i - 1] = r[i]

                    return r

                def prune_redundant_endcaps(xs, radii, eps=1e-9):
                    """
                    Iteratively prune from both ends:
                    keep removing the current leftmost/rightmost ball if it is fully
                    contained by *any* other ball toward the interior. Stops when
                    the ends are no longer contained, or only one ball remains.

                    Returns: xs_out, radii_out, keep_left, keep_right
                    keep_left/right indicate whether the *original* extremes survived.
                    """
                    xs = np.asarray(xs, dtype=float)
                    radii = np.asarray(radii, dtype=float)
                    n = len(xs)
                    if n <= 1:
                        return xs, radii, True, True

                    left, right = 0, n - 1

                    # Peel from the left while contained by any interior ball
                    while left < right:
                        i = left
                        # any j to the right that fully contains i?
                        contained = np.any(
                            radii[left+1:right+1] >= (np.abs(xs[left+1:right+1] - xs[i]) + radii[i] - eps)
                        )
                        if contained:
                            left += 1
                        else:
                            break

                    # Peel from the right while contained by any interior ball
                    while right > left:
                        i = right
                        # any j to the left that fully contains i?
                        contained = np.any(
                            radii[left:right] >= (np.abs(xs[i] - xs[left:right]) + radii[i] - eps)
                        )
                        if contained:
                            right -= 1
                        else:
                            break

                    xs_out = xs[left:right+1]
                    radii_out = radii[left:right+1]
                    keep_left = (left == 0)
                    keep_right = (right == n - 1)

                    # Ensure at least one remains (defensive; should already hold)
                    if xs_out.size == 0:
                        xs_out = xs[[0]]
                        radii_out = radii[[0]]
                        keep_left, keep_right = True, (n == 1)

                    return xs_out, radii_out, keep_left, keep_right

                def constituents_from_cov(cov, n_components, n_std=2.0,
                                        center_method="cheb", enforce_monotone=True,
                                        circular_threshold=0.90):
                    """
                    Unified O(n) construction for 2D (ellipse) or 3D (ellipsoid):
                    - If near-circular/spherical (ratio >= circular_threshold): return ONE isotropic covariance
                        centered at the mean, radius = a (semi-major), enclosing the n_std iso-contour.
                    - Else: centers along the major axis, end balls tangent at ±a, analytic neighbor switches,
                        optional monotone growth toward center, and end-cap pruning.

                    Returns:
                    centers_world, covs_iso, (radii, (axes..., R), centers_eig)
                        * For 3D: axes... = (a, b, c, R), centers_eig shape (N,3)
                    """
                    cov = np.asarray(cov)

                    axes, R = ellipsoid_params_from_cov(cov, n_std=n_std)  # sorted desc
                    a = axes[0]

                    b, c = axes[1], axes[2]
                    # Near-spherical shortcut
                    if (min(b, c) / a) >= circular_threshold:
                        r = a  # sphere that encloses the 2σ ellipsoid
                        centers_eig = np.zeros((1, 3), dtype=float)
                        centers_world = centers_eig @ R.T
                        cov_iso = ((r / n_std) ** 2) * np.eye(3)
                        radii = np.array([r], dtype=float)
                        return centers_world, [cov_iso], radii, ((a, b, c, R), centers_eig)

                    # Standard fast construction (use m = max(b, c) to cover full y–z cross-section)
                    m = max(b, c)
                    xs = centers_along_major(a, n_components, method=center_method)
                    rL = propagate_from_left(xs, a, m)
                    rR = propagate_from_right(xs, a, m)
                    radii = np.maximum(rL, rR)

                    # Exact end-cap tangency
                    radii[0]  = a + xs[0]
                    radii[-1] = a - xs[-1]

                    if enforce_monotone:
                        radii = enforce_monotone_growth(xs, radii)

                    # Prune redundant end-caps
                    xs, radii, _, _ = prune_redundant_endcaps(xs, radii)

                    # Back to world coords; isotropic constituents (σ = r / n_std)
                    centers_eig = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
                    centers_world = (R @ centers_eig.T).T
                    covs_iso = [((ri / n_std) ** 2) * np.eye(3) for ri in radii]
                    return centers_world, covs_iso, radii, ((a, b, c, R), centers_eig)

                def _reconstruct_cov_from_stds(dx, dy, dyx, dz):
                    """
                    Build a 3x3 covariance for (x, y, z) from per-axis stds and a signed cross-std 'dyx'.
                    Convention: cov_xy = sign(dyx) * (|dyx|^2). No x–z / y–z cross terms are available -> set to 0.

                    Expected intruder fields:
                    dx=intruder['d_x'], dy=intruder['d_y'], dz=intruder['d_z'], dyx=intruder['d_yx']
                    And similarly for velocity with d_vx, d_vy, d_vz, d_vyx.
                    """
                    dx = float(dx); dy = float(dy); dz = float(dz); dyx = float(dyx)
                    cov = np.zeros((3, 3), dtype=float)
                    # variances
                    cov[0, 0] = dx * dx
                    cov[1, 1] = dy * dy
                    cov[2, 2] = dz * dz
                    # xy cross term from signed cross-std
                    cov_xy = np.sign(dyx) * (abs(dyx) ** 2)
                    cov[0, 1] = cov[1, 0] = cov_xy
                    # xz, yz left at 0 (no fields provided)
                    return cov
                
                def build_intruder_constituents(
                    intruder,
                    n_pos_max=11, n_vel_max=5, n_std=2.0,
                    center_method="cheb", enforce_monotone=True,
                    circular_threshold=0.90,
                    p_world=None, c0_world=None,
                ):
                    """
                    Position constituents chosen adaptively by angular spans:

                    1) Bounding-sphere span (full) = 2*asin(R/L) in deg.
                        - If <= 5°, use ONE bounding sphere (radius = major semi-axis @ n_std).
                    2) Else compute ellipsoid min span (full) via quartic.
                        - If ellipsoid_span >= 0.8 * bounding_span, use bounding sphere.
                    3) Else increase #position constituents from 2..n_pos_max until the
                        composite constituent span (outermost tangent angles across all
                        isotropic spheres) <= 5°, or n_pos_max is reached.

                    Prints (when p_world & c0_world provided):
                    [Spans] intruder <id>: sphere=XX.XXX deg, ellipsoid_min=YY.YYY deg, constituents=ZZ.ZZZ deg (n_pos=N)

                    Velocity constituents are always built with up to n_vel_max.
                    """
                    # --- Rebuild 3x3 covariances for position and velocity from stored stds/cross-terms
                    Ppos3 = _reconstruct_cov_from_stds(intruder['d_x'],  intruder['d_y'],  intruder['d_yx'],  intruder['d_z'])
                    Pvel3 = _reconstruct_cov_from_stds(intruder['d_vx'], intruder['d_vy'], intruder['d_vyx'], intruder['d_vz'])

                    # ---------------- Helpers ----------------
                    def _inv_posdef(C):
                        C = np.asarray(C, dtype=float)
                        try:
                            return np.linalg.inv(C)
                        except np.linalg.LinAlgError:
                            eps = 1e-6 * (np.trace(C)/3.0 + 1.0)
                            try:
                                return np.linalg.inv(C + eps*np.eye(3))
                            except np.linalg.LinAlgError:
                                return np.linalg.pinv(C + eps*np.eye(3))

                    def _ellipsoid_min_halfangle(Ppos3, p_rel):
                        """Return alpha_min (radians) or None if no valid tangents."""
                        A = _inv_posdef(Ppos3) / (n_std**2)
                        Ap = A @ p_rel
                        M = np.outer(Ap, Ap) - (p_rel @ Ap - 1.0) * A

                        L0 = np.linalg.norm(p_rel)
                        if not np.isfinite(L0) or L0 <= 1e-9:
                            return None
                        c_hat = -p_rel / L0

                        tmp = np.array([1.0, 0.0, 0.0])
                        if abs(np.dot(tmp, c_hat)) > 0.9:
                            tmp = np.array([0.0, 1.0, 0.0])
                        u = tmp - np.dot(tmp, c_hat) * c_hat
                        u /= np.linalg.norm(u)
                        v = np.cross(c_hat, u)

                        m11 = float(u @ (M @ u))
                        m12 = float(u @ (M @ v))
                        m22 = float(v @ (M @ v))
                        g1  = float(u @ (M @ c_hat))
                        g2  = float(v @ (M @ c_hat))
                        mcc = float(c_hat @ (M @ c_hat))

                        N2   = np.poly1d([-2*m12,  2*(m22 - m11), 2*m12])
                        NA2  = np.poly1d([ m22,    2*m12,         m11   ])
                        NA1  = np.poly1d([ g2,     g1                     ])
                        NdA1 = np.poly1d([-g1,     g2                     ])
                        Epoly = mcc*np.polymul(N2, N2) \
                            - 4.0*np.polymul(np.polymul(NA1, NdA1), N2) \
                            + 4.0*np.polymul(NA2, np.polymul(NdA1, NdA1))

                        roots_t = np.roots(Epoly)
                        roots_t = roots_t[np.isreal(roots_t)].real

                        def _small_positive_root(C0, C1, C2):
                            if abs(C2) < 1e-15:
                                if abs(C1) < 1e-15: return None
                                s = -C0/(2*C1)
                                return s if s > 0 else None
                            disc = (2*C1)**2 - 4*C2*C0
                            if disc < 0: return None
                            sq = np.sqrt(disc)
                            s1 = (-2*C1 - sq)/(2*C2); s2 = (-2*C1 + sq)/(2*C2)
                            cands = [s for s in (s1, s2) if s is not None and s > 0]
                            return min(cands) if cands else None

                        alpha_min = None
                        for tval in roots_t:
                            den = 1.0 + tval*tval
                            cos_th = 1.0/np.sqrt(den);  sin_th = tval/np.sqrt(den)
                            wdir = cos_th*u + sin_th*v
                            C1 = float((M @ c_hat) @ wdir)
                            C2 = float(wdir @ (M @ wdir))
                            s = _small_positive_root(mcc, C1, C2)  # s = tan(alpha)
                            if s is None: continue
                            a = float(np.arctan(s))
                            if (alpha_min is None) or (a < alpha_min):
                                alpha_min = a
                        return alpha_min

                    def _constituent_composite_span_deg(pos_centers_w, pos_radii, p_world, c0_world):
                        """
                        Return the maximum full cone span (degrees) across all constituent spheres,
                        measured relative to each sphere's own center line (p_world -> sphere center).

                        For each sphere i with center c_i and radius R_i:
                            L_i = ||p_world - c_i||
                            span_i = 2 * arcsin(R_i / L_i)  [in degrees]
                        The function returns max_i(span_i). Spheres that contain the viewpoint are skipped.
                        """

                        p_world = np.asarray(p_world, dtype=float).reshape(3)
                        c0_world = np.asarray(c0_world, dtype=float).reshape(3)

                        spans = []
                        for ci_rel, Ri in zip(pos_centers_w, pos_radii):
                            c_i = c0_world + np.asarray(ci_rel, dtype=float)
                            Li  = np.linalg.norm(p_world - c_i)
                            Ri  = float(Ri)

                            # Skip if viewpoint is inside/on the sphere (no real tangents)
                            if not np.isfinite(Li) or Li <= Ri + 1e-9:
                                continue

                            alpha = np.arcsin(np.clip(Ri / Li, 0.0, 1.0))   # half-angle (rad)
                            spans.append(2.0 * np.degrees(alpha))           # full span (deg)

                        if not spans:
                            return None
                        return float(max(spans))

                    # ---------------- Decision logic ----------------
                    use_adaptive = (p_world is not None) and (c0_world is not None)
                    pos_centers_w_final = None
                    pos_covs_iso3_final = None
                    pos_radii_final     = None

                    # For printing:
                    span_sph_deg   = None
                    ell_span_deg   = None
                    comp_span_deg  = None
                    n_pos_used     = None

                    if use_adaptive:
                        p_world = np.asarray(p_world, dtype=float).reshape(3)
                        c0_world = np.asarray(c0_world, dtype=float).reshape(3)
                        p_rel = p_world - c0_world
                        L0 = np.linalg.norm(p_rel)

                        # Bounding sphere (radius = a at n_std)
                        axes_pos, _R = ellipsoid_params_from_cov(Ppos3, n_std=n_std)
                        R_enclose = float(axes_pos[0])

                        if not np.isfinite(L0) or L0 <= 1e-9:
                            span_sph_deg = None
                        elif L0 <= R_enclose + 1e-9:
                            span_sph_deg = 180.0  # inside/on sphere: treat as worst-case
                        else:
                            alpha_sph = np.arcsin(np.clip(R_enclose / L0, 0.0, 1.0))
                            span_sph_deg = float(2.0 * np.degrees(alpha_sph))

                        def _choose_bounding_sphere():
                            centers = np.zeros((1, 3), dtype=float)  # centered at ellipsoid mean
                            cov_iso = ((R_enclose / n_std) ** 2) * np.eye(3)
                            radii   = np.array([R_enclose], dtype=float)
                            return centers, [cov_iso], radii

                        if (span_sph_deg is not None) and (span_sph_deg <= SPAN_THRESH_DEG):
                            # Case 1: bounding sphere already tight enough
                            pos_centers_w_final, pos_covs_iso3_final, pos_radii_final = _choose_bounding_sphere()
                            comp_span_deg = None
                            n_pos_used = 1
                        else:
                            # Need the ellipsoid min span (full)
                            alpha_min = _ellipsoid_min_halfangle(Ppos3, p_rel)
                            ell_span_deg = float(2.0 * np.degrees(alpha_min)) if (alpha_min is not None) else None

                            if (ell_span_deg is None) or (span_sph_deg is None):
                                # Fall back to bounding sphere
                                pos_centers_w_final, pos_covs_iso3_final, pos_radii_final = _choose_bounding_sphere()
                                comp_span_deg = None
                                n_pos_used = 1
                            elif ell_span_deg >= RATIO_THRESH * span_sph_deg:
                                # Case 2: ellipsoid span close to bounding sphere -> just use bounding sphere
                                pos_centers_w_final, pos_covs_iso3_final, pos_radii_final = _choose_bounding_sphere()
                                comp_span_deg = None
                                n_pos_used = 1
                            else:
                                # Case 3: build constituents, increasing N until composite desired span or N == n_pos_max

                                # Get isotropic decomposition and daidalus history for this intruder
                                this_iso_decomp_guidance_hist = self.iso_decomp_guidance_hist.setdefault(
                                    intruder['i'],
                                    {
                                        'iters_since_guidance': 0,
                                        'last_relaxed_span': 0.0,
                                        'num_relaxations': 0,
                                        'iters_at_max_relaxation': 0
                                    }
                                )

                                # Ensure that the desired span is never less than the span of the original ellipse
                                adjusted_span_thresh_deg = max(SPAN_THRESH_DEG, ell_span_deg)

                                defaluted_to_sphere = False
                                # print("Time Since Guidance:", this_iso_decomp_guidance_hist['time_since_guidance'])
                                # Apply relaxation to isotropic component construction parameters if threshold reached
                                if this_iso_decomp_guidance_hist['iters_since_guidance'] > ISO_RELAX_ITER_THRESH:
                                    # print('############# RELAXING ##############')
                                    starting_span = max(this_iso_decomp_guidance_hist['last_relaxed_span'], adjusted_span_thresh_deg)
                                    # Special case 1: Previous Relaxed state encompasses current bounding sphere
                                    # -> Already maximally relaxed, flag special case = True
                                    if starting_span >= span_sph_deg:
                                        # print('PREV SPHERE')
                                        defaluted_to_sphere = True
                                    else:
                                        # print('APPLYING RELAXATION FACTOR')
                                        # Apply relaxation for this iteration
                                        this_iso_decomp_guidance_hist['num_relaxations'] += 1
                                        adjusted_span_thresh_deg = min(starting_span * ISO_RELAX_FACTOR ** this_iso_decomp_guidance_hist['num_relaxations'], span_sph_deg)

                                        # Special case 2: Newly Relaxed state encompasses current bounding sphere
                                        # -> Already maximally relaxed, flag special case = True
                                        # print("Relaxation Factor: ", ISO_RELAX_FACTOR ** this_iso_decomp_guidance_hist['num_relaxations'])
                                        # print("Sphere Span: ", span_sph_deg)
                                        # print("Original Span: ", starting_span)
                                        # print("Adjusted Span: ", adjusted_span_thresh_deg)
                                        if adjusted_span_thresh_deg >= span_sph_deg:
                                            # print('CURRENT SPHERE')
                                            defaluted_to_sphere = True

                                    # If special case flagged, return bounding sphere
                                    if defaluted_to_sphere:
                                        this_iso_decomp_guidance_hist['iters_at_max_relaxation'] += 1
                                            
                                        adjusted_span_thresh_deg = span_sph_deg
                                        pos_centers_w_final, pos_covs_iso3_final, pos_radii_final = _choose_bounding_sphere()
                                        comp_span_deg = None
                                        n_pos_used = 1

                                        # Maximise velocity relaxation if position has remained consistently relaxed
                                        if (
                                            this_iso_decomp_guidance_hist['iters_at_max_relaxation']
                                            >= VEL_RELAX_ITER_THRESH
                                        ):
                                            n_vel_max = 1
                                    elif this_iso_decomp_guidance_hist['iters_at_max_relaxation'] != 0:
                                        this_iso_decomp_guidance_hist['iters_at_max_relaxation'] = 0

                                    # Update relaxed span history
                                    this_iso_decomp_guidance_hist['last_relaxed_span'] = adjusted_span_thresh_deg
                                    # print('###########################')
                                else:
                                    # If time since guidance less than threshold, no relaxations
                                    this_iso_decomp_guidance_hist['num_relaxations'] = 0
                                    this_iso_decomp_guidance_hist['iters_at_max_relaxation'] = 0
                                    this_iso_decomp_guidance_hist['last_relaxed_span'] = adjusted_span_thresh_deg

                                # Skip decomposition if defaulted to sphere after relaxation
                                if not defaluted_to_sphere:
                                    chosen = False
                                    last_pc_w = last_pc_covs = last_radii = None
                                    for n_pos in range(2, max(2, n_pos_max) + 1):
                                        pc_w, pc_covs, radii, _ = constituents_from_cov(
                                            Ppos3, n_components=n_pos, n_std=n_std,
                                            center_method=center_method, enforce_monotone=enforce_monotone,
                                            circular_threshold=circular_threshold
                                        )
                                        span_comp_deg = _constituent_composite_span_deg(pc_w, radii, p_world, c0_world)
                                        # keep last to fall back to
                                        last_pc_w, last_pc_covs, last_radii = pc_w, pc_covs, radii
                                        comp_span_deg = span_comp_deg  # for printing (last evaluated)
                                        if (span_comp_deg is not None) and (span_comp_deg <= adjusted_span_thresh_deg):
                                            pos_centers_w_final, pos_covs_iso3_final, pos_radii_final = pc_w, pc_covs, radii
                                            n_pos_used = len(radii)
                                            chosen = True
                                            break
                                    if not chosen:
                                        pos_centers_w_final, pos_covs_iso3_final, pos_radii_final = last_pc_w, last_pc_covs, last_radii
                                        n_pos_used = len(pos_radii_final) if pos_radii_final is not None else None
                    else:
                        # No viewpoint/center provided -> default to n_pos_max constituents
                        pos_centers_w_final, pos_covs_iso3_final, pos_radii_final, _ = constituents_from_cov(
                            Ppos3, n_components=n_pos_max, n_std=n_std,
                            center_method=center_method, enforce_monotone=enforce_monotone,
                            circular_threshold=circular_threshold
                        )

                    # Keep only the position constituent whose surface is closest to ownship.
                    # pos_centers_w_final are offsets from the intruder mean, so compare
                    # ownship against the absolute center: c0_world + constituent_offset.
                    if (p_world is not None) and (c0_world is not None):
                        pos_centers_abs = c0_world.reshape(1, 3) + pos_centers_w_final[:, :3]
                        dists_to_surface = np.linalg.norm(pos_centers_abs - p_world.reshape(1, 3), axis=1) - pos_radii_final
                        closest_idx = int(np.argmin(dists_to_surface))

                        pos_centers_w_final = pos_centers_w_final[closest_idx:closest_idx + 1]
                        pos_covs_iso3_final = pos_covs_iso3_final[closest_idx:closest_idx + 1]
                        pos_radii_final = pos_radii_final[closest_idx:closest_idx + 1]

                    # --------- Print spans (full angles in degrees) ---------
                    # if (p_world is not None) and (c0_world is not None):
                    #     def _fmt(x): 
                    #         return f"{x:.3f} deg" if (x is not None and np.isfinite(x)) else "n/a"
                        # print(f"[Spans] intruder {intruder['i']}: "
                        #     f"sphere={_fmt(span_sph_deg)}, "
                        #     f"ellipsoid_min={_fmt(ell_span_deg)}, "
                        #     f"constituents={_fmt(comp_span_deg)} "
                        #     f"(n_pos={n_pos_used if n_pos_used is not None else 'n/a'})")

                    # =========================
                    # Build constituent intruders (combine chosen position model with velocity constituents)
                    # =========================

                    # --- Velocity constituents (fixed or bounding sphere if triggered)
                    vel_centers_w, vel_covs_iso3, _, _ = constituents_from_cov(
                        Pvel3, n_components=n_vel_max, n_std=n_std,
                        center_method=center_method, enforce_monotone=enforce_monotone,
                        circular_threshold=circular_threshold
                    )

                    base_vx = float(intruder['gs'] * np.cos(np.radians(intruder['t'])))
                    base_vy = float(intruder['gs'] * np.sin(np.radians(intruder['t'])))
                    base_vz = float(intruder['vs'])

                    out = []
                    uid_prefix = f"{intruder['i']}_const_"

                    for ip, (pc, cov_pos3) in enumerate(zip(pos_centers_w_final, pos_covs_iso3_final)):
                        px_off, py_off, pz_off = float(pc[0]), float(pc[1]), float(pc[2])

                        for iv, (vc, cov_vel3) in enumerate(zip(vel_centers_w, vel_covs_iso3)):
                            vx_off, vy_off, vz_off = float(vc[0]), float(vc[1]), float(vc[2])

                            intr = copy.deepcopy(intruder)
                            intr['i'] = f"{uid_prefix}{ip}_{iv}"

                            # Means with offsets
                            intr["la_no"] = float(intr["la_no"]) + px_off
                            intr["lo_ea"] = float(intr["lo_ea"]) + py_off
                            intr["a"]     = float(intr["a"])     + pz_off

                            vx = base_vx + vx_off
                            vy = base_vy + vy_off
                            vz = base_vz + vz_off

                            intr["t"]  = (np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0
                            intr["gs"] = float(np.hypot(vx, vy))
                            intr["vs"] = float(vz)

                            # 6x6 covariance: pos/vel blocks
                            cov = np.zeros((6, 6), dtype=float)
                            cov[0:3, 0:3] = cov_pos3
                            cov[3:6, 3:6] = cov_vel3

                            # Export per-axis stds (isotropic constituents => no cross-terms)
                            intr['d_x']   = float(np.sqrt(max(cov[0, 0], 0.0)))
                            intr['d_y']   = float(np.sqrt(max(cov[1, 1], 0.0)))
                            intr['d_z']   = float(np.sqrt(max(cov[2, 2], 0.0)))
                            intr['d_yx']  = 0.0
                            intr['d_vx']  = float(np.sqrt(max(cov[3, 3], 0.0)))
                            intr['d_vy']  = float(np.sqrt(max(cov[4, 4], 0.0)))
                            intr['d_vz']  = float(np.sqrt(max(cov[5, 5], 0.0)))
                            intr['d_vyx'] = 0.0

                            out.append(intr)

                    if self.plot:
                        self._update_constituent_layers(
                            intruder_id=intruder['i'],
                            intr_cf=intruder['cf'],
                            intr_la_no=intruder['la_no'],
                            intr_lo_ea=intruder['lo_ea'],
                            pos_centers_w=pos_centers_w_final,
                            pos_covs_iso3=pos_covs_iso3_final,
                            # For the velocity panel, pass centers already in (vx,vy) *absolute* space.
                            # If your velocity constituent centers are offsets, pre-sum with base (vx,vy) before calling.
                            vel_centers_w=[np.array([base_vx + vc[0], base_vy + vc[1], 0.0]) for vc in vel_centers_w],
                            vel_covs_iso3=vel_covs_iso3,
                            pos_cov3=Ppos3,
                            vel_cov3=Pvel3,
                            stds_plot=getattr(self, 'stds_plot', 2),
                        )

                    return out


                constituent_build_start_s = time.perf_counter()
                constituent_intruders = build_intruder_constituents(
                    intruder, n_pos_max=5, n_vel_max=3,
                    p_world=np.array([
                        ownship_state['north_m'], ownship_state['east_m'], ownship_state['alt']
                    ]),
                    c0_world=np.array(intruder_state['abs_cartesian_x_y_z_smoothed'])
                )
                constituent_build_dt_s = time.perf_counter() - constituent_build_start_s
                self.daidalus_constituent_build_count += 1
                self.daidalus_constituent_build_total_s += constituent_build_dt_s
                self.daidalus_constituent_build_max_s = max(self.daidalus_constituent_build_max_s, constituent_build_dt_s)

                last_ms = 1000.0 * constituent_build_dt_s
                avg_ms = 1000.0 * self.daidalus_constituent_build_total_s / self.daidalus_constituent_build_count
                max_ms = 1000.0 * self.daidalus_constituent_build_max_s
                self.append_daidalus_timing(
                    ownship_state['time'], len(constituent_intruders), last_ms, avg_ms, max_ms
                )

                if self.daidalus_constituent_build_count == 1 or self.daidalus_constituent_build_count % 25 == 0:
                    print(
                        '[DAIDALUS constituent timing] '
                        f'n={self.daidalus_constituent_build_count} '
                        f'last={last_ms:.3f} ms '
                        f'avg={avg_ms:.3f} ms '
                        f'max={max_ms:.3f} ms '
                        f'generated={len(constituent_intruders)}'
                    )

                intruder_states_input.extend(constituent_intruders)

        final_data = {"o": ownship_state_input,
                        "d": intruder_states_input}
        
        return final_data

    def intruder_rel_x_y_z(self, ownship_packet, intruder, heading_ref_deg = 0):
        # Ownship absolute position in NED (North, East, Up)
        ref_n = ownship_packet["north_m"]
        ref_e = ownship_packet["east_m"]
        ref_u = ownship_packet["alt"]
     
        # Intruder absolute position in NED
        i_n = intruder['abs_cartesian_x_y_z_smoothed'][0]
        i_e = intruder['abs_cartesian_x_y_z_smoothed'][1]
        i_u = intruder['abs_cartesian_x_y_z_smoothed'][2]
     
        # Relative position (from ownship to intruder) in NED frame
        delta_n = i_n - ref_n
        delta_e = i_e - ref_e
        delta_u = i_u - ref_u  # Up component
     
        # Ownship heading (track angle) in degrees
        heading_deg = heading_ref_deg
        heading_rad = np.radians(heading_deg)
     
        # Rotate NED delta into ownship body frame (X = forward/North, Y = right/East, Z = down)
        x = delta_n * np.cos(heading_rad) + delta_e * np.sin(heading_rad)
        y = -delta_n * np.sin(heading_rad) + delta_e * np.cos(heading_rad)
        z = delta_u
     
        # Intruder track angle relative to ownship heading
        i_trk = intruder['trk_smoothed']
        rel_trk = (i_trk - heading_deg + 360) % 360
     
        return {
            "x": x,    # Forward (meters)
            "y": y,    # Right (meters)
            "z": z,    # Down (meters)
            "t": rel_trk  # Relative track angle in degrees (body frame, 0° = forward, positive clockwise)
        }

    # def create_range_kalman_filter(self, initial_range, dt, initial_timestamp = 0):
    #     kf = KalmanFilter(dim_x=2, dim_z=1)

    #     # State: [range, range_rate]
    #     kf.F = np.array([
    #         [1, dt],
    #         [0, 1]
    #     ])
    #     kf.H = np.array([[1, 0]])

    #     kf.P *= 800
    #     kf.R *= 3000   # Measurement noise (adjustable)
    #     kf.Q *= 2 * 2 * 0.0001 * 0.05**2  # Process noise (adjustable)

    #     kf.x[0, 0] = initial_range
    #     kf.x[1, 0] = 0  # Assume zero initial rate of change

    #     kf.prev_timestamp = initial_timestamp

    #     return kf
    
    ####################################################
    ##### Constant Velocity Kalman Filter Tracker ######
    ####################################################

    ##### Helpers #####

    ## Update Process Noise and Transition Model Matrices Each Iteration

    def update_F_Q_matrices(self, kf_xyz, dt, coasting, fused_time, source: str = "EO"):
        # Update kf dt
        kf_xyz.dt = dt

        if source == "EO":
            # ===== Existing 6x6 CV model =====
            # Update transition matrix with new dt
            kf_xyz.F = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])

            # Process-noise timing & magnitudes
            if coasting:
                dt_Q = fused_time - kf_xyz.last_measured_time
                accel_std_vel = 0.5
                accel_std_pos = 0.5
            else:
                dt_Q = dt
                if fused_time - kf_xyz.last_measured_time < 10:
                    accel_std_vel = 10.0
                    accel_std_pos = 10.0
                else:
                    accel_std_vel = 6.0
                    accel_std_pos = 6.0

            dt2 = dt_Q**2
            dt3 = dt_Q**3
            dt4 = dt_Q**4
            q_vel = accel_std_vel**2
            q_pos = accel_std_pos**2

            # 6x6 Q (CV with white-acceleration)
            kf_xyz.Q = np.array([
                [q_pos * dt4/4, 0, 0, q_pos * dt3/2, 0, 0],
                [0, q_pos * dt4/4, 0, 0, q_pos * dt3/2, 0],
                [0, 0, q_pos * dt4/4, 0, 0, q_pos * dt3/2],
                [q_pos * dt3/2, 0, 0, q_vel * dt2, 0, 0],
                [0, q_pos * dt3/2, 0, 0, q_vel * dt2, 0],
                [0, 0, q_pos * dt3/2, 0, 0, q_vel * dt2]
            ])

        else:
            # ===== CA model (9x9) for non-EO sources =====
            # State: [x y z vx vy vz ax ay az]^T
            I3 = np.eye(3)
            Z3 = np.zeros((3, 3))

            # Transition (constant acceleration)
            # x' = x + v*dt + 0.5*a*dt^2
            # v' = v + a*dt
            # a' = a
            kf_xyz.F = np.block([
                [I3,        dt*I3,   0.5*(dt**2)*I3],
                [Z3,           I3,        dt*I3    ],
                [Z3,           Z3,          I3      ],
            ])

            # Process-noise timing & magnitudes (mirror your logic)
            if coasting:
                dt_Q = fused_time - kf_xyz.last_measured_time
                jerk_std = 2.0   # smoother when coasting
            else:
                dt_Q = dt
                # TODO: Tune these if required
                if fused_time - kf_xyz.last_measured_time < 10:
                    jerk_std = 6.0  # initial responsiveness
                else:
                    jerk_std = 6.0  # estabilshed responsiveness

            # White-jerk CA Q (block 3x3 structure)
            q = jerk_std**2
            dt2 = dt_Q**2
            dt3 = dt_Q**3
            dt4 = dt_Q**4
            dt5 = dt_Q**5

            Q11 = (dt5 / 20.0) * I3
            Q12 = (dt4 / 8.0)  * I3
            Q13 = (dt3 / 6.0)  * I3
            Q22 = (dt3 / 3.0)  * I3
            Q23 = (dt2 / 2.0)  * I3
            Q33 = (dt_Q      ) * I3

            kf_xyz.Q = q * np.block([
                [Q11, Q12, Q13],
                [Q12, Q22, Q23],
                [Q13, Q23, Q33],
            ])

        return

    # Update kalman filter history with new state

    # def update_kf_state_history(self, kf, timestamp):
    #     kf.history_state[timestamp] = copy.ceepcopy(kf)

    #     # Efficiently prune old entries from measurement history
    #     cutoff = timestamp - kf.history_lag_seconds
    #     keys_to_remove = []
    #     for t in kf.history_state:
    #         if t < cutoff:
    #             keys_to_remove.append(t)
    #         else:
    #             break  # OrderedDict ensures timestamps are ascending
    #     for t in keys_to_remove:
    #         kf.history_state.pop(t)

    #     return

    ### Propagate initialisation history thorugh kalman filter

    # TODO: Track association instead of fusing all tracks into a single intruder always
    def propagate_init_history(self, init_history, track_id, track_source):
        if not init_history:
            return

        # print("Propagating Initialiasation Detection History")      

        sorted_init_times = sorted(init_history.keys())

        # ##### Extract lagged tracker state prior to init history start #####
        # if track_id in self.xyz_kalman_filters:
        #     kf = self.xyz_kalman_filters[track_id]
            
        #     # 1. Find latest rewind time and gather subsequent measurements
        #     init_start_time = sorted_init_times[0]
        #     rewind_time = None
        #     measurements_after_rewind = {}

        #     for t in kf.history_measurements:
        #         if t < init_start_time:
        #             rewind_time = t
        #         else:
        #             measurements_after_rewind[t] = kf.history_measurements[t]
        
        #     # 2. Build full time-sorted measurement list from history and init_history
        #     full_measurement_list = []

        #     for t, (mean, cov) in measurements_after_rewind.items():
        #         full_measurement_list.append((t, 'history'))

        #     for det in init_history:
        #         full_measurement_list.append((det.time, 'init'))

        #     # 3. Sort by timestamp
        #     full_measurement_list.sort(key=lambda x: x[0])

        #     # 4. Check if each init_history time is close to any history measurement time
        #     init_times = [det.time for det in init_history]
        #     history_times = list(measurements_after_rewind.keys())

        #     time_matches = []
        #     epsilon = 1e-3  # Tolerance for time match

        #     for t_init in init_times:
        #         for t_hist in history_times:
        #             if abs(t_init - t_hist) < epsilon:
        #                 time_matches.append((t_init, t_hist))
        #                 break  # Only care about first match for now

        #     # 6. Build a dict of matched measurements from history
        #     matched_measurements_from_history = {
        #         t_init: kf.history_measurements[t_hist]
        #         for t_init, t_hist in time_matches
        #     }

        #     print(matched_measurements_from_history)

        for init_time_this in sorted_init_times: 
            detections_this_time = init_history[init_time_this]
            if len(detections_this_time) == 1:
                item = detections_this_time[0]
                cam_id = item.cam_id
                raw_range = item.distance
                range_bias, range_var, pixel_var = self.ownship.cameras[cam_id]['camera'].get_detected_uncertainty_params(raw_range)
                pixel_std = np.sqrt(pixel_var)
                range_std = np.sqrt(range_var)

                mean_global_adjusted, cov_global = self.calculate_xyz_mean_and_covariance(
                    pixel_std, pixel_std, range_std, range_bias,
                    item.abs_cx, item.abs_cy, raw_range, cam_id, init_time_this
                )

                # If all uncertainties are zero, fallback to small epsilon covariance
                if np.allclose(cov_global, 0):
                    cov_global = np.eye(3) * 1e-6

                fused_mean = mean_global_adjusted
                fused_cov = cov_global
                cam_ids = [cam_id]
            else:
                means = []
                covs = []
                cam_ids = []
                timestamps = []
                # detection_confidences = []
                # horizon_statuses = []

                for item in detections_this_time:
                    cam_id = item.cam_id
                    raw_range = item.distance
                    range_bias, range_var, pixel_var = self.ownship.cameras[cam_id]['camera'].get_detected_uncertainty_params(raw_range)
                    pixel_std = np.sqrt(pixel_var)
                    range_std = np.sqrt(range_var)

                    mean_global_adjusted, cov_global = self.calculate_xyz_mean_and_covariance(
                        pixel_std, pixel_std, range_std, range_bias,
                        item.abs_cx, item.abs_cy, raw_range, cam_id, item.time
                    )

                    # If all uncertainties are zero, fallback to small epsilon covariance
                    if np.allclose(cov_global, 0):
                        cov_global = np.eye(3) * 1e-6

                    means.append(mean_global_adjusted)
                    covs.append(cov_global)
                    cam_ids.append(cam_id)
                    timestamps.append(item.time)
                    # detection_confidences.append(item.conf)
                    # horizon_statuses.append(item.above_horizon)

                # === Gaussian Fusion ===
                precisions = []
                for C in covs:
                    if np.allclose(C, 0):
                        C = np.eye(3) * 1e-6  # fallback for perfect measurements
                    precisions.append(inv(C))
                precision_sum = sum(precisions)

                fused_cov = inv(precision_sum)
                fused_mean = fused_cov @ sum(P @ m for P, m in zip(precisions, means))

            self.iterate_tracker_state(track_id, track_source, False, init_time_this, cam_ids, fused_mean, fused_cov) 
                
        return

    ## Update filter with new measurement, update state history
    ## and retroactively pass through initialisation history if available

    # def update_kf_with_measurement(self, kf, timestamp, measurement_mean, measurement_cov, init_history = None):
    #     kf.R = measurement_cov
    #     kf.update(measurement_mean)

    #     # Store measurement in history
    #     if timestamp in kf.history_measurements:
    #         kf.history_measurements[timestamp] = (measurement_mean.copy(), measurement_cov.copy())

    #     # Efficiently prune old entries from measurement history
    #     cutoff = timestamp - kf.history_lag_seconds
    #     keys_to_remove = []
    #     for t in kf.history_measurements:
    #         if t < cutoff:
    #             keys_to_remove.append(t)
    #         else:
    #             break  # OrderedDict ensures timestamps are ascending
    #     for t in keys_to_remove:
    #         kf.history_measurements.pop(t)

    #     kf.prev_timestamp = timestamp

    
    def create_kalman_filter(
        self,
        source: str,
        initial,               # EO: [x,y,z]; non-EO: [x,y,z,vx,vy,vz]
        measurement_cov,       # EO: 3x3 for [x,y,z]; non-EO: 6x6 for [x,y,z,vx,vy,vz]
        dt: float,
        initial_timestamp,
        history_lag_seconds: float = 3.0,
    ):
        """
        Create a Kalman filter depending on `source`.

        - source == "EO": 6-state Constant-Velocity (CV) model with 3-D measurements.
            state: [x, y, z, vx, vy, vz]
            meas : [x, y, z]
            `initial` must be length-3 [x,y,z]; `measurement_cov` must be 3x3.

        - source != "EO": 9-state Constant-Acceleration (CA) model with 6-D measurements.
            state: [x, y, z, vx, vy, vz, ax, ay, az]
            meas : [x, y, z, vx, vy, vz]
            `initial` must be length-6 [x,y,z,vx,vy,vz]; `measurement_cov` must be 6x6.
        """

        if source == "EO":
            # ---- Validate inputs ----
            initial_xyz = np.asarray(initial, dtype=float).reshape(-1)
            if initial_xyz.shape[0] != 3:
                raise ValueError("For source=='EO', `initial` must be length-3 [x,y,z].")
            measurement_cov = np.asarray(measurement_cov, dtype=float)
            if measurement_cov.shape != (3, 3):
                raise ValueError("For source=='EO', `measurement_cov` must be 3x3.")

            # ---- 6x3 CV filter ----
            kf = KalmanFilter(dim_x=6, dim_z=3)  # [x,y,z,vx,vy,vz]

            # F (CV)
            kf.F = np.array([
                [1, 0, 0, dt, 0,  0],
                [0, 1, 0, 0,  dt, 0],
                [0, 0, 1, 0,  0,  dt],
                [0, 0, 0, 1,  0,  0],
                [0, 0, 0, 0,  1,  0],
                [0, 0, 0, 0,  0,  1],
            ])

            # H: measure x,y,z
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ])

            # Initial state
            kf.x = np.zeros((6, 1))
            kf.x[:3, 0] = initial_xyz

            ########### Conservatively set initial velocity pointing toward ownship ###########
            # === Compute range direction from ownship to intruder ===
            x_own, y_own, z_own = self.ownship.position(ownship_time=initial_timestamp).flatten()
            x_intr, y_intr, z_intr = initial_xyz
            range_vec = np.array([x_own - x_intr, y_own - y_intr, z_own - z_intr], dtype=float)
            range_norm = np.linalg.norm(range_vec)
            range_unit = range_vec / max(range_norm, 1e-12)
            init_speed = 10.0
            kf.x[3:, 0] = range_unit * init_speed

            # Covariances
            kf.P = np.eye(6)
            kf.P[:3, :3] = measurement_cov
            #kf.P[3:, 3:] = 0.8 *  measurement_cov / dt ** 2
            kf.P[3:, 3:] = np.eye(3) * 200.0  # high uncertainty on velocity

            # Q (white-accel driving CV)
            accel_std = 4.0 # Tune this value based on expected dynamics
            q = accel_std**2
            dt2, dt3, dt4 = dt**2, dt**3, dt**4
            kf.Q = q * np.array([
                [dt4/4,   0,      0,      dt3/2, 0,     0],
                [0,       dt4/4,  0,      0,     dt3/2, 0],
                [0,       0,      dt4/4,  0,     0,     dt3/2],
                [dt3/2,   0,      0,      dt2,   0,     0],
                [0,       dt3/2,  0,      0,     dt2,   0],
                [0,       0,      dt3/2,  0,     0,     dt2],
            ])

            # R
            kf.R = copy.deepcopy(measurement_cov)

            print(f"Created New 3D-Input Constant-Velocity {source} Kalman Filter")

        else:
            # ---- Validate inputs ----
            init6 = np.asarray(initial, dtype=float).reshape(-1)
            if init6.shape[0] != 6:
                raise ValueError("For source!='EO', `initial` must be length-6 [x,y,z,vx,vy,vz].")
            measurement_cov = np.asarray(measurement_cov, dtype=float)
            if measurement_cov.shape != (6, 6):
                raise ValueError("For source!='EO', `measurement_cov` must be 6x6.")

            # ---- 9x6 CA filter ----
            kf = KalmanFilter(dim_x=9, dim_z=6)

            I3 = np.eye(3)
            Z3 = np.zeros((3, 3))

            # F (CA)
            kf.F = np.block([
                [I3,        dt*I3,   0.5*(dt**2)*I3],
                [Z3,           I3,        dt*I3    ],
                [Z3,           Z3,          I3      ],
            ])

            # H: measure [x,y,z,vx,vy,vz]
            kf.H = np.block([
                [I3, Z3, Z3],
                [Z3, I3, Z3],
            ])  # 6x9

            # Initial state (acc = 0)
            kf.x = np.vstack([init6.reshape(6, 1), np.zeros((3, 1))])

            # P: seed pos/vel from measurement_cov; accel prior variance (σ_a=2 m/s² → var=4)
            kf.P = np.zeros((9, 9))
            kf.P[:6, :6] = copy.deepcopy(measurement_cov)
            kf.P[6:, 6:] = np.eye(3) * 4.0 # 2 m.s^2 accel std (Tunable)

            # Q: white-jerk driving CA
            jerk_std = 6.0  # m/s^3 (Tunable)
            q = jerk_std**2
            dt2, dt3, dt4, dt5 = dt**2, dt**3, dt**4, dt**5

            Q11 = (dt5 / 20.0) * I3
            Q12 = (dt4 / 8.0)  * I3
            Q13 = (dt3 / 6.0)  * I3
            Q22 = (dt3 / 3.0)  * I3
            Q23 = (dt2 / 2.0)  * I3
            Q33 = (dt        ) * I3

            kf.Q = q * np.block([
                [Q11, Q12, Q13],
                [Q12, Q22, Q23],
                [Q13, Q23, Q33],
            ])

            # R
            kf.R = copy.deepcopy(measurement_cov)

            print(f"Created New 6D-Input Constant-Acceleration {source} Kalman Filter")

        # --- Common timing & history ---
        kf.dt = dt
        kf.track_init_timestamp = initial_timestamp
        kf.prev_timestamp = initial_timestamp
        kf.last_measured_time = initial_timestamp
        kf.history_lag_seconds = history_lag_seconds
        kf.history_measurements = {}
        kf.history_state = {}
        kf.state_at_last_measured_time = None
        kf.track_history_at_last_measured_time = {}

        

        return kf



