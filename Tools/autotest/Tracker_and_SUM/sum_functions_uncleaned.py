from core.dataset import Dataset
from core.flight_track_predictions import FlightTrackPredictions
from core.flight_detection_predictions import FlightDetectionPredictions
from core.flight_transform_predictions import FlightTransformPredictions
import cv2
import os
import json
from scipy.stats import norm, t as t_dist
from copy import deepcopy
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import dill as pickle
from scipy.linalg import toeplitz
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Because Life, Universe and Everything!
# random.seed(42)

AMAZON_CAMERA_RES = [2448, 2048]
AMAZON_CAMERA_FOV = [67.32, 56.52]

class FakeBoundingBox:
    def __init__(self, bbox):
        self._bbox = bbox  # (x_min, y_min, x_max, y_max)

    def get_center(self):
        x_min, y_min, x_max, y_max = self._bbox
        return [(x_min + x_max) / 2, (y_min + y_max) / 2]

    def get_bbox_traditional(self):
        return self._bbox

class FakeDetectedObject:
    def __init__(self, bbox, range_distance_m):
        self.bb = FakeBoundingBox(bbox)
        self.range_distance_m = range_distance_m

def serialize_detection_summary(detection_summary):
    summary_clean = {}
    for obj_id, obj_data in detection_summary.items():
        summary_clean[obj_id] = {
            'ground_truth': {},
            'associated_detections': {},
            'analysis': {}
        }

        for frame_id, gt in obj_data['ground_truth'].items():
            summary_clean[obj_id]['ground_truth'][frame_id] = {
                'bbox': gt.bb.get_bbox_traditional() if gt is not None else None,
                'range_distance_m': gt.range_distance_m if gt is not None else None
            }

        for frame_id, det in obj_data['associated_detections'].items():
            summary_clean[obj_id]['associated_detections'][frame_id] = det[0] if det is not None else None

        for frame_id, analysis in obj_data['analysis'].items():
            summary_clean[obj_id]['analysis'][frame_id] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in analysis.items()
            }

    return summary_clean

def deserialize_detection_summary(summary_clean):
    deserialized = {}
    for obj_id, obj_data in summary_clean.items():
        if 'Bird' in obj_id or 'Flock' in obj_id or 'Drone' in obj_id:
            continue
        deserialized[obj_id] = {
            'ground_truth': {},
            'associated_detections': {int(key): item for key, item in obj_data['associated_detections'].items()},
            'analysis': {int(key): item for key, item in obj_data['analysis'].items()}
        }

        for frame_id, gt in obj_data['ground_truth'].items():
            if gt['bbox'] is not None:
                deserialized[obj_id]['ground_truth'][int(frame_id)] = FakeDetectedObject(
                    bbox=gt['bbox'],
                    range_distance_m=gt.get('range_distance_m', None)
                )
            else:
                deserialized[obj_id]['ground_truth'][int(frame_id)] = None

    return deserialized

summary_path = 'summaries.json'

def rescale_distances(
    distances,
    fov_new_deg,
    res_new_px,
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
    
def rescale_distances_inverse(
    distances,
    fov_new_deg,
    res_new_px,
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

def import_encounter_data(summary_path=summary_path):
    if os.path.exists(summary_path):
        print("Loading precomputed detection summaries from file...")
        # Loading
        with open('summaries.json', 'r') as f:
            summary_loaded = json.load(f)

        # Reconstruct fake objects
        all_detection_summaries = [deserialize_detection_summary(s) for s in summary_loaded]
    else:
        view_frames = True
        view_plots_per_flight = False


        dataset = Dataset(local_path='C:/Users/adamg/Documents/Original/part2', s3_path='s3://airborne-obj-detection-challenge-training/part2/', download_if_required=True)

        all_flight_ids = dataset.get_flight_ids()
        all_detection_summaries = []
        n = 0
        bad_flights = {}
        bad_frames = {}
        for flight_id in all_flight_ids:
            n += 1
            print(f"...Processing flight {n} of {len(all_flight_ids)}: {flight_id}")

            predictions_dir = 'C:/Users/adamg/Downloads/Tracking_Results_Center_Track_First_Pass/Tracking_Results_Center_Track_First_Pass'

            flight = dataset.get_flight(flight_id) # Loading single flight
            #transformPredictions = FlightTransformPredictions(filepath=f'{predictions_dir}/{flight_id}/transforms.json', flight=flight)
            transformPredictions = None
            try:
                detectPredictions = FlightDetectionPredictions(filepath=f'{predictions_dir}/{flight_id}/detections.json', flight=flight)
                trackPredictions = FlightTrackPredictions(filepath=f'{predictions_dir}/{flight_id}/tracking.json', flight=flight, transform_predictions=transformPredictions)
            except:
                continue

            frames = flight.frames # All the frames of the flight
            frame_ids = list(frames.keys()) 

            ground_truth_objects = flight.detected_objects.keys()

            raw_detections = {}
            detection_summary = {x:{'ground_truth': {}, 'associated_detections': {}, 'analysis': {}} for x in ground_truth_objects}

            for frame_id in frame_ids:
                frame = flight.get_frame(frame_id)

                ### RECORD GROUND TRUTH AND RAW DETECTIONS FOR EACH FRAME ###
                for obj in ground_truth_objects:
                    detection_summary[obj]['ground_truth'][frame_id] = frame.detected_object_locations.get(obj, None)

                raw_detections_this_frame = detectPredictions.get_detections_for_frame(frame)
                tracker_detection_for_this = trackPredictions.get_predictions_for_frame(frame)
                
                use_raw_range_vals = True
                
                if len(tracker_detection_for_this) > 0:
                    if use_raw_range_vals:
                        for tdet in tracker_detection_for_this.iloc[0]:
                            for rdet in raw_detections_this_frame.iloc[0]:
                                if rdet['x'] == tdet['x']:
                                    tdet['d'] = rdet['d']
                                    break
                    raw_detections[frame_id] = tracker_detection_for_this
                else:
                    raw_detections[frame_id] = None
                

                ### ASSOCIATE RAW DETECTIONS TO GROUND TRUTH USING CENTRE DISTANCE HEURISTIC, AND PERFORM ANALYSIS ###
                if raw_detections[frame_id] is not None:
                    # Compute raw centers and bboxs from raw detections
                    raw_centers = {}
                    detection_confidences = {}
                    raw_bboxs = {}
                    for raw_det_idx, raw_det in enumerate(raw_detections[frame_id]):
                        raw_centers[raw_det_idx] = [raw_det[0]['x'] + np.ceil((raw_det[0]['w'] - raw_det[0]['x']))/2, raw_det[0]['y'] + np.ceil((raw_det[0]['h'] - raw_det[0]['y']))/2]
                        raw_bboxs[raw_det_idx] = [raw_det[0]['x'], raw_det[0]['y'], raw_det[0]['w'], raw_det[0]['h']]
                        detection_confidences[raw_det_idx] = raw_det[0]['s']
                    # Extract the center of the ground truth bbox and calculate the detection association distance
                    for obj in ground_truth_objects:
                        if detection_summary[obj]['ground_truth'][frame_id] is None:
                            continue
                        ground_truth_bbox_center = detection_summary[obj]['ground_truth'][frame_id].bb.get_center()
                        ground_truth_bbox_coords = detection_summary[obj]['ground_truth'][frame_id].bb.get_bbox_traditional()
                        ground_truth_range = detection_summary[obj]['ground_truth'][frame_id].range_distance_m
                        association_radius = max(20, np.sqrt(2)*0.5*max(ground_truth_bbox_coords[2] - ground_truth_bbox_coords[0], ground_truth_bbox_coords[3] - ground_truth_bbox_coords[1]))
                        # Find the closest raw detection
                        min_distance = float('inf')
                        max_confidence = 0.0
                        closest_idx = None

                        for raw_det_idx, raw_center in raw_centers.items():
                            distance = np.linalg.norm(np.array(raw_center) - np.array(ground_truth_bbox_center))
                            confidence = detection_confidences[raw_det_idx]
                            if distance < min_distance:
                                if confidence > max_confidence:
                                    min_distance = distance
                                    max_confidence = confidence
                                    closest_idx = raw_det_idx

                        if closest_idx is not None and min_distance <= association_radius:
                            closest_raw_detection = raw_detections[frame_id].iloc[closest_idx]  # Get the corresponding raw detection
                            detection_summary[obj]['associated_detections'][frame_id] = closest_raw_detection
                            closest_center = raw_centers[closest_idx]
                            this_confidence = detection_confidences[closest_idx]

                            closest_bbox = raw_bboxs[closest_idx]
                            is_edge = (
                                closest_bbox[0] < 2 or closest_bbox[1] < 2 or
                                closest_bbox[2] > 2448 - 2 or closest_bbox[3] > 2048 - 2
                            )
                            detection_summary[obj]['analysis'][frame_id] = {'x__y_distances': np.array(closest_center) - np.array(ground_truth_bbox_center), 's': closest_raw_detection[0]['d'] - ground_truth_range, 'center_distance': min_distance, 'pred_distance': closest_raw_detection[0]['d'], 'pred_bbox': closest_bbox, 'center': closest_center, 'is_edge': is_edge, 'timestamp': frame.time / 1e9, 'a': closest_raw_detection[0]['a'], 'confidence': this_confidence}
                            if abs(closest_raw_detection[0]['d'] - ground_truth_range) > 2000:
                                if abs(bad_flights.get(flight_id, 0)) < abs(closest_raw_detection[0]['d'] - ground_truth_range):
                                    bad_flights[flight_id] = closest_raw_detection[0]['d'] - ground_truth_range
                                if flight_id in bad_frames:
                                    bad_frames[flight_id].append(frame_id)
                                else:
                                    bad_frames[flight_id] = [frame_id]
                        else:
                            detection_summary[obj]['associated_detections'][frame_id] = None

                if view_plots_per_flight:

                    # Iterate through detection_summary and create scatter plots for each object
                    for obj, obj_data in detection_summary.items():
                        x_values = []
                        y_values = []
                        frame_numbers = []
                        distances = []

                        dx_values = []
                        dy_values = []
                        range_values = []
                        delta_x_values = []  # X-axis for the fourth graph
                        x_component_values = []  # Y-axis for the fourth graph
                        delta_y_values = []  # X-axis for the fifth graph
                        y_component_values = []  # Y-axis for the fifth graph
                        delta_range_values = []  # Colouring function for both the fourth and fifth graphs

                        prev_center = None

                        # Extract x and y components from the 'analysis' dictionary
                        for frame_id, analysis in sorted(obj_data['analysis'].items()):
                            x_component, y_component = analysis['x__y_distances']
                            x_values.append(x_component)
                            y_values.append(y_component)
                            frame_numbers.append(frame_id)

                            # Get range distance for color mapping
                            current_distance = obj_data['ground_truth'][frame_id].range_distance_m
                            distances.append(current_distance)

                            # Compute change in object center (dx, dy) between successive frames
                            current_center = obj_data['ground_truth'][frame_id].bb.get_center()
                            if prev_center is not None:
                                dx = current_center[0] - prev_center[0]
                                dy = current_center[1] - prev_center[1]

                                dx_values.append(dx)
                                dy_values.append(dy)
                                range_values.append(current_distance)  # Use range as x-axis in third plot

                                delta_x_values.append(dx)  # X-axis for the fourth graph
                                x_component_values.append(x_component)  # Y-axis for the fourth graph
                                delta_y_values.append(dy)  # X-axis for the fifth graph
                                y_component_values.append(y_component)  # Y-axis for the fifth graph
                                delta_range_values.append(current_distance)  # Colour function

                            prev_center = current_center

                        if len(x_values) > 0:
                            # Calculate statistics
                            mean_x = np.mean(x_values)
                            mean_y = np.mean(y_values)
                            var_x = np.var(x_values)
                            var_y = np.var(y_values)
                            cov_xy = np.cov(x_values, y_values)[0, 1]

                            # Compute the ellipse parameters for the 2-sigma contour
                            lambda_, v = np.linalg.eig(np.cov(x_values, y_values))
                            lambda_ = np.sqrt(lambda_)

                        # Create a single figure with five subplots
                        fig, axes = plt.subplots(3, 2, figsize=(18, 18))

                        # First subplot: Scatter plot for X and Y distances colored by range with mean and 2-sigma ellipse
                        scatter1 = axes[0, 0].scatter(x_values, y_values, c=distances, cmap='viridis', alpha=0.7, edgecolors='k')
                        axes[0, 0].set_xlabel('X Component of Distance')
                        axes[0, 0].set_ylabel('Y Component of Distance')
                        axes[0, 0].set_title(f'X and Y Distance Components for {obj}')
                        fig.colorbar(scatter1, ax=axes[0, 0], label='Range (m)')

                        # Plot mean lines
                        axes[0, 0].axvline(mean_x, color='red', linestyle='--', label='Mean X')
                        axes[0, 0].axhline(mean_y, color='blue', linestyle='--', label='Mean Y')

                        # Plot 2-sigma ellipse
                        ell = Ellipse(xy=(mean_x, mean_y), width=2*lambda_[0]*2, height=2*lambda_[1]*2,
                                    angle=np.degrees(np.arctan2(v[1, 0], v[0, 0])), edgecolor='black', facecolor='none', linestyle='dashed', label='2-Sigma Contour')
                        axes[0, 0].add_patch(ell)

                        axes[0, 0].legend()

                        # Second subplot (Middle): 3D scatter plot for X, Y, and Range
                        ax2 = fig.add_subplot(3, 2, 2, projection='3d')
                        scatter2 = ax2.scatter(x_values, y_values, distances, c=distances, cmap='viridis', alpha=0.7, edgecolors='k')
                        ax2.set_xlabel('X Component of Distance')
                        ax2.set_ylabel('Y Component of Distance')
                        ax2.set_zlabel('Range (m)')
                        ax2.set_title(f'3D Scatter Plot of X, Y, and Range for {obj}')
                        fig.colorbar(scatter2, ax=ax2, label='Range (m)')

                        # Third subplot: Range vs. ΔX and ΔY
                        axes[1, 0].scatter(range_values, dx_values, color='red', alpha=0.7, label='ΔX')
                        axes[1, 0].scatter(range_values, dy_values, color='blue', alpha=0.7, label='ΔY')
                        axes[1, 0].set_xlabel('Range (m)')
                        axes[1, 0].set_ylabel('Change in Object Center')
                        axes[1, 0].set_title(f'Change in Object Center (ΔX, ΔY) vs. Range for {obj}')
                        axes[1, 0].legend()

                        # Fourth subplot: X-component (from first graph) vs. ΔX (from third graph), colored by range
                        scatter4 = axes[1, 1].scatter(delta_x_values, x_component_values, c=delta_range_values, cmap='viridis', alpha=0.7, edgecolors='k')
                        axes[1, 1].set_xlabel('ΔX Component of Object Center')
                        axes[1, 1].set_ylabel('X Component of Distance (from First Graph)')
                        axes[1, 1].set_title(f'ΔX vs. X Component of Distance for {obj}')
                        fig.colorbar(scatter4, ax=axes[1, 1], label='Range (m)')

                        # Fifth subplot: Y-component (from first graph) vs. ΔY (from third graph), colored by range
                        scatter5 = axes[2, 0].scatter(delta_y_values, y_component_values, c=delta_range_values, cmap='viridis', alpha=0.7, edgecolors='k')
                        axes[2, 0].set_xlabel('ΔY Component of Object Center')
                        axes[2, 0].set_ylabel('Y Component of Distance (from First Graph)')
                        axes[2, 0].set_title(f'ΔY vs. Y Component of Distance for {obj}')
                        fig.colorbar(scatter5, ax=axes[2, 0], label='Range (m)')

                        # Hide the last empty subplot (2,1) since we have an odd number of plots
                        axes[2, 1].axis('off')

                        # Show the figure with proper layout
                        plt.tight_layout()
                        plt.show()

            if flight_id in bad_flights:
                view_frames = True
            else:
                view_frames = False

            view_frames = False

            if view_frames:
                annotated_frames = []
                frame_idx = 0
                for frame_id in frame_ids:
                    if frame_id in bad_frames[flight_id][:10]:
                        frame_idx += 1
                        print(f"Downloading Frame {frame_idx} of {len(bad_frames[flight_id])}")
                        frame = flight.get_frame(frame_id)
                        scale = 0.5 # scale images and bbox by half size for viewing
                        annotated_frame = frame.image_annotated(scale=scale, download_if_required=True)
                        annotated_frame = trackPredictions.annotate_frame_with_predictions(annotated_frame, frame, scale)
                        #annotated_frame = detectPredictions.annotate_frame_with_detections(annotated_frame, frame, scale)
                        annotated_frames.append(annotated_frame)

                        save_path_raw = f'C:/Users/adamg/Documents/Original/part2/annotated_images/raw/{frame.image_path()}'
                        save_path_annotated = f'C:/Users/adamg/Documents/Original/part2/annotated_images/annotated/{frame.image_path()}'
                        os.makedirs(os.path.dirname(save_path_raw), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_annotated), exist_ok=True)
                        cv2.imwrite(save_path_raw, frame.image(type='cv2'))
                        cv2.imwrite(save_path_annotated, annotated_frame)
                
                for annotated_frame in annotated_frames:
                    cv2.imshow("Frame", annotated_frame)
                    if (cv2.waitKey(200) & 0xFF) == ord('q'):
                        break
                    cv2.destroyAllWindows()

            all_detection_summaries.append(detection_summary)

        # Saving
        summary_serialized = [serialize_detection_summary(s) for s in all_detection_summaries]
        with open('summaries.json', 'w') as f:
            json.dump(summary_serialized, f)

        # Loading
        with open('summaries.json', 'r') as f:
            summary_loaded = json.load(f)

        # Reconstruct fake objects
        all_detection_summaries = [deserialize_detection_summary(s) for s in summary_loaded]

    return all_detection_summaries

def apply_camera_rescaling(all_detection_summaries,
    fov_new_deg,
    res_new_px,
    fov_old_deg = AMAZON_CAMERA_FOV[1],
    res_old_px = AMAZON_CAMERA_RES[1]
    ):

    print("Rescaling Data")

    all_detection_summaries = deepcopy(all_detection_summaries)

    for detection_summary in all_detection_summaries:
        for obj in detection_summary:
            for frame_id in detection_summary[obj]['analysis']:
                analysis_entry = detection_summary[obj]['analysis'][frame_id]
                if analysis_entry is None:
                    continue

                # Original ground truth and error
                gt_distance = detection_summary[obj]['ground_truth'][frame_id].range_distance_m
                pred_distance = detection_summary[obj]['analysis'][frame_id]['pred_distance']

                # Rescale gt and predicted distances
                new_gt_distance = rescale_distances(
                    gt_distance,
                    fov_new_deg,
                    res_new_px,
                    fov_old_deg,
                    res_old_px
                )

                new_pred_distance = rescale_distances(
                    pred_distance,
                    fov_new_deg,
                    res_new_px,
                    fov_old_deg,
                    res_old_px
                )

                new_error = new_pred_distance - new_gt_distance

                # Overwrite values
                detection_summary[obj]['ground_truth'][frame_id].range_distance_m = new_gt_distance
                analysis_entry['pred_distance'] = new_pred_distance
                analysis_entry['s'] = new_error

    return all_detection_summaries

def generate_pixel_uncertainty_model(all_detection_summaries = None, bins = 20, plot_pixel_uncertainties = False, camera_res_xy = AMAZON_CAMERA_RES, camera_fov_xy = AMAZON_CAMERA_FOV, pregen_state_data = None):
    if pregen_state_data is None:

        if all_detection_summaries == None:
            print("No Data Specified!")
            return

        ignore_edges = True
        sigma_prune = 20
        max_range = 0

        # Iterate through detection_summary and extract uncertainty statistics
        x_values = []
        y_values = []
        #s_values = []
        obj_types = []
        distances = []
        frame_numbers = []
        bbox_centroids = []

        for detection_summary in all_detection_summaries:
            for obj, obj_data in detection_summary.items():
                # Extract x and y components from the 'analysis' dictionary
                for frame_id, analysis in sorted(obj_data['analysis'].items()):
                    if ignore_edges:
                        if analysis['is_edge']:
                            continue
                    obj_types.append(obj)
                    x_component, y_component = analysis['x__y_distances']
                    x_values.append(x_component)
                    y_values.append(y_component)
                    #s_values.append(analysis['s'])
                    frame_numbers.append(frame_id)

                    # Get range distance for color mapping
                    current_distance = obj_data['ground_truth'][frame_id].range_distance_m
                    max_range = max(max_range, current_distance)
                    distances.append(current_distance)

        # Remove None values
        x_values = np.array(x_values, dtype=object)  # Use object dtype to allow None
        y_values = np.array(y_values, dtype=object)
        #s_values = np.array(s_values, dtype=object)
        distances = np.array(distances, dtype=object)
        obj_types = np.array(obj_types, dtype=object)

        #valid_mask = (x_values != None) & (y_values != None) & (distances != None) #& (s_values != None)  # Mask to remove None
        valid_mask = (
            (x_values != None) & (y_values != None) & (distances != None) &
            ~np.isnan(x_values.astype(float)) & ~np.isnan(y_values.astype(float)) & ~np.isnan(distances.astype(float))
        )

        # Apply mask
        x_values = x_values[valid_mask].astype(float)
        y_values = y_values[valid_mask].astype(float)
        #s_values = s_values[valid_mask].astype(float)
        distances = distances[valid_mask].astype(float)
        obj_types = obj_types[valid_mask]

        if len(x_values) > 0:
            # Calculate statistics
            mean_x = np.mean(x_values)
            mean_y = np.mean(y_values)
            #mean_s = np.mean(s_values)
            var_x = np.var(x_values)
            var_y = np.var(y_values)
            #var_s = np.var(s_values)
            cov_xy = np.cov(x_values, y_values)[0, 1]

            # Compute the ellipse parameters for the 2-sigma contour
            lambda_, v = np.linalg.eig(np.cov(x_values, y_values))
            lambda_ = np.sqrt(lambda_)

        # Compute bin edges using Jenks natural breaks on the distances
        # Step 1: Construct Z
        z_values = np.where(np.abs(x_values) > np.abs(y_values), x_values, y_values)
        D = np.array(distances)
        Z = np.array(z_values)

        # Step 1: Construct uniform base bins along D
        base_k = 150
        base_edges = np.linspace(D.min(), D.max(), base_k + 1)
        bin_indices = np.digitize(D, base_edges) - 1
        bin_indices = np.clip(bin_indices, 0, base_k - 1)

        # Step 2: Group Z values by base bins
        base_bin_values = [Z[bin_indices == i] for i in range(base_k)]
        base_bin_counts = np.array([len(b) for b in base_bin_values])
        base_bin_width = base_edges[1] - base_edges[0]

        # Step 3: Precompute NLL for all merges of consecutive base bins
        nll_matrix = np.full((base_k, base_k), np.inf)
        mean_matrix = np.full((base_k, base_k), np.inf)
        var_matrix = np.full((base_k, base_k), np.inf)
        for i in range(base_k):
            for j in range(i + 1, base_k + 1):

                # Check constraints: min 1000 points and min 150m total width
                point_count = np.sum(base_bin_counts[i:j])
                dist_span = base_edges[j] - base_edges[i]
                if base_edges[i] > 1500:
                    if point_count < 0 or dist_span < 200:
                        continue

                # Determine the max number of points in any contributing base bin
                max_points = max(len(base_bin_values[b]) for b in range(i, j) if len(base_bin_values[b]) > 0)

                equalized_bins = []
                for b in range(i, j):
                    values = base_bin_values[b]
                    if len(values) == 0:
                        continue

                    full_repeats = max_points // len(values)
                    remainder = max_points % len(values)

                    repeated = np.tile(values, full_repeats)  # duplicate entire bin
                    if remainder > 0:
                        sampled = np.random.choice(values, size=remainder, replace=True)
                        resampled = np.concatenate((repeated, sampled))
                    else:
                        resampled = repeated

                    equalized_bins.append(resampled)

                z_segment = np.concatenate(equalized_bins)
                if len(z_segment) < 2:
                    continue
                try:
                    # Initial stats
                    mu = np.mean(z_segment)
                    sigma = np.std(z_segment)

                    # 5-sigma pruning
                    pruned_segment = z_segment[np.abs(z_segment - mu) <= sigma_prune * sigma]
                    
                    # Recompute stats after pruning
                    if len(pruned_segment) < 2:
                        continue
                    mu2 = np.mean(pruned_segment)
                    sigma2 = np.var(pruned_segment)

                    # Avoid log(0)
                    if sigma2 > 1e-6:
                        nll = 0.5 * len(pruned_segment) * (np.log(2 * np.pi * sigma2) + 1)
                        nll_matrix[i, j - 1] = nll
                        mean_matrix[i, j - 1] = mu2
                        var_matrix[i, j - 1] = sigma2
                except:
                    continue

        # Step 4: Dynamic programming to merge base bins into 'bins' segments
        dp = np.full((bins + 1, base_k + 1), np.inf)
        split = np.full((bins + 1, base_k + 1), -1)
        dp[0, 0] = 0

        for b in range(1, bins + 1):
            for j in range(1, base_k + 1):
                for i in range(j):
                    if dp[b - 1, i] + nll_matrix[i, j - 1] < dp[b, j]:
                        dp[b, j] = dp[b - 1, i] + nll_matrix[i, j - 1]
                        split[b, j] = i

        # Backtrack to get best split
        boundaries = []
        j = base_k
        for b in range(bins, 0, -1):
            i = split[b, j]
            boundaries.append(i)
            j = i

        boundaries = boundaries[::-1]  # make them ascending
        if boundaries[0] != 0:
            boundaries = [0] + boundaries
        if boundaries[-1] != base_k:
            boundaries = boundaries + [base_k]
        bin_edges = [base_edges[i] for i in boundaries]

        bin_means = []
        bin_vars = []

        # Loop through each merged bin segment
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Pull the precomputed mean and var for the full segment
            mu = mean_matrix[start, end - 1]
            var = var_matrix[start, end - 1]
            
            bin_means.append(mu)
            bin_vars.append(var)

        # # Compute bin statistics: mean and variance of Z for each merged bin
        # bin_means = []
        # bin_vars = []
        # bin_counts = []

        # for i in range(len(bin_edges) - 1):
        #     mask = (D >= bin_edges[i]) & (D < bin_edges[i + 1])
        #     z_vals = Z[mask]
        #     if len(z_vals) > 0:
        #         # Initial stats
        #         mu = np.mean(z_vals)
        #         sigma = np.std(z_vals)

        #         # 5-sigma pruning
        #         pruned_segment = z_vals[np.abs(z_vals- mu) <= sigma_prune * sigma]

        #         bin_means.append(np.mean(pruned_segment))
        #         bin_vars.append(np.var(pruned_segment))
        #         bin_counts.append(len(pruned_segment))
        #     else:
        #         bin_means.append(np.nan)
        #         bin_vars.append(np.nan)
        #         bin_counts.append(0)

        # for i in range(len(boundaries) - 1):
        #     start, end = boundaries[i], boundaries[i + 1]
        #     print(f'Merged Bin {i + 1}: Base bins {start} to {end - 1}')

        #     fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False, gridspec_kw={'height_ratios': [1, 1.5, 1.5]})

        #     # --- Subplot 1: Scatter plot of values per base bin ---
        #     for b in range(start, end):
        #         if b < len(base_bin_values) and len(base_bin_values[b]) > 0:
        #             y = base_bin_values[b]
        #             x = np.full_like(y, b)
        #             axes[0].scatter(x, y, label=f'Base Bin {b}', s=10)

        #     axes[0].set_title(f'Merged Bin {i + 1} — Base Bin Scatter')
        #     axes[0].set_ylabel('Z Values')
        #     axes[0].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')

        #     # --- Subplot 2: Histogram overlays ---
        #     all_z_vals = []
        #     for b in range(start, end):
        #         if b < len(base_bin_values) and len(base_bin_values[b]) > 0:
        #             axes[1].hist(base_bin_values[b], bins=30, alpha=0.5, label=f'Base Bin {b}', density=True)
        #             all_z_vals.extend(base_bin_values[b])

        #     # Overlay normal distribution using pruned bin stats
        #     mu = bin_means[i]
        #     var = bin_vars[i]
        #     if not np.isnan(mu) and var > 0:
        #         sigma = np.sqrt(var)
        #         x_vals = np.linspace(min(all_z_vals), max(all_z_vals), 300)
        #         y_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
        #         axes[1].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=2, label='Fitted Normal')

        #         # # Add vertical lines at ±prune sigma
        #         # axes[1].axvline(mu - sigma_prune * sigma, color='red', linestyle='--', linewidth=1.5, label=f'±{sigma_prune}σ' if i == 0 else None)
        #         # axes[1].axvline(mu + sigma_prune * sigma, color='red', linestyle='--', linewidth=1.5)

        #     axes[1].set_title('Histogram Overlay by Base Bin')
        #     axes[1].set_xlabel('Z Values')
        #     axes[1].set_ylabel('Density')
        #     axes[1].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')

        #     # --- Subplot 3: Combined histogram for entire merged bin ---
        #     axes[2].hist(all_z_vals, bins=30, color='gray', alpha=0.7, density=True, label='Combined Histogram')
        #     if not np.isnan(mu) and var > 0:
        #         axes[2].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=2, label='Fitted Normal')
        #         # axes[2].axvline(mu - sigma_prune * sigma, color='red', linestyle='--', linewidth=1.5, label=f'±{sigma_prune}σ' if i == 0 else None)
        #         # axes[2].axvline(mu + sigma_prune * sigma, color='red', linestyle='--', linewidth=1.5)

        #         mu2 = np.mean(all_z_vals)
        #         var2 = np.var(all_z_vals)
        #         print(mu2, var2)
        #         sigma2 = np.sqrt(var2)
        #         x_vals = np.linspace(min(all_z_vals), max(all_z_vals), 300)
        #         y_vals2 = norm.pdf(x_vals, loc=mu2, scale=sigma2)
        #         axes[2].plot(x_vals, y_vals2, color='red', linestyle='--', linewidth=2, label='Fitted Normal Original')

        #     axes[2].set_title('Combined Histogram for Merged Bin')
        #     axes[2].set_xlabel('Z Values')
        #     axes[2].set_ylabel('Density')
        #     axes[2].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')

        #     plt.tight_layout()
        #     plt.show()

        # Manually enforce floor/ceiling
        bin_edges[0] = 0
        bin_edges[-1] = 7000

        bin_edges = [int(x) for x in bin_edges]

        # Digitize: find the bin index for each distance
        bin_indices = np.digitize(distances, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)  # ensure in bounds

        # Count samples in each bin
        #bin_counts = np.bincount(bin_indices, minlength=bins)

        # Output
        print(bin_edges)
        print(bin_vars)
        #print(bin_counts)

        # Generate bin edges
        # bin_edges = np.linspace(0, cutoff_dist, bins + 1)
        # Compute means and standard deviations for each x, y and range bin
        x_values_binned, y_values_binned, bin_sample_counts = [], [], []
        x_px_bin_means, x_px_bin_vars, x_px_bin_centers = [], [], []
        y_px_bin_means, y_px_bin_vars, y_px_bin_centers = [], [], []
        xy_px_bin_covars, xy_px_bin_centers = [], []

        for i in range(len(bin_edges) - 1):
            bin_mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            
            x_bin_data = x_values[bin_mask]
            y_bin_data = y_values[bin_mask]
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2

            if len(x_bin_data) > 0 and len(y_bin_data) > 0:
                # Save Binned Values
                x_values_binned.append(x_bin_data)
                y_values_binned.append(y_bin_data)
                bin_sample_counts.append(len(x_bin_data))

                # Means and variances for x and y
                x_mean_this = np.mean(x_bin_data)
                y_mean_this = np.mean(y_bin_data)
                x_var_this = np.var(x_bin_data)
                y_var_this = np.var(y_bin_data)

                x_px_bin_means.append(x_mean_this)
                x_px_bin_vars.append(x_var_this)

                y_px_bin_means.append(y_mean_this)
                y_px_bin_vars.append(y_var_this)

                # Covariance between x and y
                covar = np.cov(x_bin_data, y_bin_data, ddof=0)[0, 1]
                xy_px_bin_covars.append(covar)

                # Elementwise max of absolute x or y, then compute mean/var
                xy_px_bin_centers.append(bin_center)
            else:
                # Save Binned Values
                x_values_binned.append(None)
                y_values_binned.append(None)

                # Handle empty bins
                x_px_bin_means.append(None)
                x_px_bin_vars.append(None)

                y_px_bin_means.append(None)
                y_px_bin_vars.append(None)

                xy_px_bin_centers.append(bin_center)

        # Compute total variance after bin loop
        xy_px_bin_total_var = [m**2 + v for m, v in zip(bin_means, bin_vars)]

        sample_pixel_uncertainty_state_data = (bin_edges, bins, xy_px_bin_total_var, AMAZON_CAMERA_FOV, AMAZON_CAMERA_RES)

    else:
        sample_pixel_uncertainty_state_data = pregen_state_data   

    def sample_pixel_uncertainty(gt_dist, num_samples=2, new_fov=None, new_res=None, state_data = sample_pixel_uncertainty_state_data):
        bin_edges, bins, xy_px_bin_total_var, AMAZON_CAMERA_FOV, AMAZON_CAMERA_RES = state_data
        
        if new_fov is None:
            new_fov = AMAZON_CAMERA_FOV[1]
        if new_res is None:
            new_res = AMAZON_CAMERA_RES[1]

        gt_dist_inv = rescale_distances_inverse(
            gt_dist,
            new_fov,
            new_res,
            fov_old_deg = AMAZON_CAMERA_FOV[1],
            res_old_px = AMAZON_CAMERA_RES[1]
        )
        bin_idx = np.digitize(gt_dist_inv, bin_edges) - 1  # bin index
        bin_idx = np.clip(bin_idx, 0, bins - 1)  # clip to valid index ran
        var = xy_px_bin_total_var[bin_idx]
        std = np.sqrt(max(var, 1e-12))  # ensure non-negative, avoid sqrt(0)

        sampled_error = np.random.normal(loc=0.0, scale=std, size=num_samples)


        if num_samples > 1:
            return tuple(sampled_error)
        else:
            return float(sampled_error)

    #### Plot Pixel Uncertainty Data ####

    if plot_pixel_uncertainties:
        # --- Recompute starts for flattened z ---
        starts = np.zeros(base_k + 1, dtype=np.int32)
        for i in range(base_k):
            starts[i + 1] = starts[i] + len(base_bin_values[i])

        # --- Flatten z_values and corresponding distances ---
        z_flat_plot = np.concatenate(base_bin_values)

        # Flatten corresponding distances for each z_value
        base_bin_distances = [D[bin_indices == i] for i in range(base_k)]
        d_flat_plot = np.concatenate(base_bin_distances)

        # --- Compute variance per base bin ---
        base_bin_variances = [np.var(b) if len(b) > 0 else 0 for b in base_bin_values]

        # --- Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False, gridspec_kw={'height_ratios': [3, 1]})

        # --- Top plot: Distance vs z_values ---
        ax1.scatter(d_flat_plot, z_flat_plot, color='black', linewidth=0.8, label='z_values', alpha=0.05)
        for edge in bin_edges:
            ax1.axvline(x=edge, color='red', linestyle='--', linewidth=1)
        ax1.set_title("Distance vs z_value with Merged Bin Boundaries")
        ax1.set_xlabel("Distance (m)")
        ax1.set_ylabel("z_value")
        ax1.legend()

        # --- Bottom plot: Variance of each base bin ---
        ax2.bar(base_edges[:-1], base_bin_variances, width=0.8 * base_bin_width, color='gray', edgecolor='black', align='edge')
        for boundary in bin_edges[1:-1]:  # exclude first and last
            ax2.axvline(x=boundary, color='red', linestyle='--', linewidth=1)
        ax2.set_title("Per-Base-Bin Variance")
        ax2.set_xlabel("Base Bin Index")
        ax2.set_ylabel("Variance")

        plt.tight_layout()
        plt.show()

        # Create a single figure with five subplots
        fig, axes = plt.subplots(2, 3, figsize=(8, 8))

        # Create a colormap for unique object types
        unique_obj_types = sorted(set(obj_types))
        cmap = plt.get_cmap('tab10')  # Or 'tab20' if you have many types
        color_map = {obj: cmap(i % cmap.N) for i, obj in enumerate(unique_obj_types)}

        # Map colors to each point
        colors = [color_map[obj] for obj in obj_types]

        # First subplot: Scatter plot for X and Y distances colored by range with mean and 2-sigma ellipse
        scatter1 = axes[0, 0].scatter(x_values, y_values, alpha=0.7, edgecolors='k', c=colors)
        axes[0, 0].set_xlabel('X Component of Distance')
        axes[0, 0].set_ylabel('Y Component of Distance')
        axes[0, 0].set_title(f'X and Y Distance Components')
        fig.colorbar(scatter1, ax=axes[0, 0], label='Range (m)')

        # Plot mean lines
        axes[0, 0].axvline(mean_x, color='red', linestyle='--', label='Mean X')
        axes[0, 0].axhline(mean_y, color='blue', linestyle='--', label='Mean Y')  

        # Add a legend for object types
        for obj, color in color_map.items():
            axes[0, 0].scatter([], [], color=color, label=obj)
        axes[0, 0].legend(title='Object Type')

        # Plot 2-sigma ellipse
        ell = Ellipse(xy=(mean_x, mean_y), width=2*lambda_[0]*2, height=2*lambda_[1]*2,
                    angle=np.degrees(np.arctan2(v[1, 0], v[0, 0])), edgecolor='red', facecolor='none', linestyle='dashed', label='2-Sigma Contour')
        axes[0, 0].add_patch(ell)

        axes[0, 0].legend()
        axes[0, 0].set_aspect('equal', adjustable='datalim')


        # Second subplot (Top-right): 3D scatter plot (stored in a separate variable)
        ax3d = fig.add_subplot(3, 2, 2, projection='3d')  # Use a separate variable for the 3D subplot
        scatter2 = ax3d.scatter(x_values, y_values, distances, c=distances, cmap='viridis', alpha=0.7, edgecolors='k')
        ax3d.set_xlabel('X Component of Distance')
        ax3d.set_ylabel('Y Component of Distance')
        ax3d.set_zlabel('Range (m)')
        ax3d.set_title(f'3D Scatter Plot of X, Y, and Range')
        fig.colorbar(scatter2, ax=ax3d, label='Range (m)')

        # Define bin number to highlight

        highlight_bin = 2 # Change this to highlight a different bin (1-based index)

        # Define colormap for bin colors (red to blue gradient) using the new method
        colormap = colormap = cm.get_cmap("coolwarm")

        # First pass: Plot all bins with low transparency
        for i in range(bins):
            if i + 1 == highlight_bin:  # Skip the highlighted bin for now
                continue

            bin_color = colormap(i / bins)  

            # Scatter points (faint)
            axes[0, 1].scatter(x_values_binned[i], y_values_binned[i], color=bin_color, alpha=0.1, edgecolors='k')#, label=f'{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f} m')

            # Compute covariance and ellipse parameters
            cov_matrix = np.cov(x_values_binned[i], y_values_binned[i])
            lambda_bin, v_bin = np.linalg.eig(cov_matrix)
            lambda_bin = np.sqrt(lambda_bin)

            # Mean lines and ellipse (faint)
            axes[0, 1].axvline(x_px_bin_means[i], color=bin_color, linestyle='--', alpha=0.2)
            axes[0, 1].axhline(y_px_bin_means[i], color=bin_color, linestyle='--', alpha=0.2)
            ell_bin = Ellipse(
                xy=(x_px_bin_means[i], y_px_bin_means[i]),
                width=2 * lambda_bin[0] * 2,
                height=2 * lambda_bin[1] * 2,
                angle=np.degrees(np.arctan2(v_bin[1, 0], v_bin[0, 0])),
                edgecolor=bin_color,
                facecolor='none',
                linestyle='dashed',
                alpha=0.2
            )
            axes[0, 1].add_patch(ell_bin)

        # Second pass: Plot the highlighted bin on top
        bin_color = colormap((highlight_bin - 1) / bins)  

        # Scatter points (strong visibility)
        axes[0, 1].scatter(x_values_binned[i], y_values_binned[i], color=bin_color, alpha=1, edgecolors='k', label=f'{bin_edges[highlight_bin-1]:.1f} - {bin_edges[highlight_bin]:.1f} m')

        # Mean lines and ellipse (strong visibility)
        axes[0, 1].axvline(x_px_bin_means[i], color=bin_color, linestyle='--', alpha=0.8)
        axes[0, 1].axhline(y_px_bin_means[i], color=bin_color, linestyle='--', alpha=0.8)
        ell_bin = Ellipse(
            xy=(x_px_bin_means[i], y_px_bin_means[i]),
            width=2 * lambda_bin[0] * 2,
            height=2 * lambda_bin[1] * 2,
            angle=np.degrees(np.arctan2(v_bin[1, 0], v_bin[0, 0])),
            edgecolor=bin_color,
            facecolor='none',
            linestyle='dashed',
            alpha=0.8
        )
        axes[0, 1].add_patch(ell_bin)

        # Set labels and title for scatter plot
        axes[0, 1].set_xlabel('X Component of Distance')
        axes[0, 1].set_ylabel('Y Component of Distance')
        axes[0, 1].set_title(f'X and Y Distance Components Split by Range Bins for {obj} (Highlight Bin {highlight_bin})')
        axes[0, 1].legend(title='Range Bins (m)')
        axes[0, 1].set_aspect('equal', adjustable='datalim')

        ### 4th subplot: Mean X and Y vs Range Bin (axes[1, 0])
        # Sort bin_centers and mean_x_list in increasing order of bin_centers
        sorted_indices = np.argsort(xy_px_bin_centers)
        bin_centers = np.array(xy_px_bin_centers)[sorted_indices]
        mean_x_list = np.array(x_px_bin_means)[sorted_indices]
        mean_y_list = np.array(y_px_bin_means)[sorted_indices]
        axes[1, 0].plot(xy_px_bin_centers, x_px_bin_means, marker='o', linestyle='-', color='red', label='Mean X')
        #axes[1, 0].plot(bin_centers, np.abs(mean_x_list), marker='o', linestyle='--', color='red', label='Mean X Abs')
        axes[1, 0].plot(xy_px_bin_centers, y_px_bin_means, marker='s', linestyle='-', color='blue', label='Mean Y')
        #axes[1, 0].plot(bin_centers_sorted, np.abs(mean_y_list_sorted), marker='s', linestyle='--', color='blue', label='Mean Y Abs')
        axes[1, 0].axvline(xy_px_bin_centers[highlight_bin - 1], color='black', linestyle='--', alpha=0.8, label=f'Highlight Bin {highlight_bin}')
        axes[1, 0].set_xlabel('Range Bin (m)')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].set_title('Mean X and Y vs Range Bin')
        axes[1, 0].legend()

        ### 5th subplot: Variance X and Y vs Range Bin (axes[1, 1])
        # axes[1, 1].plot(bin_centers, var_x_list, marker='o', linestyle='-', color='red', label='Variance X')
        # axes[1, 1].plot(bin_centers, var_y_list, marker='s', linestyle='-', color='blue', label='Variance Y')
        # axes[1, 1].axvline(bin_centers[highlight_bin - 1], color='black', linestyle='--', alpha=0.8, label=f'Highlight Bin {highlight_bin}')
        # axes[1, 1].set_xlabel('Range Bin (m)')
        # axes[1, 1].set_ylabel('Variance Value')
        # axes[1, 1].set_title('Variance X and Y vs Range Bin')
        # axes[1, 1].legend()

        # Create the primary y-axis for variance
        ax1 = axes[1, 1]

        # Plot Variance X and Y
        ax1.plot(bin_centers, x_px_bin_vars, marker='o', linestyle='-', color='red', label='Variance X')
        ax1.plot(bin_centers, y_px_bin_vars, marker='s', linestyle='-', color='blue', label='Variance Y')
        ax1.axvline(bin_centers[highlight_bin - 1], color='black', linestyle='--', alpha=0.8, label=f'Highlight Bin {highlight_bin}')

        # Labels and Title
        ax1.set_xlabel('Range Bin (m)')
        ax1.set_ylabel('Variance Value')
        ax1.set_title('Variance X and Y vs Range Bin')

        # Plot histogram in the background (without displaying its axis)
        ax1.bar(bin_centers, max(max(y_px_bin_vars), max(x_px_bin_vars))*np.array(bin_sample_counts)/max(bin_sample_counts), width=0.9 * (max(bin_edges) - min(bin_edges)) / len(bin_centers), 
                color='grey', alpha=0.5, label=f'Sample Count (max = {max(bin_sample_counts)})')

        # Create a secondary y-axis for variance difference (on the right)
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, np.array(x_px_bin_vars) - np.array(y_px_bin_vars), marker='^', linestyle='-', color='green', label='Variance Difference')

        # Label the secondary y-axis
        ax2.set_ylabel('Variance Difference')

        # Remove the histogram y-axis but keep its label in the legend
        ax2.spines['right'].set_visible(True)  # Keep the variance difference axis
        ax1.spines['right'].set_visible(False)  # Hide any extra axis for histogram
        ax1.yaxis.set_label_position("left")
        ax1.yaxis.tick_left()

        # Bring variance plot to the front
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ### 6th subplot: Covariance X and Y vs Range Bin (axes[1, 1])
        axes[1, 2].plot(bin_centers, xy_px_bin_covars, marker='o', linestyle='-', color='red', label='Covariance XY')
        axes[1, 2].axvline(bin_centers[highlight_bin - 1], color='black', linestyle='--', alpha=0.8, label=f'Highlight Bin {highlight_bin}')
        axes[1, 2].set_xlabel('Range Bin (m)')
        axes[1, 2].set_ylabel('Covariance Value')
        axes[1, 2].set_title('Covariance XY vs Range Bin')
        axes[1, 2].legend()

        '''
        # Third subplot: Range vs. ΔX and ΔY
        axes[1, 0].scatter(range_values, dx_values, color='red', alpha=0.7, label='ΔX')
        axes[1, 0].scatter(range_values, dy_values, color='blue', alpha=0.7, label='ΔY')
        axes[1, 0].set_xlabel('Range (m)')
        axes[1, 0].set_ylabel('Change in Object Center')
        axes[1, 0].set_title(f'Change in Object Center (ΔX, ΔY) vs. Range for {obj}')
        axes[1, 0].legend()

        # Fourth subplot: X-component (from first graph) vs. ΔX (from third graph), colored by range
        scatter4 = axes[1, 1].scatter(delta_x_values, x_component_values, c=delta_range_values, cmap='viridis', alpha=0.7, edgecolors='k')
        axes[1, 1].set_xlabel('ΔX Component of Object Center')
        axes[1, 1].set_ylabel('X Component of Distance (from First Graph)')
        axes[1, 1].set_title(f'ΔX vs. X Component of Distance for {obj}')
        fig.colorbar(scatter4, ax=axes[1, 1], label='Range (m)')

        # Fifth subplot: Y-component (from first graph) vs. ΔY (from third graph), colored by range
        scatter5 = axes[2, 0].scatter(delta_y_values, y_component_values, c=delta_range_values, cmap='viridis', alpha=0.7, edgecolors='k')
        axes[2, 0].set_xlabel('ΔY Component of Object Center')
        axes[2, 0].set_ylabel('Y Component of Distance (from First Graph)')
        axes[2, 0].set_title(f'ΔY vs. Y Component of Distance for {obj}')
        fig.colorbar(scatter5, ax=axes[2, 0], label='Range (m)')

        # Hide the last empty subplot (2,1) since we have an odd number of plots
        axes[2, 1].axis('off')
        '''
        # Show the figure with proper layout
        print(f"Mean x: {mean_x}, Mean y: {mean_y}, Var x: {var_x}, Var y: {var_y}, Covar_xy: {cov_xy}")
        plt.show()

        ##### NEW FIGURE - HEATMAP ###########

        # Define adaptive limits using percentiles (to mimic scatterplot behavior)
        # xlim = np.percentile(x_values, [1, 99])
        # ylim = np.percentile(y_values, [1, 99])

        # Define the number of bins for splitting the dataset
        # n_bins = 16  # Adjust as needed
        n_bins = bins

        # Compute percentile-based bin edges for distances
        # distance_bins = np.linspace(min(distances), max(distances), n_bins + 1)
        distance_bins = bin_edges
        #distance_bins = np.linspace(min(distances), 3600, n_bins + 1)
        # Determine subplot grid size based on n_bins
        grid_size = int(np.ceil(np.sqrt(n_bins)))  # Create a square-like layout

        # Create figure with subplots
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(4 * grid_size, 4 * grid_size), constrained_layout=True)
        axes = np.array(axes).reshape(grid_size, grid_size)  # Ensure it's a 2D array

        for i in range(n_bins):
            # Get the correct subplot location
            row, col = divmod(i, grid_size)
            ax = axes[row, col]

            ### CREATE PLOT USING TRIMMED VALUES ####

            # Set artificial limits for x and y axes
            xlim = np.percentile(x_values, [1, 99])
            ylim = np.percentile(y_values, [1, 99])

            # Filter data for the current bin for display
            display_mask = (
                (distances >= distance_bins[i]) & 
                (distances < distance_bins[i + 1]) & 
                (x_values >= xlim[0]) & (x_values <= xlim[1]) & 
                (y_values >= ylim[0]) & (y_values <= ylim[1])
            )
            x_subset_display = x_values[display_mask]
            y_subset_display = y_values[display_mask]

            # Define bin edges for heatmap
            x_bins = np.arange(min(x_subset_display), max(x_subset_display) + 0.1, 0.1)
            y_bins = np.arange(min(y_subset_display), max(y_subset_display) + 0.1, 0.1)

            # Compute 2D histogram for the subset
            heatmap_counts, x_edges, y_edges = np.histogram2d(x_subset_display, y_subset_display, bins=[x_bins, y_bins])

            # Plot heatmap
            heatmap_plot = ax.imshow(
                heatmap_counts.T,  # Transpose to align with correct orientation
                origin='lower',
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],  # Match bin edges
                cmap='viridis',
                aspect='auto'
            )

            # Add black dashed lines at x=0 and y=0
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)

            ### CALCULATE AND PLOT STATISTICS USING UNTRIMMED VALUES ####

            # Filter data for the current bin for calculations (no trimming)
            mask = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
            x_subset = x_values[mask]
            y_subset = y_values[mask]

            # Compute statistics
            mean_x_bin = np.mean(x_subset)
            mean_y_bin = np.mean(y_subset)
            var_x_bin = np.var(x_subset)
            var_y_bin = np.var(y_subset)
            covar_xy = np.cov(x_subset, y_subset)[0, 1]

            # Compute covariance and ellipse parameters
            cov_matrix = np.cov(x_subset, y_subset)
            lambda_bin, v_bin = np.linalg.eig(cov_matrix)
            lambda_bin = np.sqrt(lambda_bin)

            # Mean lines and ellipse (faint)
            ax.axvline(mean_x_bin, color='red', linestyle='--', alpha=0.4)
            ax.axhline(mean_y_bin, color='red', linestyle='--', alpha=0.4)
            ell_bin = Ellipse(
                xy=(mean_x_bin, mean_y_bin),
                width=2 * lambda_bin[0],
                height=2 * lambda_bin[1],
                angle=np.degrees(np.arctan2(v_bin[1, 0], v_bin[0, 0])),
                edgecolor='red',
                facecolor='none',
                linestyle='dashed',
                alpha=0.4
            )
            ax.add_patch(ell_bin)

            # Set subplot title with range of distances
            ax.set_title(
            f'D: {distance_bins[i]:.0f} - {distance_bins[i+1]:.0f} m\n'
            f'μx, μy: {mean_x_bin:.2f}, {mean_y_bin:.2f}\n'
            f'σ²x, σ²y: {var_x_bin:.2f}, {var_y_bin:.2f}, σ²x/σ²y: {var_x_bin/var_y_bin:.2f}, covar: {covar_xy:.2f}'
            )

            # Labels
            ax.set_xlabel('X Component of Distance')
            if col == 0:  # Only label y-axis on the leftmost column
                ax.set_ylabel('Y Component of Distance')
            
            ax.set_xlim(xlim)  # Limits x-axis from -10 to 10
            ax.set_ylim(ylim)  # Limits y-axis from -10 to 10

            ax.set_aspect('equal', adjustable='datalim')

        # Remove any unused subplots if n_bins is not a perfect square
        for i in range(n_bins, grid_size * grid_size):
            row, col = divmod(i, grid_size)
            fig.delaxes(axes[row, col])  # Remove extra axes

        # Add colorbar
        cbar = fig.colorbar(heatmap_plot, ax=axes, location='right', shrink=0.8)
        cbar.set_label('Number of Data Points')

        # Show plot
        plt.show()

        #### Plot Pixel Values ####

        # Create figure with 3 subplots
        fig, ax = plt.subplots(1, 4, figsize=(18, 6), sharey=True, sharex=True)

        # First subplot: X Values vs Range Error
        ax[0].scatter(x_values, distances, alpha=0.05, color='orange', marker='x')
        ax[0].set_xlabel('X Values')
        ax[0].set_ylabel('Range Error')
        ax[0].set_title('X Values vs Range Error')

        # Second subplot: Y Values vs Range Error
        ax[1].scatter(y_values, distances, alpha=0.05, color='orange', marker='x')
        ax[1].set_xlabel('Y Values')
        ax[1].set_title('Y Values vs Range Error')

        # Third subplot: Range vs Range Error
        ax[2].scatter(np.where(np.abs(x_values) > np.abs(y_values), x_values, y_values), distances, alpha=0.05, color='red', marker='x')
        ax[2].set_xlabel('Range')
        ax[2].set_title('max Pxel Error vs Range Error')

        # Third subplot: Range vs Range Error
        ax[3].scatter(np.where(np.abs(x_values) > np.abs(y_values), x_values, y_values), distances, alpha=0.05, color='red', marker='x')
        ax[3].set_xlabel('Range')
        ax[3].set_title('max Pxel Error vs Range Error')

        # Add horizontal bin edges and annotate means/variances
        for i in range(len(bin_edges)):
            for j in range(4):
                ax[j].axhline(y=bin_edges[i], color='red', linestyle='--', linewidth=1)

        # Get x-limits for text positioning
        xlim0 = ax[0].get_xlim()
        xlim1 = ax[1].get_xlim()
        xlim2 = ax[2].get_xlim()
        xlim3 = ax[3].get_xlim()

        for i in range(len(x_px_bin_centers)):
            if x_px_bin_means[i] is not None:
                ax[0].errorbar(x_px_bin_means[i], x_px_bin_centers[i], xerr=np.sqrt(x_px_bin_vars[i]), fmt='o', color='blue', capsize=5)
                ax[0].text(xlim0[0] + 0.05 * (xlim0[1] - xlim0[0]), x_px_bin_centers[i] - 0.25*np.diff(bin_edges)[0], f"{x_px_bin_means[i]:.2f}",
                        color='black', ha='left', fontsize=8)
                ax[0].text(xlim0[1] - 0.05 * (xlim0[1] - xlim0[0]), x_px_bin_centers[i] - 0.25*np.diff(bin_edges)[0], f"{np.sqrt(x_px_bin_vars[i]):.2f}",
                        color='black', ha='right', fontsize=8)

        for i in range(len(y_px_bin_centers)):
            if y_px_bin_means[i] is not None:
                ax[1].errorbar(y_px_bin_means[i], y_px_bin_centers[i], xerr=np.sqrt(y_px_bin_vars[i]), fmt='o', color='blue', capsize=5)
                ax[1].text(xlim1[0] + 0.05 * (xlim1[1] - xlim1[0]), y_px_bin_centers[i] - 0.25*np.diff(bin_edges)[0], f"{y_px_bin_means[i]:.2f}",
                        color='black', ha='left', fontsize=8)
                ax[1].text(xlim1[1] - 0.05 * (xlim1[1] - xlim1[0]), y_px_bin_centers[i] - 0.25*np.diff(bin_edges)[0], f"{np.sqrt(y_px_bin_vars[i]):.2f}",
                        color='black', ha='right', fontsize=8)

        for i in range(len(xy_px_bin_centers)):
            if bin_means[i] is not None:
                ax[2].errorbar(bin_means[i], xy_px_bin_centers[i], xerr=np.sqrt(bin_vars[i]), fmt='o', color='blue', capsize=5)
                ax[2].text(xlim2[0] + 0.05 * (xlim2[1] - xlim2[0]), xy_px_bin_centers[i] - 0.25*np.diff(bin_edges)[0], f"{bin_means[i]:.2f}",
                        color='black', ha='left', fontsize=8)
                ax[2].text(xlim2[1] - 0.05 * (xlim2[1] - xlim2[0]), xy_px_bin_centers[i] - 0.25*np.diff(bin_edges)[0], f"{np.sqrt(bin_vars[i]):.2f}",
                        color='black', ha='right', fontsize=8)
                
        for i in range(len(xy_px_bin_centers)):
            if bin_means[i] is not None:
                ax[3].errorbar(0, xy_px_bin_centers[i], xerr=np.sqrt(xy_px_bin_total_var[i]), fmt='o', color='blue', capsize=5)
                ax[3].text(xlim2[0] + 0.05 * (xlim2[1] - xlim2[0]), xy_px_bin_centers[i] - 0.25*np.diff(bin_edges)[0], "0",
                        color='black', ha='left', fontsize=8)
                ax[3].text(xlim2[1] - 0.05 * (xlim2[1] - xlim2[0]), xy_px_bin_centers[i] - 0.25*np.diff(bin_edges)[0], f"{np.sqrt(xy_px_bin_total_var[i]):.2f}",
                        color='black', ha='right', fontsize=8)
                
        ax[0].text(xlim0[0] + 0.05 * (xlim0[1] - xlim0[0]), bin_edges[-1] + 0.25*np.diff(bin_edges)[0], "Mean",
                        color='black', ha='left', fontsize=8)
        ax[0].text(xlim0[1] - 0.05 * (xlim0[1] - xlim0[0]), bin_edges[-1] + 0.25*np.diff(bin_edges)[0], "STD",
                    color='black', ha='right', fontsize=8)
        ax[1].text(xlim1[0] + 0.05 * (xlim1[1] - xlim1[0]), bin_edges[-1] + 0.25*np.diff(bin_edges)[0], "Mean",
                        color='black', ha='left', fontsize=8)
        ax[1].text(xlim1[1] - 0.05 * (xlim1[1] - xlim1[0]), bin_edges[-1] + 0.25*np.diff(bin_edges)[0], "STD",
                    color='black', ha='right', fontsize=8)
        ax[2].text(xlim2[0] + 0.05 * (xlim2[1] - xlim2[0]), bin_edges[-1] + 0.25*np.diff(bin_edges)[0], "Mean",
                        color='black', ha='left', fontsize=8)
        ax[2].text(xlim2[1] - 0.05 * (xlim2[1] - xlim2[0]), bin_edges[-1] + 0.25*np.diff(bin_edges)[0], "STD",
                    color='black', ha='right', fontsize=8)
        ax[3].text(xlim3[0] + 0.05 * (xlim3[1] - xlim3[0]), bin_edges[-1] + 0.25*np.diff(bin_edges)[0], "Mean",
                        color='black', ha='left', fontsize=8)
        ax[3].text(xlim3[1] - 0.05 * (xlim3[1] - xlim3[0]), bin_edges[-1] + 0.25*np.diff(bin_edges)[0], "STD",
                    color='black', ha='right', fontsize=8)

        plt.show()

    if pregen_state_data:
        return sample_pixel_uncertainty
    else:
        return sample_pixel_uncertainty, sample_pixel_uncertainty_state_data

def generate_range_uncertainty_model_bins_or_measurement_uncertainty_params(all_detection_summaries, bins = 30, error_trim = 2500, plot_range_uncertainties = False, manual_removal_indices = [], gt_dist_basis = True):
    if all_detection_summaries == None:
        print("No Data Specified!")
        return

    max_range = 0

    ignore_edges = False
    trim_outliers = True
    sigma_prune = 5

    # Iterate through detection_summary and extract uncertainty statistics
    s_values = []
    obj_types = []
    distances = []
    confidences = []
    frame_numbers = []

    for detection_summary in all_detection_summaries:
        for obj, obj_data in detection_summary.items():
            # Extract x and y components from the 'analysis' dictionary
            for frame_id, analysis in sorted(obj_data['analysis'].items()):
                if ignore_edges:
                    if analysis['is_edge']:
                        continue

                s_this = analysis['s']
                confidence_this = analysis['confidence']
                if gt_dist_basis:
                    gt_dist_this = obj_data['ground_truth'][frame_id].range_distance_m
                else:
                    gt_dist_this = analysis['pred_distance']
                max_range = max(max_range, gt_dist_this)

                if trim_outliers and gt_dist_basis:
                    if gt_dist_this < 5000:
                        if abs(s_this) < error_trim:
                            distances.append(gt_dist_this)
                            s_values.append(s_this)
                            obj_types.append(obj)
                            frame_numbers.append(frame_id)
                            confidences.append(confidence_this)
                    else:
                        distances.append(gt_dist_this)
                        s_values.append(s_this)
                        obj_types.append(obj)
                        frame_numbers.append(frame_id)
                        confidences.append(confidence_this)
                elif trim_outliers:
                    if abs(s_this) < 1800:
                        distances.append(gt_dist_this)
                        s_values.append(s_this)
                        obj_types.append(obj)
                        frame_numbers.append(frame_id)
                        confidences.append(confidence_this)
                else:
                    distances.append(gt_dist_this)
                    s_values.append(s_this)
                    obj_types.append(obj)
                    frame_numbers.append(frame_id)
                    confidences.append(confidence_this)

    # Remove None values
    s_values = np.array(s_values, dtype=object)
    distances = np.array(distances, dtype=object)
    obj_types = np.array(obj_types, dtype=object)
    confidences = np.array(confidences, dtype=object)

    #valid_mask = (x_values != None) & (y_values != None) & (distances != None) #& (s_values != None)  # Mask to remove None
    valid_mask = (
        (s_values != None) & (distances != None) & (confidences != None) &
        ~np.isnan(s_values.astype(float)) & ~np.isnan(distances.astype(float))
    )

    # Apply mask
    s_values = s_values[valid_mask].astype(float)
    distances = distances[valid_mask].astype(float)
    obj_types = obj_types[valid_mask]
    confidences = confidences[valid_mask]

    #s_values = confidences

    # Compute bin edges using Jenks natural breaks on the distances
    # Step 1: Construct Z
    D = np.array(distances)
    Z = np.array(s_values)
    C = np.array(confidences)

    # Step 1: Construct uniform base bins along D
    base_k = 150
    base_edges = np.linspace(D.min(), D.max(), base_k + 1)
    bin_indices = np.digitize(D, base_edges) - 1
    bin_indices = np.clip(bin_indices, 0, base_k - 1)

    # Step 2: Group Z values by base bins
    base_bin_values = [Z[bin_indices == i] for i in range(base_k)]
    base_bin_counts = np.array([len(b) for b in base_bin_values])
    base_bin_width = base_edges[1] - base_edges[0]

    # Step 3: Precompute NLL for all merges of consecutive base bins
    nll_matrix = np.full((base_k, base_k), np.inf)
    mean_matrix = np.full((base_k, base_k), np.inf)
    var_matrix = np.full((base_k, base_k), np.inf)
    for i in range(base_k):
        for j in range(i + 1, base_k + 1):
            # Check constraints: min 1000 points and min 150m total width
            point_count = np.sum(base_bin_counts[i:j])
            dist_span = base_edges[j] - base_edges[i]
            if base_edges[i] > 1500:
                if point_count < 900 or dist_span < 200:
                    continue

            # Determine the max number of points in any contributing base bin
            max_points = max(len(base_bin_values[b]) for b in range(i, j) if len(base_bin_values[b]) > 0)

            equalized_bins = []
            for b in range(i, j):
                values = base_bin_values[b]
                if len(values) == 0:
                    continue

                full_repeats = max_points // len(values)
                remainder = max_points % len(values)

                repeated = np.tile(values, full_repeats)  # duplicate entire bin
                if remainder > 0:
                    sampled = np.random.choice(values, size=remainder, replace=True)
                    resampled = np.concatenate((repeated, sampled))
                else:
                    resampled = repeated

                equalized_bins.append(resampled)

            z_segment = np.concatenate(equalized_bins)
            if len(z_segment) < 2:
                continue
            try:
                # Initial stats
                mu = np.mean(z_segment)
                sigma = np.std(z_segment)

                # 5-sigma pruning
                pruned_segment = z_segment[np.abs(z_segment - mu) <= sigma_prune * sigma]
                
                # Recompute stats after pruning
                if len(pruned_segment) < 2:
                    continue
                mu2 = np.mean(pruned_segment)
                sigma2 = np.var(pruned_segment)

                # Avoid log(0)
                if sigma2 > 1e-6:
                    nll = 0.5 * len(pruned_segment) * (np.log(2 * np.pi * sigma2) + 1)
                    nll_matrix[i, j - 1] = nll
                    mean_matrix[i, j - 1] = mu2
                    var_matrix[i, j - 1] = sigma2
            except:
                continue

    # Step 4: Dynamic programming to merge base bins into 'bins' segments
    dp = np.full((bins + 1, base_k + 1), np.inf)
    split = np.full((bins + 1, base_k + 1), -1)
    dp[0, 0] = 0

    for b in range(1, bins + 1):
        for j in range(1, base_k + 1):
            for i in range(j):
                if dp[b - 1, i] + nll_matrix[i, j - 1] < dp[b, j]:
                    dp[b, j] = dp[b - 1, i] + nll_matrix[i, j - 1]
                    split[b, j] = i

    # Backtrack to get best split
    boundaries = []
    j = base_k
    for b in range(bins, 0, -1):
        i = split[b, j]
        boundaries.append(i)
        j = i

    boundaries = boundaries[::-1]  # make them ascending

    if boundaries[0] != 0:
        boundaries = [0] + boundaries
    if boundaries[-1] != base_k:
        boundaries = boundaries + [base_k]

    # If any manually removed bins in function argument, remove here
    if manual_removal_indices:
        boundaries = [x for i, x in enumerate(boundaries) if i not in {len(boundaries) + idx if idx < 0 else idx for idx in manual_removal_indices}]

    bin_edges = [base_edges[i] for i in boundaries]

    bin_means = []
    bin_vars = []

    # Loop through each merged bin segment
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        # Pull the precomputed mean and var for the full segment
        mu = mean_matrix[start, end - 1]
        var = var_matrix[start, end - 1]
        
        bin_means.append(mu)
        bin_vars.append(var)

    intermediate = np.linspace(bin_edges[0], bin_edges[1], 3)[1:-1]
    bin_edges = bin_edges[:1] + list(intermediate) + bin_edges[1:]

    # Manually enforce floor/ceiling
    bin_edges[0] = 0
    bin_edges[-1] = 7000

    bin_edges = [int(x) for x in bin_edges]

    # Digitize: find the bin index for each distance
    bin_indices = np.digitize(distances, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)  # ensure in bounds

    # Generate bin edges
    # Compute means and standard deviations for each range bin
    s_values_binned, s_bin_centers = [], []

    for i in range(len(bin_edges) - 1):
        bin_mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        
        s_bin_data = s_values[bin_mask]
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2

        if len(s_bin_data) > 0:
            # Save Binned Values
            s_values_binned.append(s_bin_data)
            s_bin_centers.append(bin_center)
        else:
            # Save Binned Values
            s_values_binned.append(None)
            s_bin_centers.append(bin_center)

    if not gt_dist_basis:
        gaussian_params_per_bin = {}
        num_bins = len(bin_edges) - 1

        for i in range(num_bins):
            s_bin_data = s_values_binned[i]

            if s_bin_data is not None and len(s_bin_data) >= 2:  # Need at least 2 points to get a variance
                mean_val = np.mean(s_bin_data)
                variance_val = np.var(s_bin_data)
                gaussian_params_per_bin[(bin_edges[i], bin_edges[i+1])] = {'mean': mean_val, 'variance': variance_val}
            else:
                gaussian_params_per_bin[(bin_edges[i], bin_edges[i+1])] = None  # For empty or too-small bins
    
    #### Plot Range Uncertainty Data ####

    if plot_range_uncertainties:
        if not gt_dist_basis:
            num_cols = 5
            num_rows = int(np.ceil(num_bins / num_cols))

            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 3 * num_rows), constrained_layout=True)
            axes = axes.flatten()

            for i in range(num_bins):
                bins_tuple = list(gaussian_params_per_bin.keys())[i]
                ax = axes[i]
                bin_center = s_bin_centers[i]
                s_bin_data = s_values_binned[i]

                if s_bin_data is not None and len(s_bin_data) >= 2:
                    # Plot histogram
                    ax.hist(s_bin_data, bins=30, color='blue', alpha=0.7, edgecolor='black', density=True)

                    # Retrieve stored Gaussian parameters
                    params = gaussian_params_per_bin[bins_tuple]
                    mean_val = params['mean']
                    std_val = np.sqrt(params['variance'])

                    # Generate x-axis range
                    x_min, x_max = np.min(s_bin_data), np.max(s_bin_data)
                    x_plot = np.linspace(x_min, x_max, 300)

                    # Normal PDF
                    pdf = norm.pdf(x_plot, loc=mean_val, scale=std_val)
                    ax.plot(x_plot, pdf, color='red', linewidth=2, label='Normal Fit')
                    ax.legend()

                    ax.set_title(f'Bin {i+1}: Range ~ {bin_edges[i]} to {bin_edges[i+1]} (N={len(s_bin_data)})')
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10)
                    ax.set_title(f'Bin {i+1}: Range ~ {bin_center:.2f} (Empty)')

                ax.set_xlabel('S Values')
                ax.set_ylabel('Density')

            # Hide unused axes
            for j in range(num_bins, len(axes)):
                fig.delaxes(axes[j])

            plt.show()

            num_cols = 5
            num_rows = int(np.ceil(num_bins / num_cols))

            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 3 * num_rows), constrained_layout=True)
            axes = axes.flatten()

            for i in range(num_bins):
                ax = axes[i]
                bin_center = s_bin_centers[i]
                s_bin_data = s_values_binned[i]

                if s_bin_data is not None and len(s_bin_data) >= 2:
                    # Plot histogram
                    ax.hist(s_bin_data, bins=30, color='blue', alpha=0.7, edgecolor='black', density=True)

                    # Retrieve stored Gaussian parameters
                    params = gaussian_params_per_bin[i]
                    mean_val = params['mean']
                    std_val = np.sqrt(params['variance'])

                    # Generate x-axis range
                    x_min, x_max = np.min(s_bin_data), np.max(s_bin_data)
                    x_plot = np.linspace(x_min, x_max, 300)

                    # Normal PDF
                    pdf = norm.pdf(x_plot, loc=mean_val, scale=std_val)
                    ax.plot(x_plot, pdf, color='red', linewidth=2, label='Normal Fit')
                    ax.legend()

                    ax.set_title(f'Bin {i+1}: Range ~ {bin_edges[i]} to {bin_edges[i+1]} (N={len(s_bin_data)})')
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10)
                    ax.set_title(f'Bin {i+1}: Range ~ {bin_center:.2f} (Empty)')

                ax.set_xlabel('S Values')
                ax.set_ylabel('Density')

            # Hide unused axes
            for j in range(num_bins, len(axes)):
                fig.delaxes(axes[j])

            plt.show()

        # --- Recompute starts for flattened z ---
        starts = np.zeros(base_k + 1, dtype=np.int32)
        for i in range(base_k):
            starts[i + 1] = starts[i] + len(base_bin_values[i])

        # --- Flatten z_values and corresponding distances ---
        z_flat_plot = np.concatenate(base_bin_values)

        # Flatten corresponding distances for each z_value
        base_bin_distances = [D[bin_indices == i] for i in range(base_k)]
        d_flat_plot = np.concatenate(base_bin_distances)

        # Flatten corresponding confidences for each c_value
        base_bin_confidences = [C[bin_indices == i] for i in range(base_k)]
        c_flat_plot = np.concatenate(base_bin_confidences)

        #############################################

        # --- Compute variance and mean per base bin ---
        base_bin_variances = [np.var(b) if len(b) > 0 else 0 for b in base_bin_values]
        base_bin_means = [np.mean(b) if len(b) > 0 else 0 for b in base_bin_values]

        # --- Plot ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                            gridspec_kw={'height_ratios': [3, 1, 1]})

        # --- Top plot: Distance vs z_values ---
        ax1.scatter(d_flat_plot, z_flat_plot, color='black', linewidth=0.8, label='z_values', alpha=0.05)
        for edge in bin_edges:
            ax1.axvline(x=edge, color='red', linestyle='--', linewidth=1)
        ax1.set_title("Distance vs z_value with Merged Bin Boundaries")
        ax1.set_xlabel("Distance (m)")
        ax1.set_ylabel("z_value")
        ax1.legend()

        # --- Middle plot: Variance of each base bin ---
        ax2.bar(base_edges[:-1], base_bin_variances, width=0.8 * base_bin_width,
                color='gray', edgecolor='black', align='edge')
        for boundary in bin_edges:  # exclude first and last
            ax2.axvline(x=boundary, color='red', linestyle='--', linewidth=1)
        ax2.set_title("Per-Base-Bin Variance")
        ax2.set_xlabel("Distance (m)")
        ax2.set_ylabel("Variance")

        # --- Bottom plot: Mean of each base bin ---
        ax3.bar(base_edges[:-1], base_bin_means, width=0.8 * base_bin_width,
                color='steelblue', edgecolor='black', align='edge')
        for boundary in bin_edges:
            ax3.axvline(x=boundary, color='red', linestyle='--', linewidth=1)
        ax3.set_title("Per-Base-Bin Mean")
        ax3.set_xlabel("Distance (m)")
        ax3.set_ylabel("Mean z_value")

        plt.tight_layout()
        plt.show()

    if gt_dist_basis:
        return bin_edges
    else:
        return gaussian_params_per_bin

def generate_range_uncertainty_model(range_bins, all_detection_summaries = None, error_trim = 2500, camera_res_xy = AMAZON_CAMERA_RES, camera_fov_xy = AMAZON_CAMERA_FOV, test_and_visualise = False, pregen_state_data = None, pregen_model_params = None):
    if pregen_state_data is None:
        if all_detection_summaries is None:
            print("No Data Specified!")
            return
        
        def stitch_and_interpolate(track_dict, n=1, noise_scale=1.0):
            result = dict(track_dict)
            sorted_frames = sorted(track_dict.keys())
            new_frames = {}

            # First pass: interpolate fully missing frames between gaps ≤ n
            for i in range(len(sorted_frames) - 1):
                f1 = sorted_frames[i]
                f2 = sorted_frames[i + 1]
                d1, s1, t1 = track_dict[f1]
                d2, s2, t2 = track_dict[f2]
                gap = f2 - f1

                if 1 < gap <= n:
                    for j in range(1, gap):
                        f_interp = f1 + j

                        can_interp_s = (s1 is not None and s2 is not None)
                        can_interp_d = (d1 is not None and d2 is not None)
                        can_interp_t = (t1 is not None and t2 is not None)

                        if can_interp_s and can_interp_d and can_interp_t:
                            interp_s = s1 + (s2 - s1) * (j / gap)
                            interp_d = d1 + (d2 - d1) * (j / gap)
                            interp_t = t1 + (t2 - t1) * (j / gap)
                            new_frames[f_interp] = [interp_d, interp_s, interp_t]

            result.update(new_frames)

            # Second pass: fill in None values with added noise
            def interpolate_missing_field(field_index):
                interpolated_segments = {}
                keys = sorted(result.keys())
                i = 0
                while i < len(keys):
                    if result[keys[i]][field_index] is not None:
                        i += 1
                    else:
                        missing_keys = []
                        start_idx = i - 1 if i > 0 else None
                        while i < len(keys) and result[keys[i]][field_index] is None:
                            missing_keys.append(keys[i])
                            i += 1
                        end_idx = i if i < len(keys) else None
                        
                        

                        if start_idx is not None and end_idx is not None:
                            start_frame = keys[start_idx]
                            end_frame = keys[end_idx]
                            gap = end_frame - start_frame
                            if gap > n:
                                continue

                            v_start = result[start_frame][field_index]
                            v_end = result[end_frame][field_index]

                            # Gather neighborhood for autocorr and segment spread
                            neighborhood_keys = [
                                k for k in keys
                                if (
                                    abs(k - keys[start_idx]) <= max(2*gap, n) or abs(k - keys[end_idx]) <= max(2*gap, n)
                                )
                            ]
                            neighborhood_vals = [result[k][field_index] for k in neighborhood_keys if result[k][field_index] is not None]

                            # --- Compute local spread from sliding linear segments (length = gap) within neighborhood if gap > 1 ---
                            segment_std_list = []

                            min_fraction_valid = 0.7

                            # Ensure neighborhood_keys are sorted
                            neighborhood_keys_sorted = sorted(neighborhood_keys)

                            # For plotting
                            segment_lines = []
                            actual_x = [k for k in neighborhood_keys if result[k][field_index] is not None]
                            actual_y = [result[k][field_index] for k in actual_x]

                            residual_vectors = [] 

                            for j in range(len(neighborhood_keys_sorted) - gap):
                                k1 = neighborhood_keys_sorted[j]
                                k2 = neighborhood_keys_sorted[j + gap]
                                v1 = result[k1][field_index]
                                v2 = result[k2][field_index]

                                if None in (v1, v2):
                                    continue

                                segment_keys = neighborhood_keys_sorted[j + 1 : j + gap]
                                segment_vals = [result[k][field_index] for k in segment_keys]

                                # Filter out None values and track their positions
                                valid_pairs = [(k, v) for k, v in zip(segment_keys, segment_vals) if v is not None]

                                if len(valid_pairs) / len(segment_keys) < min_fraction_valid:
                                    continue  # skip this segment if too many missing values

                                # For Plotting
                                # Build full x and y for the segment, including endpoints
                                x_vals_plot = [k1] + [k for k, _ in valid_pairs] + [k2]
                                lin_vals_plot = [v1] + [v1 + (v2 - v1) * ((k - k1) / gap) for k, _ in valid_pairs] + [v2]
                                segment_lines.append((x_vals_plot, lin_vals_plot))

                                # Linear interpolation along the segment for valid points only
                                lin_vals = [v1 + (v2 - v1) * ((k - k1) / gap) for k, _ in valid_pairs]
                                actual_vals = [v for _, v in valid_pairs]
                
                                residuals = np.array(actual_vals) - np.array(lin_vals)
                                segment_std = np.std(residuals)
                                residual_vectors.append(residuals)

                                if segment_std > 0:
                                    segment_std_list.append(segment_std)

                                # # Plot
                                # if segment_lines:
                                #     plt.figure(figsize=(10, 5))

                                #     # Plot all neighborhood segments
                                #     for x_vals_plot, y_vals_plot in segment_lines:
                                #         plt.plot(x_vals_plot, y_vals_plot, linestyle='--', marker='o', label=f'Segment {x_vals_plot[0]}–{x_vals_plot[-1]}')

                                #     # Plot the actual interpolation line (solid red)
                                #     gap_x = [start_frame] + missing_keys + [end_frame]
                                #     gap_y = [v_start] + [v_start + (v_end - v_start) * ((x - start_frame) / gap) for x in missing_keys] + [v_end]
                                #     plt.plot(gap_x, gap_y, color='red', linewidth=2, label='Actual Interpolation')

                                #     # Plot actual data
                                #     plt.scatter(actual_x, actual_y, color='black', label='Actual Data', zorder=10)
                                #     plt.plot(actual_x, actual_y, color='black', label='Actual Data', alpha = 0.3)

                                #     plt.title(f"Neighborhood Segment Fit for Gap Starting at Frame {start_frame}")
                                #     plt.xlabel("Frame")
                                #     plt.ylabel("Value")
                                #     plt.grid(True)
                                #     plt.legend()
                                #     plt.tight_layout()
                                #     plt.show()

                                # Average residual-based spread

                            if gap > 2:
                                if not segment_std_list:
                                    continue

                                weights = [len(vec) for vec in residual_vectors[:len(segment_std_list)]]
                                std = np.average(segment_std_list, weights=weights)
                            else:
                                std = np.std(residual_vectors)
                                
                            # --- Build autocorrelation-based covariance ---
                            auto_cov = np.array([])
                            max_lag = 1
                            if gap > 2:
                                if len(residual_vectors) >= 5:
                                    max_len_residuals = max(len(vec) for vec in residual_vectors)
                                    padded_residuals = [
                                        np.pad(vec, (0, max_len_residuals- len(vec)), constant_values=np.nan)
                                        for vec in residual_vectors
                                    ]
                                    residual_matrix = np.array(padded_residuals)  # shape: (num_segments, gap)
                                    gap_length = residual_matrix.shape[1]

                                    # Compute mean-centered data
                                    residual_matrix -= np.nanmean(residual_matrix, axis=0)

                                    # Compute average autocorrelation at each lag
                                    max_lag = min(4, gap_length - 1)  # limit to available lags
                                    autocorrs = []

                                    for lag in range(max_lag + 1):
                                        values = []
                                        for row in residual_matrix:
                                            if len(row) > lag:
                                                r = np.corrcoef(row[:-lag], row[lag:])[0, 1] if lag > 0 else 1.0
                                                if not np.isnan(r):
                                                    values.append(r)
                                        autocorrs.append(np.mean(values) if values else 0.0)

                                    # Build lag-only covariance matrix
                                    residual_std = np.nanstd(residual_matrix)
                                    auto_cov = toeplitz(autocorrs)

                            # --- Inject interpolated values with autocorrelated noise ---
                            num_missing = len(missing_keys)
                            step_size = max(1, min(max_lag - 1, auto_cov.shape[0]))  # prevent zero
                            for k in range(0, num_missing, step_size):
                                chunk_keys = missing_keys[k : k + step_size]
                                chunk_len = len(chunk_keys)

                                if chunk_len == 0:
                                    continue

                                if auto_cov.shape[0] >= chunk_len:
                                    scaled_cov = auto_cov[:chunk_len, :chunk_len] * (std * noise_scale) ** 2
                                    
                                    noise_vector = np.atleast_1d(
                                            np.random.multivariate_normal(mean=np.zeros(chunk_len), cov=scaled_cov)
                                        )
                                else:
                                    noise_vector = np.random.normal(0, std * noise_scale, size=chunk_len)

                                for l, mk in enumerate(chunk_keys):
                                    frac = (mk - start_frame) / gap
                                    interp_val = v_start + (v_end - v_start) * frac
                                    interpolated_segments[mk] = interp_val + noise_vector[l]
                for mk, val in interpolated_segments.items():
                    result[mk][field_index] = val

            interpolate_missing_field(0)  # distance
            interpolate_missing_field(1)  # s
            interpolate_missing_field(2)  # t

            return result

        def group_consecutive_frames_stitched(stitched_dict):
                items = sorted(stitched_dict.items())  # Sort by frame number
                groups = {}
                group_index = 0

                for frame, data in items:
                    if (data[0] is None) or (data[1] is None) or (data[2] is None) or np.any(np.isnan(data)):
                        continue  # Skip if 's' is None
                    if not groups:
                        groups[group_index] = [(frame, data)]
                    else:
                        last_frame = groups[group_index][-1][0]
                        if frame == last_frame + 1:
                            groups[group_index].append((frame, data))
                        else:
                            group_index += 1
                            groups[group_index] = [(frame, data)]
                return groups

        def plot_interpolated_vs_raw(raw, stitched, title='Interpolated vs Raw Track Data'):
            stitched_groups = group_consecutive_frames_stitched(stitched)

            plt.figure(figsize=(12, 6))
            color_cycle = itertools.cycle(plt.cm.tab10.colors)

            # Plot raw data, colored by stitched groups + line connecting all values
            for group_id, frame_data in stitched_groups.items():
                color = next(color_cycle)

                dist_x_raw, dist_y_raw = [], []
                s_x_raw, s_y_raw = [], []

                dist_x_all, dist_y_all = [], []
                s_x_all, s_y_all = [], []
            
                for frame, (dist, s, t) in frame_data:
                    # Collect all values (raw or interpolated)
                    if dist is not None:
                        dist_x_all.append(frame)
                        dist_y_all.append(dist)
                    if s is not None:
                        s_x_all.append(frame)
                        s_y_all.append(s)

                    # Collect only raw values
                    if frame in raw:
                        d, s, t = raw[frame]
                        if d is not None:
                            dist_x_raw.append(frame)
                            dist_y_raw.append(d)
                        if s is not None:
                            s_x_raw.append(frame)
                            s_y_raw.append(s)

                # Line through all values
                plt.plot(dist_x_all, dist_y_all, color=color, alpha=0.6, linewidth=1.5)
                plt.plot(s_x_all, s_y_all, color=color, alpha=0.6, linewidth=1.5)

                # Scatter raw points
                plt.scatter(dist_x_raw, dist_y_raw, color=color, label=f'Raw distance (group {group_id})')
                plt.scatter(s_x_raw, s_y_raw, color=color, marker='s', label=f'Raw s (group {group_id})')

            # Overlay interpolated points
            interp_dist_x = []
            interp_dist_y = []
            interp_s_x = []
            interp_s_y = []

            for frame, (dist, s, t) in stitched.items():
                is_raw = frame in raw
                raw_dist = raw.get(frame, [None, None])[0]
                raw_s = raw.get(frame, [None, None])[1]

                if not is_raw or raw_dist is None:
                    if dist is not None:
                        interp_dist_x.append(frame)
                        interp_dist_y.append(dist)

                if not is_raw or raw_s is None:
                    if s is not None:
                        interp_s_x.append(frame)
                        interp_s_y.append(s)

            plt.scatter(interp_dist_x, interp_dist_y, color='cyan', label='Interpolated distance', marker='x')
            plt.scatter(interp_s_x, interp_s_y, color='red', label='Interpolated s', marker='x')

            plt.xlabel("Frame")
            plt.ylabel("Value")
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Collect x, y, and s values
        s_values, gt_distances = [], []
        idx = -1

        # Binning data
        bins = range_bins
        means = {}
        variances = {}
        variances_abs = {}
        datapoint_counts = {}
        continuous_track_segments = []

        for detection_summary in all_detection_summaries:
            for _, obj in detection_summary.items():
                if len(obj['analysis'].values()) > 0 and np.any(~np.isnan([x['s'] for x in obj['analysis'].values()])):
                    idx += 1
                    means[idx] = [np.nan] * (len(bins) - 1)  # Initialize with NaNs
                    variances[idx] = [np.nan] * (len(bins) - 1)  # Initialize with NaNs
                    variances_abs[idx] = [np.nan] * (len(bins) - 1)  # Initialize with NaNs
                    datapoint_counts[idx] = [np.nan] * (len(bins) - 1)
                else:
                    continue

                trim_outliers = True
                ignore_edges = False

                if ignore_edges:
                    raw_track_groups = {
                        frame: [
                            float(obj['ground_truth'][frame].range_distance_m) if (obj['ground_truth'][frame] is not None) and (not obj['analysis'][frame]['is_edge']) else None,
                            obj['analysis'].get(frame, {}).get('s'),
                            obj['analysis'].get(frame, {}).get('timestamp')
                        ]
                        for frame in obj['ground_truth'].keys() & obj['analysis'].keys()
                    }
                else:
                    raw_track_groups = {
                        frame: [
                            float(obj['ground_truth'][frame].range_distance_m) if (obj['ground_truth'][frame] is not None) else None,
                            obj['analysis'].get(frame, {}).get('s'),
                            obj['analysis'].get(frame, {}).get('timestamp')
                        ]
                        for frame in obj['ground_truth'].keys()
                    }

                stitched_track = stitch_and_interpolate(deepcopy(raw_track_groups), n=20)
                
                #plot_interpolated_vs_raw(raw_track_groups, stitched_track)

                stitched_track_groups = group_consecutive_frames_stitched(stitched_track)

                # BUILD DATAFRAME OF STITCHED DATA TO TRAIN GAUSSIAN ON

                for _, frame_group in stitched_track_groups.items():
                    if len(frame_group) < 10:
                        continue
                    s_values_this = []
                    t_values_this = []
                    gt_distances_this = []
                    for frame, analysis in frame_group:
                        gt_dist_this = analysis[0]
                        s_this = analysis[1]
                        t_this = analysis[2]
                        if trim_outliers:
                            if gt_dist_this < 5000:
                                if abs(s_this) < error_trim:
                                    gt_distances_this.append(gt_dist_this)
                                    s_values_this.append(s_this)
                                    t_values_this.append(t_this)
                            else:
                                gt_distances_this.append(gt_dist_this)
                                s_values_this.append(s_this)
                                t_values_this.append(t_this)
                        else:
                            gt_distances_this.append(gt_dist_this)
                            s_values_this.append(s_this)
                            t_values_this.append(t_this)
                    
                    if trim_outliers:
                        if len(s_values_this) < 10:
                            continue

                    smoothed = np.array(s_values_this)

                    continuous_track_segments.append([gt_distances_this, smoothed, s_values_this, t_values_this])

        # Configurable parameters
        history_first = [1, 2, 3]
        history_second = [1]
        history_s = [1, 2, 4]
        history_s_rel = [10]#[10,20,30,40,50,60,70,80]
        history_dist = [0, 4]#[5, 100]#[100,150,200,250,300]

        # Output format of input data structure
        input_space_format = {
            'history_first_deriv_error': history_first,
            'history_second_deriv_error': history_second,
            'history_error': history_s,
            'history_error_rel': history_s_rel,
            'history_gt_dist': history_dist,
        }

        # Flatten all distances into a single array
        all_distances = np.concatenate([track[0] for track in continuous_track_segments])
        bin_edges = range_bins
        num_bins = len(bin_edges) - 1
        
        # Digitize: find the bin index for each distance
        bin_indices = np.digitize(all_distances, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # ensure in bounds

        # Count samples in each bin
        bin_counts = np.bincount(bin_indices, minlength=num_bins)

        # Initialize binned containers
        joint_models = {}
        binned_joint = {i: [] for i in range(num_bins)}

        # --- Accumulate features from all tracks ---
        for track_index, data_this in enumerate(continuous_track_segments):
            gt_distances_this = data_this[0]
            smoothed = data_this[1]
            first_deriv = [np.nan] + list(np.diff(smoothed))
            second_deriv =  [np.nan, np.nan] + list(np.diff(first_deriv))

            for i in range(max(history_first + history_second + history_s + history_dist + history_s_rel), len(second_deriv) - 2):
                dist = gt_distances_this[i]
                bin_idx = np.digitize(dist, bin_edges) - 1
                bin_idx = np.clip(bin_idx, 0, num_bins - 1)

                d1_hist = [first_deriv[i - j] if i - j >= 0 else np.nan for j in history_first]
                d2_hist = [second_deriv[i - j] if i - j >= 0 else np.nan for j in history_second]
                s_abs = [smoothed[i - j] if i - j >= 0 else np.nan for j in history_s]
                s_rel = [smoothed[i - 1] - smoothed[i - 1 - j] if i - 1 - j >= 0 else np.nan for j in history_s_rel]
                dist_hist = [gt_distances_this[i - j] if i - j >= 0 else np.nan for j in history_dist]

                d0_target = smoothed[i]
                d1_target = first_deriv[i]
                d2_target = second_deriv[i]

                features = d1_hist + d2_hist + s_abs + s_rel + dist_hist + [d0_target, d1_target, d2_target]
                binned_joint[bin_idx].append(features)

                
                # print(i)
                # print(gt_distances_this[:i+1])
                # print(dist)
                # print(dist_hist)
                # print('d1')
                # print(first_deriv[:i+1])
                # print(d1_hist)
                # print('d2')
                # print(second_deriv[:i+1])
                # print(d2_hist)
                # print('s')
                # print(smoothed[:i+1])
                # print(s_abs)
                # print('srel')
                # print(s_rel)
                
                #cow
                

            # --- Fit multivariate Gaussians for each bin ---


        model_type = 2 #0 = n-dimensional gaussian, 1=t-dist, 2=gmm, 3=gmm with imputing, 4=gmm with split, 5=tunes gmm

        #TODO: HANDLE THE NAN VALS THAT ARE CURRENTLY CULLING MY DATA

        for i in range(num_bins):
            if len(binned_joint[i]) > 0:
                mat = np.vstack(binned_joint[i])
                
                # Step 1: Compute feature-wise mean and std, ignoring NaNs
                mean = np.nanmean(mat, axis=0)
                std = np.nanstd(mat, axis=0)
                std[std == 0] = 1  # avoid division by zero

                # Step 2: Standardize the data (ignoring NaNs)
                z_scores = (mat - mean) / std

                # Step 3: Find rows with any feature > x std from mean
                too_far = np.abs(z_scores) > 9
                invalid_rows = np.any(too_far, axis=1)

                # Step 4: Remove rows that are either NaN or > 5 std
                valid_rows = ~np.isnan(mat).any(axis=1) & ~invalid_rows
                mat_clean= mat[valid_rows]

                plot_bins = False

                if plot_bins:

                    # Plot histogram for each variable
                    n_vars = mat_clean.shape[1]
                    n_cols = 4
                    n_rows = int(np.ceil(n_vars / n_cols))

                    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 2 * n_rows))
                    axs = axs.flatten()

                    x_vals = np.linspace(np.min(mat_clean), np.max(mat_clean), 500).reshape(-1, 1)

                    for j in range(n_vars):
                        x = mat_clean[:, j]
                        ax = axs[j]

                        if j in [15]:
                            x_nonan = x[~np.isnan(x)].reshape(-1, 1)
                            gmm = GaussianMixture(n_components=2).fit(x_nonan)
                            labels = gmm.predict(x_nonan)

                            group1 = x_nonan[labels == 0].ravel()
                            group2 = x_nonan[labels == 1].ravel()

                            ax.hist(group1, bins=40, color='skyblue', alpha=0.6, density=True, label='Component 1')
                            ax.hist(group2, bins=40, color='salmon', alpha=0.6, density=True, label='Component 2')

                            ax.set_title(f'Variable {i} (GMM Split)')
                            ax.set_xlabel('Value')
                            ax.set_ylabel('Density')
                            ax.legend()
                        else:
                            ax.hist(x, bins=40, color='lightgray', alpha=0.6, density=True, label='Histogram')

                            ax.set_title(f'Variable {j}')
                            ax.set_xlabel('Value')
                            ax.set_ylabel('Density')
                            ax.legend()

                    # Remove unused subplots
                    for l in range(n_vars, len(axs)):
                        fig.delaxes(axs[l])

                    plt.tight_layout()
                    plt.show()

                if len(mat_clean) > 0:
                    # Standardize data
                    scaler = StandardScaler()
                    mat_clean_std = scaler.fit_transform(mat_clean)  # shape (n_samples, n_features)

                    # Fit a 3-component GMM to standardized data
                    gmm = GaussianMixture(n_components=2, covariance_type='full')
                    gmm.fit(mat_clean_std)

                    # Store in joint_models with standardization parameters
                    joint_models[i] = {
                        #'gmm': gmm,
                        'weights': gmm.weights_,             # shape (3,)
                        'means': gmm.means_,                # shape (3, n_features)
                        'covariances': gmm.covariances_,    # shape (3, n_features, n_features)
                        'scaler': {
                            'mean': scaler.mean_,           # shape (n_features,)
                            'scale': scaler.scale_          # shape (n_features,)
                        }
                    }
        model_params = joint_models
    else:
        input_space_format = pregen_state_data
        model_params = pregen_model_params

    # Create input-output function for this model    
    def uncertainty_model_function(input_history = None, new_fov = None, new_res = None, input_space_format = input_space_format, joint_models = model_params):
        assert input_history is not None, "No range error model input provided!"
        assert input_space_format is not None, "No model input format provided!"
        assert (new_fov is not None) and (new_res is not None), "Camera FOV and Resolution not specified!"

        history_first = input_space_format['history_first_deriv_error']
        history_second = input_space_format['history_second_deriv_error']
        history_s = input_space_format['history_error']
        history_s_rel = input_space_format['history_error_rel']
        history_dist = input_space_format['history_gt_dist']

        gt_distances_this_input, generated_s_input, ds_history_input, d2_history_input = input_history

        # Scale inputs down to model scale
        gt_distances_this_input_scaled = rescale_distances_inverse(
            gt_distances_this_input,
            new_fov,
            new_res,
            fov_old_deg = AMAZON_CAMERA_FOV[1],
            res_old_px = AMAZON_CAMERA_RES[1]
        )
        
        gt_dist_scaled_last = gt_distances_this_input_scaled[-1]

        generated_s_input_scaled = rescale_distances_inverse(
            generated_s_input,
            new_fov,
            new_res,
            fov_old_deg = AMAZON_CAMERA_FOV[1],
            res_old_px = AMAZON_CAMERA_RES[1]
        )

        ds_history_input_scaled = rescale_distances_inverse(
            ds_history_input,
            new_fov,
            new_res,
            fov_old_deg = AMAZON_CAMERA_FOV[1],
            res_old_px = AMAZON_CAMERA_RES[1]
        )

        d2_history_input_scaled = rescale_distances_inverse(
            d2_history_input,
            new_fov,
            new_res,
            fov_old_deg = AMAZON_CAMERA_FOV[1],
            res_old_px = AMAZON_CAMERA_RES[1]
        )

        # Construct full joint history input with possible NaNs
        x_d1 = [ds_history_input_scaled[-1-j] if j <= len(ds_history_input_scaled) - 1 else np.nan for j in history_first]
        x_d2 = [d2_history_input_scaled[-1-j] if j <= len(d2_history_input_scaled) - 1 else np.nan for j in history_second]
        x_s = [generated_s_input_scaled[-1-j] if j <= len(generated_s_input_scaled) - 1 else np.nan for j in history_s]
        x_s_rel = [generated_s_input_scaled[-2] - generated_s_input_scaled[-2-j] if len(generated_s_input_scaled) - 2 - j >= 0 else np.nan for j in history_s_rel]
        x_dist = [gt_distances_this_input_scaled[-1-j]  if j <= len(gt_distances_this_input_scaled) - 1 else np.nan for j in history_dist]

        # print(gt_distances_this_input_scaled)
        # print(x_dist)
        # print('d1')
        # print(ds_history_input_scaled)
        # print(x_d1)
        # print('d2')
        # print(d2_history_input_scaled)
        # print(x_d2)
        # print('s')
        # print(generated_s_input_scaled)
        # print(x_s)
        # print('srel')
        # print(x_s_rel)
        
        # if len(gt_distances_this_input_scaled) > 5:
        #     cow
        #print(x_d1 + x_d2 + x_s + x_s_rel + x_dist)
        #print(x_d1, x_d2, x_s, x_s_rel, x_dist)
        x_full = np.array(x_d1 + x_d2 + x_s + x_s_rel + x_dist)

        # Mask to select valid dimensions
        valid_mask = ~np.isnan(x_full)
        x_valid = x_full[valid_mask]

        full_dim = len(x_full)
        total_dim = full_dim + 3  # includes targets
        
        # Indices of known inputs
        input_indices = np.arange(full_dim)[valid_mask]
        target_indices = np.arange(full_dim, total_dim)  # last 3 dims are targets

        # Unpack model
        bin_idx = np.digitize(gt_dist_scaled_last, range_bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(range_bins) - 2)
        model = joint_models[bin_idx]
        gmm_weights = model['weights']
        gmm_means = model['means']
        gmm_covs = model['covariances']
        scaler_mean = model['scaler']['mean']
        scaler_scale = model['scaler']['scale']

        # Standardize x_valid
        x_valid_std = (x_valid - scaler_mean[input_indices]) / scaler_scale[input_indices]

        # Store per-component conditional distributions
        weights = []
        cond_means = []
        cond_covs = []

        for k in range(len(gmm_weights)):
            mu_k = gmm_means[k]
            cov_k = gmm_covs[k]

            mu_x = mu_k[input_indices]
            mu_y = mu_k[target_indices]

            Sigma_xx = cov_k[np.ix_(input_indices, input_indices)]
            Sigma_xy = cov_k[np.ix_(input_indices, target_indices)]
            Sigma_yx = Sigma_xy.T
            Sigma_yy = cov_k[np.ix_(target_indices, target_indices)]

            # Conditional mean and covariance
            inv_Sigma_xx = np.linalg.pinv(Sigma_xx)
            delta = x_valid_std - mu_x
            mu_cond = mu_y + Sigma_yx @ inv_Sigma_xx @ delta
            cov_cond = Sigma_yy - Sigma_yx @ inv_Sigma_xx @ Sigma_xy
            cov_cond = np.maximum(cov_cond, 1e-8)

            # Store conditional stats
            cond_means.append(mu_cond)
            cond_covs.append(cov_cond)

            # Compute unnormalized responsibility weight
            likelihood = multivariate_normal.pdf(x_valid_std, mean=mu_x, cov=Sigma_xx)
            weights.append(gmm_weights[k] * likelihood)

        # Normalize weights to get posterior over components
        weights = np.array(weights)
        total_weight = np.sum(weights)

        if total_weight == 0 or not np.isfinite(total_weight):
            print("Warning: All component likelihoods are zero or invalid. Falling back to uniform weights.")
            print(len(gt_distances_this_input))
            weights = np.ones(len(gmm_weights)) / len(gmm_weights)
        else:
            weights /= total_weight

        # Sample a component index
        k_sampled = np.random.choice(len(weights), p=weights)

        sample_means = cond_means[k_sampled]
        sample_covs = cond_covs[k_sampled]

        sample_std = np.random.multivariate_normal(mean=sample_means, cov=sample_covs)

        # Unstandardize the result
        mean_targets = scaler_mean[target_indices]
        scale_targets = scaler_scale[target_indices]
        sample_unstd = sample_std * scale_targets + mean_targets

        # Extract individual target values
        d_next, ds_next, d2_next = sample_unstd

        # Scale model outputs to real scale
        d_next = rescale_distances(
            d_next,
            new_fov,
            new_res,
            fov_old_deg = AMAZON_CAMERA_FOV[1],
            res_old_px = AMAZON_CAMERA_RES[1]
        )

        ds_next = rescale_distances(
            ds_next,
            new_fov,
            new_res,
            fov_old_deg = AMAZON_CAMERA_FOV[1],
            res_old_px = AMAZON_CAMERA_RES[1]
        )

        d2_next = rescale_distances(
            d2_next,
            new_fov,
            new_res,
            fov_old_deg = AMAZON_CAMERA_FOV[1],
            res_old_px = AMAZON_CAMERA_RES[1]
        )

        # Ensure predicated range is over 30m (tunable)
        gt_dist = gt_distances_this_input[-1]

        if gt_dist + d_next < 0:
            d_next = -gt_dist + 30

        return d_next, ds_next, d2_next

    # # Plotting
    # n_features = mat_clean_std.shape[1]
    # fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 15))
    # axes = axes.flatten()

    # x = np.linspace(-5, 5, 1000)
    # for i in range(n_features):
    #     ax = axes[i]
    #     data = mat_clean_std[:, i]
    #     ax.hist(data, bins=30, density=True, alpha=0.5, color='gray')

    #     for k in range(2):
    #         mean = gmm.means_[k, i]
    #         std = np.sqrt(gmm.covariances_[k][i, i])
    #         y = gmm.weights_[k] * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std**2))
    #         ax.plot(x, y, label=f'Component {k+1}')
        
    #     ax.set_title(f'Feature {i+1}')
    #     ax.legend()

    # # Hide any unused subplots
    # for j in range(n_features, len(axes)):
    #     fig.delaxes(axes[j])

    # fig.tight_layout()
    # plt.show()

    # fig2, ax_1d = plt.subplots(figsize=(6, 4))
    # feature_16 = mat_clean_std[:, 15].reshape(-1, 1)
    # gmm_1d = GaussianMixture(n_components=2, covariance_type='full')
    # gmm_1d.fit(feature_16)

    # ax_1d.hist(feature_16, bins=30, density=True, alpha=0.5, color='gray')
    # x_1d = np.linspace(-5, 5, 1000).reshape(-1, 1)
    # for k in range(2):
    #     mean = gmm_1d.means_[k, 0]
    #     std = np.sqrt(gmm_1d.covariances_[k][0, 0])
    #     y = gmm_1d.weights_[k] * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x_1d - mean)**2 / (2 * std**2))
    #     ax_1d.plot(x_1d, y, label=f'Component {k+1}')

    # ax_1d.set_title('Feature 15 (Isolated GMM)')
    # ax_1d.legend()

    # plt.tight_layout()
    # plt.show()
    
    if test_and_visualise:

        generated_tracks = []
        existing_bins_this = {x:[] for x in range(num_bins)}
        for data_this in continuous_track_segments:
            gt_distances_this = rescale_distances(
                        data_this[0],
                        camera_fov_xy[1],
                        camera_res_xy[1],
                        fov_old_deg = AMAZON_CAMERA_FOV[1],
                        res_old_px = AMAZON_CAMERA_RES[1]
                    )
            
            smoothed = rescale_distances(
                        data_this[1],
                        camera_fov_xy[1],
                        camera_res_xy[1],
                        fov_old_deg = AMAZON_CAMERA_FOV[1],
                        res_old_px = AMAZON_CAMERA_RES[1]
                    )
            plot = False
            if len(smoothed) < 5:
                continue
            if plot:
                # Create side-by-side subplots for original and generated curves with color mapping
                fig, axs = plt.subplots(
                    5, 2,
                    figsize=(16, 12),
                    sharey='row',  # this shares Y axis only within each row
                    sharex='row',
                    gridspec_kw={'hspace': 0.3, 'wspace': 0.2}
                )
                # fig2, axs2 = plt.subplots(2, 2, figsize=(14, 4), sharey=True)
            
            generated_s_multi = []
            for _ in range(1):
                random_init = True
                if not random_init:
                    first_deriv = np.diff(smoothed)
                    second_deriv = np.diff(first_deriv)

                    first_s = smoothed[0]
                    first_ds = first_deriv[0]
                    first_d2 = second_deriv[0]
                else:
                    #first_s, first_ds, first_d2 = uncertainty_model_function(gt_dist_init = gt_distances_this[0], new_fov = camera_fov_xy[1], new_res = camera_res_xy[1])
                    first_s, first_ds, first_d2 = uncertainty_model_function(input_history = [[np.nan, gt_distances_this[0]], [np.nan], [np.nan], [np.nan]], new_fov = camera_fov_xy[1], new_res = camera_res_xy[1])

                # Integration loop using joint model
                generated_s = [first_s]
                ds_history = [first_ds]
                d2_history = [first_d2]

                for i in range(1, len(gt_distances_this) - 1):
                    # Add nan to make the other input arrays (which only go up to the n-1th frame) the same length as the gt distance array (which goes up to the nth frame)
                    generated_s_input = generated_s + [np.nan]
                    ds_history_input = ds_history + [np.nan]
                    d2_history_input = d2_history + [np.nan]
                    s_next, ds_next, d2_next = uncertainty_model_function(input_history = [list(gt_distances_this[:i+1]), generated_s_input, ds_history_input, d2_history_input], new_fov = camera_fov_xy[1], new_res = camera_res_xy[1])

                    generated_s.append(s_next)

                    # if len(generated_s) >= 3:
                    #     ds_history.append(generated_s[-1] - generated_s[-2])
                    #     d2_history.append(generated_s[-1] - 2 * generated_s[-2] + generated_s[-3])
                    # else:
                    #     ds_history.append(ds_next)
                    #     d2_history.append(d2_next)

                    ds_history.append(ds_next)
                    d2_history.append(d2_next)

                    if len(ds_history) > max(history_first):
                        ds_history.pop(0)
                    if len(d2_history) > max(history_second):
                        d2_history.pop(0)

                    #print([generated_s, ds_history, d2_history])
                        
                
                generated_s_multi.append(generated_s)
                generated_tracks.append([gt_distances_this[:len(generated_s)], generated_s])

                # if plot:
                #     # -----------------------------
                #     # Step 5: Plot with coolwarm coloring by distance
                #     # -----------------------------
                #     indices = np.arange(len(generated_s))
                #     distances = gt_distances_this[:len(generated_s)]

                #     norm = Normalize(vmin=min(distances), vmax=max(distances))
                #     colors = cm.coolwarm(norm(distances))

                #     # Left: Original smoothed curve colored by distance
                #     for i in range(len(smoothed) - 2):
                #         axs[0,0].plot([i, i + 1], [smoothed[i], smoothed[i + 1]], color=colors[i])

                #     axs[0,0].set_title("Original Smoothed Curve")
                #     axs[0,0].set_xlabel("Index")
                #     axs[0,0].set_ylabel("s")
                #     # axs[0,0].set_ylim(-1000, 1000)
                #     axs[0,0].grid(True)

                #     # Right: Generated curve with same color mapping
                #     for i in range(len(generated_s) - 1):
                #         axs[0,1].plot(indices[i:i+2], generated_s[i:i+2], color=colors[i])

                #     axs[0,1].set_title("Stochastic Generated Curve")
                #     axs[0,1].set_xlabel("Index")
                #     axs[0,1].grid(True)


            if plot:
                # Shared colorbar for distance
                sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), orientation='vertical', shrink=0.9)
                cbar.set_label('Distance (m)')

                ####### HISTOGRAMS #######
                # Define number of top bins to show
                num_bins_plot = 2
                ordered_bins = {}

                # Step 1: Count how many values fall into each bin
                for bin_idx_plt in range(num_bins):
                    bin_start = bin_edges[bin_idx_plt]
                    bin_end = bin_edges[bin_idx_plt + 1]

                    distances1 = gt_distances_this[:len(smoothed)]
                    mask1 = (np.array(distances1) >= bin_start) & (np.array(distances1) < bin_end)
                    ordered_bins[bin_idx_plt] = mask1.sum()

                # Step 2: Sort bins by descending count
                sorted_keys = sorted(ordered_bins, key=ordered_bins.get, reverse=True)
                top_bins = sorted([k for k, v in ordered_bins.items() if v > 0], key=ordered_bins.get, reverse=True)[:num_bins_plot]

                axs_all = [[axs[1 + m * 2, 0], axs[1 + m * 2, 1], axs[1 + m * 2 + 1, 0], axs[1 + m * 2 + 1, 1]] for m in range(len(top_bins))]
                # Step 4: Plot each bin
                for bin_plot_idx, bin_idx_plt in enumerate(top_bins):
                    bin_start = bin_edges[bin_idx_plt]
                    bin_end = bin_edges[bin_idx_plt + 1]

                    distances1 = gt_distances_this[:len(smoothed)]
                    distances2 = gt_distances_this[:len(generated_s)]

                    mask1 = (np.array(distances1) >= bin_start) & (np.array(distances1) < bin_end)
                    mask2 = (np.array(distances2) >= bin_start) & (np.array(distances2) < bin_end)

                    smoothed_bin = smoothed[mask1]
                    generated_bin = np.array(generated_s)[mask2]

                    if len(smoothed_bin) >= 3 and len(generated_s_multi) > 0:
                        # Compute first and second difference (gradient and curvature) for original
                        grad_deltas_original = np.diff(smoothed_bin, n=1)[1:] - np.diff(smoothed_bin, n=1)[:-1]
                        curv_deltas_original = np.diff(smoothed_bin, n=2)[1:] - np.diff(smoothed_bin, n=2)[:-1]

                        # Prepare containers
                        grad_deltas_generated_all = []
                        curv_deltas_generated_all = []

                        for generated_bin in generated_s_multi:
                            if len(generated_bin) < 3:
                                continue
                            grad_deltas_generated_all.append(np.diff(generated_bin, n=1)[1:] - np.diff(generated_bin, n=1)[:-1])
                            curv_deltas_generated_all.append(np.diff(generated_bin, n=2)[1:] - np.diff(generated_bin, n=2)[:-1])

                        def plot_hist_with_fit_multi(ax, all_data, color, title, xlabel, bin_edges=None):
                            all_flat = np.concatenate(all_data)
                            if bin_edges is None:
                                bin_edges = np.histogram_bin_edges(all_flat, bins=50)

                            for idx, data in enumerate(all_data):
                                mu_norm, std_norm = norm.fit(data)
                                x = np.linspace(min(data), max(data), 500)
                                pdf_norm = norm.pdf(x, mu_norm, std_norm)

                                df_t, loc_t, scale_t = t_dist.fit(data)
                                pdf_t = t_dist.pdf(x, df_t, loc_t, scale_t)

                                if idx == 0:
                                    ax.hist(data, bins=bin_edges, alpha=0.2, color=color, density=True)
                                ax.plot(x, pdf_norm, linestyle='--', linewidth=1.5, label=f'N fit {idx+1}')
                                ax.plot(x, pdf_t, linestyle='-', linewidth=1.5, label=f'T fit {idx+1} (df={df_t:.1f})')

                            ax.set_title(title)
                            ax.set_xlabel(xlabel)
                            ax.set_ylabel('Probability Density')
                            ax.grid(True)
                            ax.legend(framealpha=0.0)

                        # Get axes
                        ax_grad_orig, ax_grad_gen, ax_curv_orig, ax_curv_gen = axs_all[bin_plot_idx]

                        # Add bin info to original gradient plot
                        ax_grad_orig.text(0.5, 1.0, f"Bin {bin_idx_plt} ({bin_start}–{bin_end}m)", 
                                        fontsize=12, ha='center', transform=ax_grad_orig.transAxes)

                        # Shared bin edges
                        shared_bin_edges_grad = np.histogram_bin_edges(
                            np.concatenate([grad_deltas_original] + grad_deltas_generated_all), bins=50)
                        shared_bin_edges_curv = np.histogram_bin_edges(
                            np.concatenate([curv_deltas_original] + curv_deltas_generated_all), bins=50)

                        # Plot original once
                        plot_hist_with_fit_multi(ax_grad_orig, [grad_deltas_original], 'blue', '', 'Gradient Delta', shared_bin_edges_grad)
                        plot_hist_with_fit_multi(ax_curv_orig, [curv_deltas_original], 'blue', '', 'Curvature Delta', shared_bin_edges_curv)

                        # Plot all generated
                        plot_hist_with_fit_multi(ax_grad_gen, grad_deltas_generated_all, 'salmon', '', 'Gradient Delta', shared_bin_edges_grad)
                        plot_hist_with_fit_multi(ax_curv_gen, curv_deltas_generated_all, 'salmon', '', 'Curvature Delta', shared_bin_edges_curv)

                plt.tight_layout()
                plt.show()

        # Create the figure and subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        axs[0].set_title("Original Smoothed Tracks")
        axs[1].set_title("Generated Tracks")

        axs[0].set_xlabel("Distance")
        axs[0].set_ylabel("Smoothed Value")
        axs[1].set_xlabel("Distance")

        # Plot original smoothed data
        for data_this in continuous_track_segments:
            gt_distances_this, smoothed = data_this[0], data_this[1]
            axs[0].plot(gt_distances_this[:len(smoothed)], smoothed, alpha=0.6)

        # Plot generated tracks
        for dist_array, gen_array in generated_tracks:
            axs[1].plot(dist_array, gen_array, alpha=0.6)

        plt.tight_layout()
        plt.show()

    if pregen_state_data is None:
        return uncertainty_model_function, model_params, input_space_format
    else:
        return uncertainty_model_function

if __name__ == "__main__":
    # Define Camera Parameters
    scaling = False

    if scaling:
        new_camera_res = (9344, 7000)
        camera_fovs = [(61.67, 48.26)]#, [68.40, 53.96], [76.40, 61.04], [61.67, 48.26], [68.40, 53.96], [76.40, 61.04]]
    else:
        new_camera_res = tuple(AMAZON_CAMERA_RES)
        camera_fovs = [tuple(AMAZON_CAMERA_FOV)]

    SUM_ON = True
    regenerate_SUM = False
    save_SUM = False
    script_dir = "."
    if SUM_ON:
        SUM_FOLDER = script_dir + "/SUM"
        os.makedirs(SUM_FOLDER, exist_ok=True)

        def save_pickle(obj, filename):
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)

        def load_pickle(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)

        def save_numpy(obj, filename):
            np.save(filename, obj)

        def load_numpy(filename):
            return np.load(filename, allow_pickle=True)

        # Check if saved models exist
        pixel_model_state_path = os.path.join(SUM_FOLDER, "pixel_uncertainty_model.pkl")
        range_bins_path = os.path.join(SUM_FOLDER, "range_bins.npy")
        range_model_input_format_path = os.path.join(SUM_FOLDER, "range_error_model_input_format.pkl")
        range_model_params_path = os.path.join(SUM_FOLDER, "range_model_params.pkl")
        detected_range_model_params_path = os.path.join(SUM_FOLDER, "detected_range_model_params.pkl")

        if regenerate_SUM or not (
            os.path.exists(pixel_model_state_path) and
            os.path.exists(range_bins_path) and
            os.path.exists(range_model_input_format_path) and
            os.path.exists(range_model_params_path) and
            os.path.exists(detected_range_model_params_path)
        ):
            print("Generating sensor uncertainty models from data...")

             # Load data
            data = import_encounter_data()

            # Generate models
            detected_range_uncertainty_model_params = generate_range_uncertainty_model_bins_or_measurement_uncertainty_params(data, bins=30, gt_dist_basis = False)
            pixel_uncertainty_model, pixel_uncertainty_model_state_data = generate_pixel_uncertainty_model(all_detection_summaries = data, bins=20)
            range_bins = generate_range_uncertainty_model_bins_or_measurement_uncertainty_params(data, bins=30, error_trim=2500)
            range_uncertainty_model, range_model_params, range_error_model_input_format = generate_range_uncertainty_model(all_detection_summaries = data, range_bins = range_bins, error_trim=2500)

            if save_SUM:
                # === Save state only ===
                save_pickle(pixel_uncertainty_model_state_data, pixel_model_state_path)
                save_numpy(range_bins, range_bins_path)
                save_pickle(range_error_model_input_format, range_model_input_format_path)
                save_pickle(range_model_params, range_model_params_path)
                save_pickle(detected_range_uncertainty_model_params, detected_range_model_params_path)
        else:
            print("Loading sensor uncertainty models from SUM/ cache...")
    
            # === Load Uncertainty Model Function State Data ===
            pixel_uncertainty_model_state_data = tuple(load_pickle(pixel_model_state_path))
            range_bins = load_numpy(range_bins_path)
            range_error_model_input_format = load_pickle(range_model_input_format_path)
            range_model_params = load_pickle(range_model_params_path)
            detected_range_uncertainty_model_params = load_pickle(detected_range_model_params_path)

            # === Rebuild Uncertainty Model Functions ===
            pixel_uncertainty_model = generate_pixel_uncertainty_model(bins=20, pregen_state_data = pixel_uncertainty_model_state_data)
            range_uncertainty_model = generate_range_uncertainty_model(range_bins = range_bins, error_trim=2500, pregen_state_data = range_error_model_input_format, pregen_model_params = range_model_params)
'''           
    #########################
    # import math 

    # # Load data
    # data = import_encounter_data()

    # all_gt_ranges = []
    # all_confidences = []

    # for detection_summary in data:
    #     for obj, obj_data in detection_summary.items():
    #         for frame_id, assoc_det in obj_data['analysis'].items():
    #             if assoc_det is None:
    #                 continue

    #             gt_obj = obj_data['ground_truth'].get(frame_id, None)
    #             if gt_obj is None:
    #                 continue

    #             gt_range = gt_obj.range_distance_m
    #             conf = assoc_det['confidence']

    #             all_gt_ranges.append(gt_range)
    #             all_confidences.append(conf)

    # # Convert to numpy arrays
    # all_gt_ranges = np.array(all_gt_ranges)
    # all_confidences = np.array(all_confidences)

    # # Scatter plot (keep as-is)
    # plt.figure(figsize=(10,6))
    # plt.scatter(all_gt_ranges, all_confidences, alpha=0.6, edgecolors='k')
    # plt.xlabel('Ground Truth Range (meters)')
    # plt.ylabel('Detection Confidence')
    # plt.title('Ground Truth Range vs Detection Confidence')
    # plt.grid(True)

    # # Define 300m bin edges
    # max_range = np.nanmax(all_gt_ranges)
    # bin_edges = np.arange(0, max_range + 300, 300)
    # num_bins = len(bin_edges) - 1

    # # Calculate grid size (e.g., 2 columns)
    # ncols = 4
    # nrows = math.ceil(num_bins / ncols)

    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows), sharex=True, sharey=True)
    # axes = axes.flatten()  # Flatten to easily index even if it's 1D for small cases

    # for i in range(num_bins):
    #     lower = bin_edges[i]
    #     upper = bin_edges[i+1]

    #     # Filter detections in this bin
    #     mask = (all_gt_ranges >= lower) & (all_gt_ranges < upper)
    #     bin_confidences = all_confidences[mask]

    #     # Plot histogram
    #     axes[i].hist(bin_confidences, bins=30, color='blue', alpha=0.7, edgecolor='black', density=True)
    #     axes[i].set_title(f"{int(lower)}m - {int(upper)}m (N={np.sum(mask)})")
    #     axes[i].set_xlabel('Confidence')
    #     axes[i].set_ylabel('Density')
    #     axes[i].set_xlim([0, 1])
    #     axes[i].grid(True)

    # # Hide any unused subplots (if total number of bins isn't a perfect multiple of ncols)
    # for j in range(num_bins, len(axes)):
    #     fig.delaxes(axes[j])

    # plt.tight_layout()
    # plt.show()

    ############################
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # === Load data ===
    data = import_encounter_data()

    all_gt_ranges = []
    all_confidences = []
    all_missed_prior_detections = []

    for detection_summary in data:
        for obj, obj_data in detection_summary.items():
            all_gt_ranges_this = []
            all_confidences_this = []
            all_missed_prior_detections_this = []
            first_detection = False
            for frame_id, gt_obj in obj_data['ground_truth'].items():
                assoc_det = obj_data['analysis'].get(frame_id, None)
                
                # Skip frames before first detection
                if not first_detection and assoc_det is None:
                    continue

                # After first detection: Always append one GT and one conf per frame

                # Ground truth
                if gt_obj is not None:
                    gt_range = gt_obj.range_distance_m
                    all_gt_ranges_this.append(gt_range)
                else:
                    all_gt_ranges_this.append(np.nan)

                # Confidence
                if assoc_det is not None:
                    conf = assoc_det['confidence']
                    first_detection = True
                    all_confidences_this.append(conf)
                else:
                    all_confidences_this.append(np.nan)

            if all_confidences_this:
                missed_dets = 0
                for idx, conf in enumerate(all_confidences_this):
                    if not np.isnan(conf):
                        all_missed_prior_detections_this.append(missed_dets)
                        missed_dets = 0
                    else:
                        if not np.isnan(all_gt_ranges_this[idx]):
                            all_missed_prior_detections_this.append(np.nan)
                            missed_dets += 1
                        else:
                            # If missing ground truth, the chain of known track is lost so we discard this
                            all_missed_prior_detections_this.append(np.nan)
                            missed_dets = 0

            all_gt_ranges.append(all_gt_ranges_this)
            all_confidences.append(all_confidences_this)
            all_missed_prior_detections.append(all_missed_prior_detections_this)

    # Flatten
    all_gt_ranges_flat = np.concatenate(all_gt_ranges)
    all_confidences_flat = np.concatenate(all_confidences)
    all_missed_flat = np.concatenate(all_missed_prior_detections)

    # Filter out NaNs (common for missed detections where there was no detection)
    valid_mask = ~np.isnan(all_missed_flat) & ~np.isnan(all_confidences_flat) & ~np.isnan(all_gt_ranges_flat)#(all_missed_flat < 20) & (all_missed_flat > 0)

    x = all_gt_ranges_flat[valid_mask]
    y = all_missed_flat[valid_mask]
    z = all_confidences_flat[valid_mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ####################

    loop_values = [0, None]  # Values of y to iterate over
    nrows = len(loop_values)
    ncols = 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))

    # for row_idx, i in enumerate(loop_values):
    #     if i is None:
    #         mask = (y > 1)
    #         x_this = x[mask]
    #         z_this = z[mask]
    #     else:
    #         mask = (y == i)
    #         x_this = x[mask]
    #         z_this = z[mask]

    #     # Handle cases where no data exists for this i
    #     if len(x_this) == 0 or len(z_this) == 0:
    #         continue

    #     # === Histogram of x (Ground Truth Range) ===
    #     axes[row_idx, 0].hist(x_this, bins=50, color='blue', alpha=0.7, edgecolor='black', density=True)
    #     axes[row_idx, 0].set_xlabel('Ground Truth Range (m)')
    #     axes[row_idx, 0].set_ylabel('Count')
    #     axes[row_idx, 0].set_title(f'X Histogram (y={i})')

    #     # === Histogram of z (Confidence) ===
    #     axes[row_idx, 1].hist(z_this, bins=50, color='green', alpha=0.7, edgecolor='black', density=True)
    #     axes[row_idx, 1].set_xlabel('Detection Confidence')
    #     axes[row_idx, 1].set_ylabel('Count')
    #     axes[row_idx, 1].set_xlim(0, 1)
    #     axes[row_idx, 1].set_title(f'Z Histogram (y={i})')

    #     # === 2D Histogram (x vs z Heatmap) ===
    #     x_bins = np.linspace(np.nanmin(x), np.nanmax(x), 50)
    #     z_bins = np.linspace(0, 1, 50)
    #     hist, xedges, zedges = np.histogram2d(x_this, z_this, bins=(x_bins, z_bins), density=True)

    #     pcm = axes[row_idx, 2].pcolormesh(xedges, zedges, hist.T, cmap='viridis', shading='auto')
    #     fig.colorbar(pcm, ax=axes[row_idx, 2], label='Count')

    #     axes[row_idx, 2].set_xlabel('Ground Truth Range (m)')
    #     axes[row_idx, 2].set_ylabel('Detection Confidence')
    #     axes[row_idx, 2].set_title(f'2D Histogram: X vs Z (y={i})')

    # plt.tight_layout()
    # plt.show() 
    
    from sklearn.mixture import GaussianMixture
    from scipy.stats import beta
    from sklearn.preprocessing import MinMaxScaler

    for row_idx, i in enumerate(loop_values):
        if i is None:
            mask = (y > 1)
        else:
            mask = (y == i)

        x_this = x[mask]
        z_this = z[mask]

        x_this = MinMaxScaler().fit_transform(x_this.reshape(-1, 1)).flatten()
        z_this = MinMaxScaler().fit_transform(z_this.reshape(-1, 1)).flatten()

        if len(x_this) == 0 or len(z_this) == 0:
            continue

        if i == 0:  
            # === Step 1: Bin x_this into 20 equal-width bins ===
            num_bins = 30
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bin_indices = np.digitize(x_this, bin_edges) - 1  # 0-based bin indices

            # Clip bin indices to valid range
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)

            # === Step 2: Create subplots ===
            fig, axes = plt.subplots(6, 5, figsize=(18, 12), sharex=True, sharey=True)
            axes = axes.flatten()

            for j in range(num_bins):
                mask_bin = (bin_indices == j)
                z_bin = z_this[mask_bin]

                ax = axes[j]
                if len(z_bin) > 0:
                    ax.hist(z_bin, bins=50, color='blue', alpha=0.7, edgecolor='black', density=True)
                    ax.set_title(f'Bin {j+1}: X in [{bin_edges[j]:.2f}, {bin_edges[j+1]:.2f}] (N={len(z_bin)})')
                else:
                    ax.set_title(f'Bin {j+1}: No Data')

                valid_z_mask = (z_bin > 0) & ~np.isnan(z_bin)
                z_log = np.log(z_bin[valid_z_mask].reshape(-1, 1))  # Step 1: Log-transform data
                z_log = z_log.reshape(-1, 1)
                gmm_z_log = GaussianMixture(n_components=3, random_state=0).fit(z_log)  # Step 2: Fit GMM in log-space

                # Step 3: Evaluate PDF in original Z space
                z_plot = np.linspace(0.001, 1, 300)  # Avoid log(0), start slightly above zero
                log_z_plot = np.log(z_plot.reshape(-1, 1))
                log_density = gmm_z_log.score_samples(log_z_plot)
                pdf = np.exp(log_density) / z_plot  # Jacobian correction for log-to-linear space

                # Step 4: Plot
                ax.plot(z_plot, pdf, color='red', linewidth=2, label='Lognormal GMM')
                ax.legend()

                ax.set_xlim(0, 1)  # Because z_this is normalized between 0 and 1
                ax.set_xlabel('Z (Normalized)')
                ax.set_ylabel('Density')

            plt.tight_layout()
            plt.show()

        # === Fit 2-Gaussian GMM for x ===
        x_vals = x_this[~np.isnan(x_this)].reshape(-1, 1)
        if len(x_vals) >= 2:
            gmm_x = GaussianMixture(n_components=2, random_state=0).fit(x_vals)
            x_plot = np.linspace(np.nanmin(x_this), np.nanmax(x_this), 300)
            x_pdf = np.exp(gmm_x.score_samples(x_plot.reshape(-1, 1)))
        else:
            x_plot, x_pdf = None, None

        # === Fit 2-Gaussian GMM for z ===
        z_vals = z_this[~np.isnan(z_this)].reshape(-1, 1)
        if len(z_vals) >= 2:
            gmm_z = GaussianMixture(n_components=2, random_state=0).fit(z_vals)
            z_plot = np.linspace(0, 1, 300)
            z_pdf = np.exp(gmm_z.score_samples(z_plot.reshape(-1, 1)))
        else:
            z_plot, z_pdf = None, None

        # === Histogram of x (Ground Truth Range) ===
        ax_x = axes[row_idx, 0]
        ax_x.hist(x_this, bins=50, color='blue', alpha=0.7, edgecolor='black', density=True)
        # if x_plot is not None:
        #     ax_x.plot(x_plot, x_pdf, color='red', linewidth=2, label='2-Gaussian GMM')
        ax_x.set_xlabel('Ground Truth Range (m)')
        ax_x.set_ylabel('Density')
        ax_x.set_title(f'X Histogram + GMM (y={i})')

        # === Beta Fit for i == 0 ===
        if i == 0:
            x_vals_beta = x_this[~np.isnan(x_this)]
            if len(x_vals_beta) > 0:
                # # Fit Beta distribution
                # a, b, loc, scale = beta.fit(x_vals_beta)
                # print(f"Fitted Beta params for y={i}: a={a:.3f}, b={b:.3f}")

                # # Generate Beta PDF
                # x_plot_beta = np.linspace(np.min(x_vals_beta), np.max(x_vals_beta), 300)
                # beta_pdf = beta.pdf(x_plot_beta, a, b, loc=loc, scale=scale)

                # # Plot Beta PDF
                # ax_x.plot(x_plot_beta, beta_pdf, color='purple', linewidth=2, label='Beta Fit')

                # from scipy.stats import lognorm

                # # Lognormal fit
                # lognorm_params = lognorm.fit(x_vals_beta)  # Force positive and meaningful scale
                # x_plot_lognorm = np.linspace(np.min(x_vals_beta), np.max(x_vals_beta), 300)
                # lognorm_pdf = lognorm.pdf(x_plot_lognorm, *lognorm_params)

                # ax_x.plot(x_plot_lognorm, lognorm_pdf, color='magenta', linewidth=2, label='Lognormal Fit')

                # Filter out non-positive data
                x_vals_lognorm = x_vals_beta[x_vals_beta > 0]
                log_x = np.log(x_vals_lognorm).reshape(-1, 1)

                # Fit GMM in log-space
                gmm_log = GaussianMixture(n_components=2, random_state=0).fit(log_x)

                # Evaluate PDF back in original x-space
                x_plot = np.linspace(np.min(x_vals_lognorm), np.max(x_vals_lognorm), 300)
                log_x_plot = np.log(x_plot).reshape(-1, 1)
                log_gmm_pdf = np.exp(gmm_log.score_samples(log_x_plot)) / x_plot  # Apply Jacobian correction for log-to-x transform

                # Plot
                ax_x.plot(x_plot, log_gmm_pdf, color='darkgreen', linewidth=2, label='Lognormal Mixture Fit')

        ax_x.legend()

        # === Histogram of z (Confidence) ===
        ax_z = axes[row_idx, 1]
        ax_z.hist(z_this, bins=50, color='green', alpha=0.7, edgecolor='black', density=True)
        # if z_plot is not None:
        #     ax_z.plot(z_plot, z_pdf, color='red', linewidth=2, label='2-Gaussian GMM')

        # === Beta Fit for i == 0 ===
        if i == 0:
            z_vals_beta = z_this[~np.isnan(z_this)]
            if len(z_vals_beta) > 0:
                # # Fit Beta distribution
                # a, b, loc, scale = beta.fit(z_vals_beta)
                # print(f"Fitted Beta params for y={i}: a={a:.3f}, b={b:.3f}")

                # # Generate Beta PDF
                # z_plot_beta = np.linspace(np.min(z_vals_beta), np.max(z_vals_beta), 300)
                # beta_pdf = beta.pdf(z_plot_beta, a, b, loc=loc, scale=scale)

                # # Plot Beta PDF
                # ax_z.plot(z_plot_beta, beta_pdf, color='purple', linewidth=2, label='Beta Fit')

                # from scipy.stats import lognorm

                # # Lognormal fit
                # lognorm_params = lognorm.fit(z_vals_beta)  # Force positive and meaningful scale
                # z_plot_lognorm = np.linspace(np.min(z_vals_beta), np.max(z_vals_beta), 300)
                # lognorm_pdf = lognorm.pdf(z_plot_lognorm, *lognorm_params)

                # ax_z.plot(z_plot_lognorm, lognorm_pdf, color='magenta', linewidth=2, label='Lognormal Fit')

                # Filter out non-positive data
                z_vals_lognorm = z_vals_beta[z_vals_beta > 0]
                log_z = np.log(z_vals_lognorm).reshape(-1, 1)

                # Fit GMM in log-space
                gmm_log = GaussianMixture(n_components=2, random_state=0).fit(log_z)

                # Evaluate PDF back in original x-space
                z_plot = np.linspace(np.min(z_vals_lognorm), np.max(z_vals_lognorm), 300)
                log_z_plot = np.log(z_plot).reshape(-1, 1)
                log_gmm_pdf = np.exp(gmm_log.score_samples(log_z_plot)) / z_plot  # Apply Jacobian correction for log-to-x transform

                # Plot
                ax_z.plot(z_plot, log_gmm_pdf, color='darkgreen', linewidth=2, label='Lognormal Mixture Fit')

        ax_z.legend()
        ax_z.set_xlabel('Detection Confidence')
        ax_z.set_ylabel('Density')
        ax_z.set_xlim(0, 1)
        ax_z.set_title(f'Z Histogram + GMM/Beta (y={i})')

        # === 2D Histogram (x vs z Heatmap) ===
        x_bins = np.linspace(np.nanmin(x_this), np.nanmax(x_this), 100)
        z_bins = np.linspace(0, 1, 100)
        hist, xedges, zedges = np.histogram2d(x_this, z_this, bins=(x_bins, z_bins), density=True)

        pcm = axes[row_idx, 2].pcolormesh(xedges, zedges, hist.T, cmap='viridis', shading='auto')
        fig.colorbar(pcm, ax=axes[row_idx, 2], label='Density')

        axes[row_idx, 2].set_xlabel('Ground Truth Range (m)')
        axes[row_idx, 2].set_ylabel('Detection Confidence')
        axes[row_idx, 2].set_title(f'2D Histogram: X vs Z (y={i})')

        #################

    plt.tight_layout()
    plt.show()                     

    #############################

    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, alpha=0.3)

    ax.set_xlabel('Ground Truth Range (meters)')
    ax.set_ylabel('Number of Missed Prior Detections')
    ax.set_zlabel('Detection Confidence')
    ax.set_title('3D Scatter: Range vs Missed Detections vs Confidence')

    plt.tight_layout()
    
    # === Second: 2D Projections (x vs y, x vs z, y vs z) ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # X vs Y
    axes[0].scatter(x, y, alpha=0.3)
    axes[0].set_xlabel('Ground Truth Range (meters)')
    axes[0].set_ylabel('Number of Missed Prior Detections')
    axes[0].set_title('X vs Y')

    # X vs Z
    axes[1].scatter(x, z, alpha=0.3)
    axes[1].set_xlabel('Ground Truth Range (meters)')
    axes[1].set_ylabel('Detection Confidence')
    axes[1].set_title('X vs Z')

    # Y vs Z
    axes[2].scatter(y, z, alpha=0.3)
    axes[2].set_xlabel('Number of Missed Prior Detections')
    axes[2].set_ylabel('Detection Confidence')
    axes[2].set_title('Y vs Z')

    plt.tight_layout()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_pairs = [
        (x, y, 'Ground Truth Range (meters)', 'Missed Prior Detections', 'X vs Y'),
        (x, z, 'Ground Truth Range (meters)', 'Detection Confidence', 'X vs Z'),
        (y, z, 'Missed Prior Detections', 'Detection Confidence', 'Y vs Z')
    ]

    for idx, (xdata, ydata, xlabel, ylabel, title) in enumerate(plot_pairs):
        ax = axes[idx]
        
        # Create 2D histogram (heatmap data)
        x_bins = np.linspace(np.nanmin(xdata), np.nanmax(xdata), 20)
        y_bins = np.linspace(np.nanmin(ydata), np.nanmax(ydata), 20)
        hist, xedges, yedges = np.histogram2d(xdata, ydata, bins=(x_bins, y_bins))

        # Plot heatmap
        pcm = ax.pcolormesh(xedges, yedges, np.log(hist.T), cmap='viridis', shading='auto')
        fig.colorbar(pcm, ax=ax, label='Count')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Heatmap: {title}')

    plt.tight_layout()
    plt.show()

    # unique_x_values = np.unique(x[~np.isnan(x)]).astype(int)
    # unique_x_values.sort()

    # histogram_ranges = []
    # start_idx = 0

    # while start_idx < len(unique_x_values):
    #     current_x_vals = []
    #     total_y_count = 0
    #     idx = start_idx

    #     while idx < len(unique_x_values) and total_y_count < 50:
    #         current_x = unique_x_values[idx]
    #         current_mask = x == current_x
    #         current_y_count = np.sum(current_mask)

    #         total_y_count += current_y_count
    #         current_x_vals.append(current_x)
    #         idx += 1

    #     histogram_ranges.append(current_x_vals)
    #     start_idx = idx  # Move to next X after this group

    # # Plotting
    # ncols = 4
    # nrows = math.ceil(len(histogram_ranges) / ncols)
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
    # axes = axes.flatten()

    # for plot_idx, x_vals_group in enumerate(histogram_ranges):
    #     mask = np.isin(x, x_vals_group)
    #     y_subset = y[mask]

    #     ax = axes[plot_idx]
    #     ax.hist(y_subset, bins=30, color='blue', alpha=0.7, edgecolor='black', density=True)

    #     x_label = f"Missed Prior X: {x_vals_group[0]} to {x_vals_group[-1]}" if len(x_vals_group) > 1 else f"X = {x_vals_group[0]}"
    #     ax.set_title(f"{x_label} (N={np.sum(mask)})")
    #     ax.set_xlabel('Confidence')
    #     ax.set_ylabel('Density')
    #     ax.set_xlim(0, 1)
    #     ax.grid(True)

    # # Hide unused axes
    # for j in range(len(histogram_ranges), len(axes)):
    #     fig.delaxes(axes[j])

    # plt.tight_layout()
    # plt.show()


    # # === Define range bins ===
    # max_range = np.nanmax(all_gt_ranges)
    # bin_edges = np.arange(0, max_range + 300, 300)
    # num_bins = len(bin_edges) - 1

    # # === Grid layout for subplots ===
    # ncols = 4
    # nrows = math.ceil(num_bins / ncols)

    # fig = plt.figure(figsize=(5 * ncols, 4 * nrows))

    # for i in range(num_bins):
    #     lower = bin_edges[i]
    #     upper = bin_edges[i + 1]

    #     mask = (all_gt_ranges >= lower) & (all_gt_ranges < upper)
    #     bin_confidences = all_confidences[mask]

    #     ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

    #     if len(bin_confidences) > 0:
    #         # Histogram bins along confidence axis
    #         hist_counts, y_edges = np.histogram(bin_confidences, bins=30, density=True)

    #         # Dummy X-axis: all x=1
    #         x_pos = np.ones_like(hist_counts)

    #         # Bar positions
    #         x = x_pos
    #         y = y_edges[:-1]
    #         z = np.zeros_like(hist_counts)

    #         dx = np.full_like(hist_counts, 0.8)  # Width of x-axis bars
    #         dy = np.diff(y_edges)               # Bin widths in confidence axis
    #         dz = hist_counts

    #         ax.bar3d(x, y, z, dx, dy, dz, color='blue', alpha=0.7, edgecolor='black')

    #     ax.set_title(f"{int(lower)}m - {int(upper)}m (N={np.sum(mask)})")
    #     ax.set_xlabel('Dummy X')
    #     ax.set_ylabel('Confidence')
    #     ax.set_zlabel('Log Density')
    #     ax.set_xlim(0, 2)  # Keep dummy X axis visible
    #     ax.grid(True)

    # plt.tight_layout()
    # plt.show()
'''
