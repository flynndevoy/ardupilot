import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import copy
import json
import math
from typing import List
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from Tracker_and_SUM import common_utils as common_utils
from Tracker_and_SUM.prediction_structure import *


def build_transform(frame_items):
    w = 2560
    h = 2048

    prev_tr = common_utils.build_geom_transform(
        dst_w=w,
        dst_h=h,
        src_center_x=w / 2 + frame_items.transform_dx,
        src_center_y=h / 2 + frame_items.transform_dy,
        scale_x=1.0,
        scale_y=1.0,
        angle=frame_items.transform_angle,
        return_params=True
    )

    return prev_tr


def track_items_trivial(flight_items):
    for frame_num, frame_items in flight_items.items():
        nb_frame_detections = 0
        for detected_object in frame_items.predicted:
            detected_object.add_to_submit = False

            if detected_object.distance > 800:
                continue
            if detected_object.confidence < 0.8:
                continue

            detected_object.add_to_submit = True
            detected_object.track_id = nb_frame_detections + 1
            nb_frame_detections += 1

    return flight_items


class SimpleOffsetTracker:
    def __init__(self,
                 min_track_size=8,
                 threshold_to_find=0.8,
                 threshold_to_continue=0.7,
                 threshold_distance=32,
                 min_distance=850,
                 max_frames_offset=3):
        self.prev_frames_items: List[List[DetectedItem]] = [[]]
        self.min_track_size = min_track_size
        self.threshold_to_find = threshold_to_find
        self.threshold_to_continue = threshold_to_continue
        self.threshold_distance = threshold_distance
        self.track_ids_in_use = set()
        self.min_distance = min_distance
        self.max_frames_offset = max_frames_offset
        

    def cost(self, cur_item: DetectedItem, prev_item: DetectedItem, frames_offset):
        diag = (cur_item.w ** 2 + cur_item.h ** 2) ** 0.5

        # allow some level of uncertainty for longer distance prediction
        predict_distance = ((cur_item.dx * frames_offset) ** 2 + (cur_item.dy * frames_offset) ** 2) ** 0.5

        diag32 = max(32.0, diag, predict_distance * 0.5) / 32.0
        # diag32_prev = max(32, (prev_item.w ** 2 + prev_item.h ** 2) ** 0.5) / 32.0
        diff_w = abs(math.log2(max(16, cur_item.w) / max(16, prev_item.w)))
        diff_h = abs(math.log2(max(16, cur_item.h) / max(16, prev_item.h)))

        offset_x = cur_item.cx + cur_item.dx * frames_offset - prev_item.cx
        offset_y = cur_item.cy + cur_item.dy * frames_offset - prev_item.cy

        offset = (offset_x**2 + offset_y**2) ** 0.5

        res = offset / diag32 + diff_w * 16 + diff_h * 16 + (1.0 - prev_item.confidence) * 8
        return res

    # def cost(self, cur_item: DetectedItem, prev_item: DetectedItem, frames_offset):
    #     diag = (cur_item.w ** 2 + cur_item.h ** 2) ** 0.5

    #     # === Acceleration-aware prediction ===
    #     # Try to get last 3 valid points (frames back: -0, -1, -2)
    #     past_cx = []
    #     past_cy = []

    #     # Helper: get previous positions from prev_item's linked history
    #     history_item = prev_item
    #     for _ in range(3):
    #         if history_item is None or np.isnan(history_item.cx) or np.isnan(history_item.cy):
    #             break
    #         past_cx.append(history_item.cx)
    #         past_cy.append(history_item.cy)

    #         if history_item.prev_item_idx is None:
    #             break
    #         frame_offset, prev_idx = history_item.prev_item_idx
    #         history_frame_index = -frame_offset
    #         if len(self.prev_frames_items) >= abs(history_frame_index):
    #             history_frame = self.prev_frames_items[history_frame_index]
    #             if prev_idx < len(history_frame):
    #                 history_item = history_frame[prev_idx]
    #             else:
    #                 break
    #         else:
    #             break

    #     past_cx = past_cx[::-1]  # Oldest first
    #     past_cy = past_cy[::-1]

    #     # Default prediction (position only)
    #     pred_cx = prev_item.cx
    #     pred_cy = prev_item.cy

    #     if len(past_cx) >= 3:
    #         # Fit quadratic over time (t = -2, -1, 0) and extrapolate to future
    #         t = np.array([-2, -1, 0])
    #         coeffs_x = np.polyfit(t, past_cx, 2)
    #         coeffs_y = np.polyfit(t, past_cy, 2)
    #         future_t = frames_offset
    #         pred_cx = np.polyval(coeffs_x, future_t)
    #         pred_cy = np.polyval(coeffs_y, future_t)

    #     elif len(past_cx) >= 2:
    #         # Linear extrapolation if only 2 points
    #         vx = past_cx[-1] - past_cx[-2]
    #         vy = past_cy[-1] - past_cy[-2]
    #         pred_cx = prev_item.cx + vx * frames_offset
    #         pred_cy = prev_item.cy + vy * frames_offset

    #     # Else: just static position (no prediction possible with less than 2 points)

    #     # === Spatial offset ===
    #     offset_x = cur_item.cx - pred_cx
    #     offset_y = cur_item.cy - pred_cy
    #     offset = (offset_x**2 + offset_y**2) ** 0.5

    #     # === Normalization scale ===
    #     predict_distance = np.hypot(cur_item.dx * frames_offset, cur_item.dy * frames_offset)
    #     diag32 = max(32.0, diag, predict_distance * 0.5) / 32.0

    #     # === Size penalties ===
    #     diff_w = abs(math.log2(max(16, cur_item.w) / max(16, prev_item.w)))
    #     diff_h = abs(math.log2(max(16, cur_item.h) / max(16, prev_item.h)))

    #     # === Final cost ===
    #     res = offset / diag32 + diff_w * 16 + diff_h * 16 + (1.0 - prev_item.confidence) * 8
    #     return res

    def allocate_track_id(self, init_time):
        for i in range(1, 10000):
            if i not in self.track_ids_in_use:
                self.track_ids_in_use.add(i)
                return f"{i}_init_{init_time}" 
        return f"1_init_{init_time}"

    def process_frame_detections(self, detections, transform, cam_obj, detection_time = 0.0) -> List[DetectedItem]:
        # project previous items to align with the current frame
        
        for prev_frame_items in self.prev_frames_items:
            prev_items_pos = []
            
            for prev_item in prev_frame_items:
                prev_items_pos.append([prev_item.cx, prev_item.cy])
            prev_items_pos = np.array(prev_items_pos)
            
            if len(prev_items_pos) == 0:
                continue

            prev_items_pos_projected = dict()
            prev_items_pos_projected = (transform[:2, :2] @ prev_items_pos.T).T + transform[:2, 2]
                

            for i, item in enumerate(prev_frame_items):
                # print(f"cam_id {item.cam_id}")
                # print(f"cx {item.cx}")
                # print(f"cy {item.cy}")
                item.cx, item.cy = prev_items_pos_projected[i, :]
                # print(f"proj cx {item.cx}")
                # print(f"proj cy {item.cy}")
                # print("---")


            ###
            # prev_items_pos = np.array([[it.cx, it.cy] for it in prev_frame_items])
            # if len(prev_items_pos) == 0:
            #     continue

            # prev_items_pos_projected = (transform[:2, :2] @ prev_items_pos.T).T + transform[:2, 2]

            # for i, item in enumerate(prev_frame_items):
            #     item.cx, item.cy = prev_items_pos_projected[i, :]

        items_to_keep = []  # to be added to history

        detections = copy.deepcopy(detections)
        for cur_item in detections:
            # Check for duplicates or superceded past detections
            cur_item_skipped = False
            for prev_frame_items_list in self.prev_frames_items:
                for prev_frame_item in prev_frame_items_list:
                    # 1. Detection in the past (shouldn't happen for the same camera since frames are sequential)
                    if detection_time < prev_frame_item.time:
                        cur_item_skipped = True
                        break
                    # 2. Duplicate entry - ignore
                    if (detection_time == prev_frame_item.time and
                        cur_item.abs_cx == prev_frame_item.abs_cx and
                        cur_item.abs_cy == prev_frame_item.abs_cy):
                        cur_item_skipped = True
                        break
                if cur_item_skipped:
                    break

            if cur_item_skipped:
                continue

            if cur_item.confidence < self.threshold_to_continue:
                continue

            # Recalibrate detection range via camera intrinsics
            cur_item.distance = cam_obj.calibrate_ranges(cur_item.distance, pre_scale_factor = 1.5)

            if cur_item.distance > self.min_distance:
                continue

            cur_item.prev_item_idx = None
            cur_item.next_item_idx = None
            cur_item.track_id = -1
            cur_item.items_in_track = 1
            cur_item.init_history = []
            cur_item.time = detection_time

            min_distance = 1e10
            prev_item_distances = []

            # check up to max_frames_offset frames back (in case you lose the track for a frame or two??!)
            for frames_offset in range(1, self.max_frames_offset + 1):
                if len(self.prev_frames_items) < frames_offset:
                    break

                prev_item_distances = [
                    self.cost(cur_item, prev_item, frames_offset) if ((prev_item.next_item_idx is None) and (cur_item.time > prev_item.time)) else 1000
                    for prev_item in self.prev_frames_items[-frames_offset]
                ]

                min_distance = min(prev_item_distances) if prev_item_distances else 1e10
                if min_distance < self.threshold_distance * frames_offset:
                    break

            if min_distance < self.threshold_distance * frames_offset:
                prev_item_idx = int(np.nanargmin(prev_item_distances))
                prev_item = self.prev_frames_items[-frames_offset][prev_item_idx]

                # Ignore if track not initialised and below threshold to find
                if not prev_item.add_to_submit and (cur_item.confidence < self.threshold_to_find):
                    continue
                else:
                    cur_item.items_in_track = prev_item.items_in_track + 1

                prev_item.next_item_idx = (frames_offset, len(items_to_keep))
                cur_item.prev_item_idx = (-frames_offset, prev_item_idx)

                cur_item.track_id = prev_item.track_id

                if cur_item.items_in_track > self.min_track_size:
                    cur_item.add_to_submit = True                    

                    if cur_item.track_id == -1:
                        cur_item.track_id = self.allocate_track_id(detection_time)

                ###### Populate initialisation history for this track ######
                # (1) If previous item was not added to submit (i.e. still in init phase)
                if not prev_item.add_to_submit:
                    # Populate current history with cumulative previous history
                    cur_item.init_history.extend(prev_item.init_history)
                    # Reset previous history now that it's been propagated up the chain to prevent nested duplicates
                    prev_item.init_history = []
                    # Now add cleaned up prev item to current history
                    cur_item.init_history.append(prev_item)
                    
                    #print("add to submit: " + str(cur_item.track_id))

            items_to_keep.append(cur_item)

        self.prev_frames_items.append(items_to_keep)

        # TODO: only store past min_track_size frames for already-initialised tracks and only store min track length for uninitialised tracks
        self.prev_frames_items = self.prev_frames_items[-self.min_track_size:]  # Store prev frame up to min track length to ensure init history

        self.track_ids_in_use.update(
            cur_item.track_id for cur_item in sum(self.prev_frames_items, [])
        )

        return [cur_item for cur_item in items_to_keep if cur_item.add_to_submit]

class KalmanOffsetTracker:
    def __init__(self,
                 min_track_size=8,
                 threshold_to_find=0.8,
                 threshold_to_continue=0.7,
                 threshold_distance=32,
                 min_distance=850):
        self.min_track_size = min_track_size
        self.threshold_to_find = threshold_to_find
        self.threshold_to_continue = threshold_to_continue
        self.threshold_distance = threshold_distance
        self.min_distance = min_distance

        self.active_tracks = {}  # track_id -> dict with KalmanFilter, track length, last detection, etc.
        self.next_track_id = 1
        self.max_track_id = 63

    def create_kalman_filter(self, initial_item):
        dt = 1.0
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State: [x, y, vx, vy]
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

        kf.P *= 500
        kf.R *= 10
        kf.Q *= 0.1

        kf.x[:2, 0] = np.array([initial_item.cx, initial_item.cy])
        kf.x[2:, 0] = np.array([initial_item.dx if not np.isnan(initial_item.dx) else 0,
                                initial_item.dy if not np.isnan(initial_item.dy) else 0])

        return kf

    def predict_all_tracks(self):
        predictions = {}
        for track_id, track_data in self.active_tracks.items():
            kf = track_data['kf']
            kf.predict()
            pred_x, pred_y = kf.x[0, 0], kf.x[1, 0]
            predictions[track_id] = (pred_x, pred_y)
        return predictions

    def process_frame_detections(self, detections: List['DetectedItem'], transform: np.ndarray) -> List['DetectedItem']:
        detections = copy.deepcopy(detections)

        # Step 1: Predict next positions for all active tracks
        track_predictions = self.predict_all_tracks()
        track_ids = list(self.active_tracks.keys())

        # Step 2: Build cost matrix (detections x tracks)
        cost_matrix = np.full((len(detections), len(track_ids)), 1e6)

        for d_idx, det in enumerate(detections):
            if det.confidence < self.threshold_to_continue or det.distance > self.min_distance:
                continue
            for t_idx, track_id in enumerate(track_ids):
                pred_x, pred_y = track_predictions[track_id]

                # Simple Euclidean distance for cost
                dx = det.cx - pred_x
                dy = det.cy - pred_y
                dist = np.hypot(dx, dy)

                # Normalize by box size
                diag = max(32.0, math.hypot(det.w, det.h))
                cost = dist / diag

                if cost < self.threshold_distance:
                    cost_matrix[d_idx, t_idx] = cost

        # Step 3: Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        submitted_items = []

        for d_idx, t_idx in zip(row_ind, col_ind):
            if cost_matrix[d_idx, t_idx] >= 1e5:
                continue  # Skip bad matches

            det = detections[d_idx]
            track_id = track_ids[t_idx]
            track_data = self.active_tracks[track_id]
            kf = track_data['kf']

            # Update Kalman filter with detection
            kf.update(np.array([det.cx, det.cy]))

            # Smooth detection output
            det.cx = kf.x[0, 0]
            det.cy = kf.x[1, 0]
            det.track_id = track_id
            track_data['length'] += 1
            track_data['last_detection'] = det

            if track_data['length'] >= self.min_track_size and det.confidence > self.threshold_to_find:
                det.add_to_submit = True
                submitted_items.append(det)

            matched_tracks.add(track_id)

        # Step 4: Handle unmatched detections → start new tracks
        for d_idx, det in enumerate(detections):
            if (d_idx not in row_ind) and det.confidence >= self.threshold_to_continue and det.distance <= self.min_distance:
                track_id = self.next_track_id
                self.next_track_id += 1
                if self.next_track_id > self.max_track_id:
                    self.next_track_id = 1

                kf = self.create_kalman_filter(det)
                self.active_tracks[track_id] = {'kf': kf, 'length': 1, 'last_detection': det}
                det.track_id = track_id

                if det.confidence > self.threshold_to_find and self.min_track_size <= 1:
                    det.add_to_submit = True
                    submitted_items.append(det)

        # Step 5: Optionally prune tracks that haven’t been matched for several frames (not shown here for simplicity)

        return submitted_items
