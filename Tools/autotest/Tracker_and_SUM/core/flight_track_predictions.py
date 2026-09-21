import matching_helpers
import math
import pandas as pd
import cv2
import statistics
import numpy as np


MIN_OBJECT_AREA_ = 100 # pixels
DEFAULT_IOU_IS_MATCH_ = 0.1
DEFAULT_IOU_IS_NO_MATCH_ = 0.02

class FlightTrackPredictions:
    """
    This class holds tracking prediction data related to a flight.
    """

    def __init__(self, filepath, flight, transform_predictions=None):
        self.filepath = filepath
        self.flight = flight
        self.transform_predictions = transform_predictions
        self.tr_global = np.eye(4)

        self.predictions_by_image_name = {}
        self.predictions_by_frame_id = {}
        self.predicted_angular_rate_by_frame = {}
        self.prediction_ground_truth_matches = {}
        self.angular_rate_errors_by_frame = {}
        self.range_errors_by_frame = {}
        
        self.bin_edges = list(range(0, 10001, 500))  # 0 to 10,000 in xm increments
        self.bin_labels = [f"{self.bin_edges[i]}-{self.bin_edges[i+1]}" for i in range(len(self.bin_edges) - 1)]
        # Initialize range bins
        self.angular_rate_pred_range_bins = {label: [] for label in self.bin_labels}
        self.angular_rate_error_range_bins = {label: [] for label in self.bin_labels}
        self.range_error_range_bins = {label: [] for label in self.bin_labels}
        self.range_error_percentage_range_bins = {label: [] for label in self.bin_labels}

        self.load_prediction_file()
        
        self.register_predictions()

        self.organise_all_stats_into_range_bins()

    def load_prediction_file(self):
        # Opening JSON prediction file

        self.file_predictions = pd.read_json(self.filepath)

    def register_predictions(self):

        frames = self.flight.frames # All the frames of the flight
        frame_ids = list(frames.keys())

        for frame_id in frame_ids:
            frame = self.flight.get_frame(frame_id)
            img_name = frame.frame_name

            # find all matching detections for particular frame

            if 'img_name' in self.file_predictions.columns and 'detections' in self.file_predictions.columns:
                detections = self.file_predictions.loc[self.file_predictions['img_name'] == img_name, 'detections']
            else:
                print("One or both columns 'img_name' or 'detections' are missing.")
                detections = pd.Series()  # Empty series or handle accordingly

            # detections = self.file_predictions.loc[self.file_predictions['img_name'] == img_name]["detections"]
            
            if img_name not in self.predictions_by_image_name:
                self.predictions_by_image_name[img_name] = detections
            
            if frame_id not in self.predictions_by_frame_id:
                self.predictions_by_frame_id[frame_id] = detections


            prev_frame_id = frame_id - 1
            
            if prev_frame_id in frame_ids:
                prev_frame = self.flight.get_frame(prev_frame_id)
                prev_img_name = prev_frame.frame_name

                if 'img_name' in self.file_predictions.columns and 'detections' in self.file_predictions.columns:
                    prev_detections = self.file_predictions.loc[self.file_predictions['img_name'] == prev_img_name, 'detections']
                else:
                    print("One or both columns 'img_name' or 'detections' are missing.")
                    prev_detections = pd.Series()  # Empty series or handle accordingly


                # prev_detections = self.file_predictions.loc[self.file_predictions['img_name'] == prev_img_name]["detections"]
            
                all_object_angular_rates = {}

                for detection in detections:

                    for prev_detection in prev_detections:

                        if detection[0]['track_id'] == prev_detection[0]['track_id']:

                            x = detection[0]['x']
                            y = detection[0]["y"]
                            w = detection[0]["w"]
                            h = detection[0]["h"]

                            detection_center_x = (x+w)/2
                            detection_center_y = (y+h)/2

                            prev_x = prev_detection[0]['x']
                            prev_y = prev_detection[0]["y"]
                            prev_w = prev_detection[0]["w"]
                            prev_h = prev_detection[0]["h"]

                            prev_detection_center_x = (prev_x+prev_w)/2
                            prev_detection_center_y = (prev_y+prev_h)/2

                            cam_angular_res = frame.cam_angular_res
                            dt = frame.dt

                            pred_dx = detection_center_x - prev_detection_center_x
                            pred_dy = detection_center_y - prev_detection_center_y
                            
                            pred_vx = pred_dx/dt
                            pred_vy = pred_dy/dt

                            prediction_angular_rate = cam_angular_res*math.sqrt(pred_vx*pred_vx + pred_vy*pred_vy)

                            all_object_angular_rates[detection[0]['track_id']] = prediction_angular_rate

                            # print(f"Track id {detection[0]['track_id']} predicted angular rate {prediction_angular_rate} between frame {prev_frame_id} and {frame_id}")


                if frame_id not in self.predicted_angular_rate_by_frame:
                    self.predicted_angular_rate_by_frame[frame_id] = all_object_angular_rates

       
            matches = self.find_matching_groundtruth(frame)
            
            if frame_id not in self.prediction_ground_truth_matches:
                self.prediction_ground_truth_matches[frame_id] = matches

            all_object_angular_rate_errors = {}
            all_object_range_errors = {}

            for ground_truth_object, prediction_track_id in self.get_prediction_ground_truth_matches(frame).items():
                gt_ang_rate = frame.get_obj_angular_rate(ground_truth_object)
                gt_distance_m = frame.detected_object_locations[ground_truth_object].range
                
                all_object_angular_rate_errors[ground_truth_object] = None
                all_object_range_errors[ground_truth_object] = None
                
                if gt_ang_rate is not None:
                    # print(f"Frame {frame_id} Object {ground_truth_object} angular rate = {gt_ang_rate}")
                    if prediction_track_id is not None:
                            pred_ang_rate = self.get_pred_id_angular_rate(frame, prediction_track_id)
                            if pred_ang_rate is not None:
                                ang_rate_error = abs(pred_ang_rate-gt_ang_rate)
                                all_object_angular_rate_errors[ground_truth_object] = ang_rate_error
                                # print(f"Frame {frame_id} Prediction {prediction_track_id} angular rate = {pred_ang_rate} (Error to Object {ground_truth_object} = {abs(pred_ang_rate-gt_ang_rate)})")           
                    # else:
                        # print(f"Frame {frame_id} No matched prediction")

                if not math.isnan(gt_distance_m):

                    if prediction_track_id is not None:
                        pred_range = self.get_pred_id_range(frame, prediction_track_id)

                        if pred_range is not None:
                            range_error = pred_range - gt_distance_m
                            all_object_range_errors[ground_truth_object] = range_error

            if frame_id not in self.angular_rate_errors_by_frame:
                self.angular_rate_errors_by_frame[frame_id] = all_object_angular_rate_errors

            if frame_id not in self.range_errors_by_frame:
                self.range_errors_by_frame[frame_id] = all_object_range_errors

    def get_predictions_for_frame(self, frame):
        if frame.frame in self.predictions_by_frame_id.keys():
            return self.predictions_by_frame_id[frame.frame]
        else:
            return None
        
    def get_all_prediction_bboxes_for_frame(self, frame):
        frame_predictions = self.get_predictions_for_frame(frame)
        result = []
        pred_ids = []

        if frame_predictions is not None:

            for pred in frame_predictions:
                track_id = pred[0]['track_id']
                x = int(pred[0]['x'])
                y = int(pred[0]["y"])
                w = int(pred[0]["w"])
                h = int(pred[0]["h"])
                detection_w = w-x
                detection_h = h-y
                result.append([x, y, detection_w, detection_h])
                pred_ids.append(track_id)

        return result, pred_ids
    
    def get_all_prediction_bboxes_for_frame_traditional(self, frame):
        frame_predictions = self.get_predictions_for_frame(frame)
        result = []
        pred_ids = []

        if frame_predictions is not None:

            for pred in frame_predictions:
                track_id = pred[0]['track_id']
                x = int(pred[0]['x'])
                y = int(pred[0]["y"])
                w = int(pred[0]["w"])
                h = int(pred[0]["h"])
                result.append([x, y, w, h])
                pred_ids.append(track_id)

        return result, pred_ids
    
    def get_only_planned_prediction_bboxes_for_frame(self, frame):
        # separate out planned and unplanned predicted bboxes (i.e. don't penalise for predicting an unplanned object: just remove it, but keep false positives, false negatives and true positives to planned objects)
        # 1: if it is matched to a planned object, keep it - TP
        # 2: if it is matched to an unplanned object, remove it -
        # 3: if it is NOT matched to any gt object planned or unplanned, keep it - FP
        # 4: Remaining unmatched GT - FN
        bboxes = []
        pred_ids = []

        frame_predictions = self.get_predictions_for_frame(frame)
        all_gt_object_locations = frame.get_all_object_locations()

        for pred in frame_predictions:
            track_id = pred[0]['track_id']
            detection_x = int(pred[0]['x'])
            detection_y = int(pred[0]["y"])
            w = int(pred[0]["w"])
            h = int(pred[0]["h"])
            detection_w = w-detection_x
            detection_h = h-detection_y
            pred_bbox = [detection_x, detection_y, detection_w, detection_h]
            matched = False

            for obj_id, obj_loc in all_gt_object_locations.items():

                
                eiou, is_match, is_false_positive = matching_helpers.calculate_eiou(obj_loc.bb.get_bbox_traditional(), [pred[0]['x'],pred[0]['y'],pred[0]['w'],pred[0]['h']], MIN_OBJECT_AREA_, DEFAULT_IOU_IS_MATCH_, DEFAULT_IOU_IS_NO_MATCH_)
                # print(f'eiou between pred {track_id} and gt {obj_id} = {eiou}')
                if is_match:
                    matched = True
                    if obj_loc.planned:
                        # Matched to a planned object (TP)
                        # print(f"Frame {frame.id} matched {track_id} to a planned object {obj_id}")
                        bboxes.append(pred_bbox)
                        pred_ids.append(track_id)
                    # If matched to an unplanned object, do nothing (remove it/don't add it to result lists)
                    break

            # If not matched to any GT object, keep it (FP)
            if not matched:
                # print(f"Frame {frame.id} NO match for pred {track_id}")
                bboxes.append(pred_bbox)
                pred_ids.append(track_id)

        return bboxes, pred_ids
    
    def get_only_planned_prediction_bboxes_for_frame_traditional(self, frame):
        # separate out planned and unplanned predicted bboxes (i.e. don't penalise for predicting an unplanned object: just remove it, but keep false positives, false negatives and true positives to planned objects)
        # 1: if it is matched to a planned object, keep it - TP
        # 2: if it is matched to an unplanned object, remove it -
        # 3: if it is NOT matched to any gt object planned or unplanned, keep it - FP
        # 4: Remaining unmatched GT - FN
        bboxes = []
        pred_ids = []

        frame_predictions = self.get_predictions_for_frame(frame)
        all_gt_object_locations = frame.get_all_object_locations()

        for pred in frame_predictions:
            track_id = pred[0]['track_id']
            detection_x = int(pred[0]['x'])
            detection_y = int(pred[0]["y"])
            w = int(pred[0]["w"])
            h = int(pred[0]["h"])
            detection_w = w-detection_x
            detection_h = h-detection_y
            pred_bbox = [detection_x, detection_y, detection_w, detection_h]
            pred_bbox_traditional = [detection_x, detection_y, w, h]
            matched = False

            for obj_id, obj_loc in all_gt_object_locations.items():

                
                eiou, is_match, is_false_positive = matching_helpers.calculate_eiou(obj_loc.bb.get_bbox_traditional(), [pred[0]['x'],pred[0]['y'],pred[0]['w'],pred[0]['h']], MIN_OBJECT_AREA_, DEFAULT_IOU_IS_MATCH_, DEFAULT_IOU_IS_NO_MATCH_)
                # print(f'eiou between pred {track_id} and gt {obj_id} = {eiou}')
                if is_match:
                    matched = True
                    if obj_loc.planned:
                        # Matched to a planned object (TP)
                        # print(f"Frame {frame.id} matched trk {track_id} to a planned object {obj_id}")
                        bboxes.append(pred_bbox_traditional)
                        pred_ids.append(track_id)
                    # If matched to an unplanned object, do nothing (remove it/don't add it to result lists)
                    break

            # If not matched to any GT object, keep it (FP)
            if not matched:
                # print(f"Frame {frame.id} NO match for pred {track_id}")
                bboxes.append(pred_bbox_traditional)
                pred_ids.append(track_id)

        return bboxes, pred_ids

    def get_prediction_count_for_frame(self, frame):
        if frame.frame in self.predictions_by_frame_id.keys():
            return self.predictions_by_frame_id[frame.frame].shape[0]
        else:
            return None
    
    def find_matching_groundtruth(self, frame):
        frame_predictions = self.get_predictions_for_frame(frame)

        groundtruth_matches = {}

        for obj_id in frame.detected_objects.keys():
                    
            if obj_id not in groundtruth_matches:
                groundtruth_matches[obj_id] = None

            if frame_predictions is not None:

                for pred in frame_predictions:
                
                    # start_point = (int(pred['x']), int(pred['y']))
                    # end_point = (int(pred['w']), int(pred['h']))

                    # see if bounding box overlaps
                    # Each box is represented as [x1, y1, x2, y2].
                    eiou, is_match, is_false_positive = matching_helpers.calculate_eiou(frame.detected_object_locations[obj_id].bb.get_bbox_traditional(), [pred[0]['x'],pred[0]['y'],pred[0]['w'],pred[0]['h']], MIN_OBJECT_AREA_, DEFAULT_IOU_IS_MATCH_, DEFAULT_IOU_IS_NO_MATCH_)
                    
                    # we have a detection match
                    if is_match:

                        groundtruth_matches[obj_id] = pred[0]['track_id']

                        # print(f"Frame {frame.frame} eIOU Match between Ground Truth {obj} and Track ID {pred[0]['track_id']}. eiou = {eiou}")
                        # return obj, pred[0]['track_id']

        return groundtruth_matches
                        
    def get_all_pred_angular_rate(self, frame):
        frame_id = frame.frame
        if frame_id in self.predicted_angular_rate_by_frame.keys():
            return self.predicted_angular_rate_by_frame[frame_id]
        else:
            return None
        
    def get_pred_id_angular_rate(self, frame, track_id):
        result = None
        frame_id = frame.frame
        if frame_id in self.predicted_angular_rate_by_frame.keys():
            if track_id in self.predicted_angular_rate_by_frame[frame_id]:
                result = self.predicted_angular_rate_by_frame[frame_id][track_id]
    
        return result
    
    def get_object_id_range_error(self, frame, object_id):
        result = None
        frame_id = frame.frame
        if frame_id in self.range_errors_by_frame.keys():
            if object_id in self.range_errors_by_frame[frame_id]:
                result = self.range_errors_by_frame[frame_id][object_id]
    
        return result

    def get_pred_id_range(self, frame, track_id):
        result = None
        frame_id = frame.frame
        if frame_id in self.predictions_by_frame_id.keys():
            for detection in self.predictions_by_frame_id[frame_id]:
                if detection[0]['track_id'] == track_id:
                    result = detection[0]['d']
                    break
    
        return result

    def get_angular_rate_errors_in_frame(self, frame):

        # empty dict means no gt objects, None means no match with pred

        frame_id = frame.frame
        if frame_id in self.angular_rate_errors_by_frame.keys():
            #return all angular rate error for each matched object
            return self.angular_rate_errors_by_frame[frame_id]
        else:
            return None

    def annotate_frame_with_predictions(self, frame_to_draw_on, frame, predictions_scale=1.0):
        """
        Annotates the frame with predictions, scaling the frame and bounding boxes if necessary.

        Parameters:
            frame_to_draw_on (np.ndarray): The frame to annotate.
            frame (np.ndarray): The scaled frame for obtaining predictions.
            scale (float): The scaling factor to resize the frame and bounding boxes. Default is 1.0 (no scaling).

        Returns:
            np.ndarray: The annotated and resized frame.
        """
        predictions = self.get_predictions_for_frame(frame)

        if predictions is not None:
            for detection in predictions:
                # Scale bounding box coordinates
                x = int(detection[0]['x'] * predictions_scale)
                y = int(detection[0]["y"] * predictions_scale)
                w = int(detection[0]["w"] * predictions_scale)
                h = int(detection[0]["h"] * predictions_scale)

                pred_range = str(round(detection[0]["d"]))

                start_point = (x, y)
                end_point = (w, h)

                # Draw the bounding box
                frame_to_draw_on = cv2.rectangle(frame_to_draw_on, start_point, end_point, (0, 100, 170), 2)

                # Scale text position and add text
                text_position = (x, h + int(40 * predictions_scale))
                frame_to_draw_on = cv2.putText(frame_to_draw_on, f"Track {detection[0]['track_id']} {pred_range} m", text_position, 
                                            cv2.FONT_HERSHEY_SIMPLEX, predictions_scale, (0, 100, 170), 2)

        return frame_to_draw_on

    def get_prediction_ground_truth_matches(self, frame):
        # For each object in a dict where obj id is key
        # None means no match for object id, otherwise pred_id is returned. 
        frame_id = frame.frame
        if frame_id in self.prediction_ground_truth_matches.keys():
            return self.prediction_ground_truth_matches[frame_id]

    def get_pred_ids_in_frame(self, frame):
        frame_preds_ids = []
        frame_predictions = self.get_predictions_for_frame(frame)

        if frame_predictions is not None:

            for pred in frame_predictions:
                frame_preds_ids.append(pred[0]['track_id'])

        return frame_preds_ids

    def get_all_mean_angular_rate_error(self):
        # should return None if no matched predictions == no angular rate errors
        
        result = None
        count = 0

        for frame_id, all_frame_errors in self.angular_rate_errors_by_frame.items():
            for object_id, object_error in all_frame_errors.items():
                if object_error is not None:
                    if result is None:
                        result = object_error
                    else:
                        result += object_error
                    count += 1

        if result is None:
            return result
        else:
            return result/count
        
    def get_all_angular_rate_error_std_dev_and_var(self):
        result = []
        
        for frame_id, all_frame_errors in self.angular_rate_errors_by_frame.items():
            for object_id, object_error in all_frame_errors.items():
                if object_error is not None:
                   result.append(object_error)

        return statistics.stdev(result), statistics.variance(result)
    
    def organise_all_stats_into_range_bins(self):
        # angular rate predictions based on gt range + angular rate error based on gt range + range error

        for frame_id, all_frame_ang_rate_errors in self.angular_rate_errors_by_frame.items():
            
            frame = self.flight.get_frame(frame_id)

            for object_id, object_ang_rate_error in all_frame_ang_rate_errors.items():

                matching_pred_track_id = self.prediction_ground_truth_matches[frame_id][object_id]
                
                if matching_pred_track_id is not None:
                    predicted_ang_rate = self.get_pred_id_angular_rate(frame, matching_pred_track_id)

                else:
                    predicted_ang_rate = None

                if object_ang_rate_error is not None:
                
                    distance_m = frame.detected_object_locations[object_id].range

                    if not math.isnan(distance_m):

                        # Determine the range bin
                        for i, (low, high) in enumerate(zip(self.bin_edges[:-1], self.bin_edges[1:])):
                            
                            if low <= distance_m < high:
                                
                                self.angular_rate_error_range_bins[self.bin_labels[i]].append(object_ang_rate_error)

                                if predicted_ang_rate is not None:
                                    self.angular_rate_pred_range_bins[self.bin_labels[i]].append(predicted_ang_rate)

                                break

                    # print(f"Frame {frame_id} Object {object_id} Distance {distance_m} ang_rate_error {object_error}")

                range_error = self.get_object_id_range_error(frame, object_id)

                if range_error is not None:
                    
                    abs_range_error = abs(range_error)
                    distance_m = frame.detected_object_locations[object_id].range

                    if not math.isnan(distance_m):

                        range_error_percentage = abs_range_error / distance_m * 100

                        # Determine the range bin
                        for i, (low, high) in enumerate(zip(self.bin_edges[:-1], self.bin_edges[1:])):
                            
                            if low <= distance_m < high:
                                
                                self.range_error_range_bins[self.bin_labels[i]].append(abs_range_error)
                                self.range_error_percentage_range_bins[self.bin_labels[i]].append(range_error_percentage)

                                break

    def get_angular_rate_error_range_bin_stats(self):
        return {
            bin_label: {
                "mean": np.mean(rates) if rates else None,
                "count": len(rates),
                "std": np.std(rates) if rates else None,
                "var": np.var(rates) if rates else None,
            }
            for bin_label, rates in self.angular_rate_error_range_bins.items()}
    
    def get_angular_rate_pred_range_bin_stats(self):
        return {
            bin_label: {
                "mean": np.mean(rates) if rates else None,
                "count": len(rates),
                "std": np.std(rates) if rates else None,
                "var": np.var(rates) if rates else None,
            }
            for bin_label, rates in self.angular_rate_pred_range_bins.items()}
    
    def get_range_error_range_bin_stats(self):
        return {
            bin_label: {
                "mean": np.mean(rates) if rates else None,
                "count": len(rates),
                "std": np.std(rates) if rates else None,
                "var": np.var(rates) if rates else None,
            }
            for bin_label, rates in self.range_error_range_bins.items()}
    
    def get_range_error_percentange_range_bin_stats(self):
        return {
            bin_label: {
                "mean": np.mean(rates) if rates else None,
                "count": len(rates),
                "std": np.std(rates) if rates else None,
                "var": np.var(rates) if rates else None,
            }
            for bin_label, rates in self.range_error_percentage_range_bins.items()}
    
    def get_planned_prediction_coordinates_inertial_frame(self, obj_id):
        result = {}

        if self.flight.detected_objects[obj_id].planned:

            for frame_id in self.flight.frames.keys():
                frame = self.flight.frames[frame_id]

                bboxes, pred_ids = self.get_only_planned_prediction_bboxes_for_frame(frame)

                for bbox, pred_id in zip(bboxes, pred_ids):
                    cx = bbox[0] + bbox[2]
                    cy = bbox[1] + bbox[3]
                    bbox_range = self.get_pred_id_range(frame, pred_id)

                    points_homogeneous = np.array([cx, cy, bbox_range, 1])

                    image_cx = int(frame.flight.metadata.center_x)
                    image_cy = int(frame.flight.metadata.center_y)
                    theta_x = (points_homogeneous[0]-image_cx)*frame.flight.metadata.cam_angular_res
                    theta_y = (points_homogeneous[1]-image_cy)*frame.flight.metadata.cam_angular_res

                    x = bbox_range * math.tan(math.radians(theta_x))
                    y = bbox_range * math.tan(math.radians(theta_y))
                    z = bbox_range

                    result[frame_id] =  {
                                        'x': x,
                                        'y': y,
                                        'z': z
                                    }
                    
        return result

    def get_planned_prediction_coordinates_transform_corrected(self, obj_id):
        result = {}

        if self.flight.detected_objects[obj_id].planned:

            for frame_id in self.flight.frames.keys():
                frame = self.flight.frames[frame_id]

                bboxes, pred_ids = self.get_only_planned_prediction_bboxes_for_frame(frame)

                for bbox, pred_id in zip(bboxes, pred_ids):
                    cx = bbox[0] + bbox[2]
                    cy = bbox[1] + bbox[3]
                    bbox_range = self.get_pred_id_range(frame, pred_id)

                    points_homogeneous = np.array([cx, cy, bbox_range, 1])

                    if self.transform_predictions is not None:
                        
                        affine_matrix = self.transform_predictions.get_3D_transform_for_frame(frame)
                        
                        if affine_matrix is not None:

                            self.tr_global = self.tr_global @ np.linalg.inv(affine_matrix)

                            transformed_points = self.tr_global @ points_homogeneous

                            adjusted_3d = transformed_points[:3]


                            image_cx = int(frame.flight.metadata.center_x)
                            image_cy = int(frame.flight.metadata.center_y)
                            theta_x = (adjusted_3d[0]-image_cx)*frame.flight.metadata.cam_angular_res
                            theta_y = (adjusted_3d[1]-image_cy)*frame.flight.metadata.cam_angular_res


                            x = bbox_range * math.tan(math.radians(theta_x))
                            y = bbox_range * math.tan(math.radians(theta_y))
                            z = bbox_range

                            result[frame_id] =  {
                                                'x': x,
                                                'y': y,
                                                'z': z
                                            }
                        else:
                            print(f'affine is none frame {frame_id}')
                    
        return result