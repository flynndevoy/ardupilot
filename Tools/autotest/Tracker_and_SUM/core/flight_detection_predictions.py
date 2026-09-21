import pandas as pd
import cv2
import matching_helpers

MIN_OBJECT_AREA_ = 100 # pixels
DEFAULT_IOU_IS_MATCH_ = 0.1
DEFAULT_IOU_IS_NO_MATCH_ = 0.02


class FlightDetectionPredictions:
    """
    This class holds detection prediction data related to a flight.
    """

    def __init__(self, filepath, flight):
        self.filepath = filepath
        self.flight = flight

        self.predictions_by_image_name = {}
        self.predictions_by_frame_id = {}

        self.load_prediction_file()

        self.register_detections()
    
    def load_prediction_file(self):
        # Opening JSON prediction file

        self.file_predictions = pd.read_json(self.filepath)

    def register_detections(self):
        
        frames = self.flight.frames # All the frames of the flight
        frame_ids = list(frames.keys())

        for frame_id in frame_ids:
            frame = self.flight.get_frame(frame_id)
            img_name = frame.frame_name

            # find all matching detections for particular frame
            detections = self.file_predictions.loc[self.file_predictions['img_name'] == img_name]["detections"]

            if img_name not in self.predictions_by_image_name:
                self.predictions_by_image_name[img_name] = detections
            
            if frame_id not in self.predictions_by_frame_id:
                self.predictions_by_frame_id[frame_id] = detections

    def get_detections_for_frame(self, frame):
        if frame.frame in self.predictions_by_frame_id.keys():
            return self.predictions_by_frame_id[frame.frame]
        else:
            return None
    
    def get_all_detection_bboxes_for_frame(self, frame):
        frame_detections = self.get_detections_for_frame(frame)
        
        result = []

        if frame_detections is not None:

            for pred in frame_detections:
                x = int(pred[0]['x'])
                y = int(pred[0]["y"])
                w = int(pred[0]["w"])
                h = int(pred[0]["h"])
                detection_w = w-x
                detection_h = h-y
                result.append([x, y, detection_w, detection_h])

        return result

    def get_only_planned_detections_for_frame(self, frame):
        # separate out planned and unplanned predicted bboxes (i.e. don't penalise for predicting an unplanned object: just remove it, but keep false positives, false negatives and true positives to planned objects)
        # 1: if it is matched to a planned object, keep it - TP
        # 2: if it is matched to an unplanned object, remove it -
        # 3: if it is NOT matched to any gt object planned or unplanned, keep it - FP
        # 4: Remaining unmatched GT - FN
        detections = []

        frame_detections = self.get_detections_for_frame(frame)
        all_gt_object_locations = frame.get_all_object_locations()

        for detection in frame_detections:
            detection_x = int(detection[0]['x'])
            detection_y = int(detection[0]["y"])
            w = int(detection[0]["w"])
            h = int(detection[0]["h"])
            detection_w = w-detection_x
            detection_h = h-detection_y
            pred_bbox = [detection_x, detection_y, detection_w, detection_h]
            matched = False

            for obj_id, obj_loc in all_gt_object_locations.items():

                eiou, is_match, is_false_positive = matching_helpers.calculate_eiou(obj_loc.bb.get_bbox_traditional(), [detection[0]['x'],detection[0]['y'],detection[0]['w'],detection[0]['h']], MIN_OBJECT_AREA_, DEFAULT_IOU_IS_MATCH_, DEFAULT_IOU_IS_NO_MATCH_)
                # print(f'eiou between pred {track_id} and gt {obj_id} = {eiou}')
                if is_match:
                    matched = True
                    if obj_loc.planned:
                        # Matched to a planned object (TP)
                        # print(f"Frame {frame.id} matched {track_id} to a planned object {obj_id}")
                        detections.append(detection)
                    # If matched to an unplanned object, do nothing (remove it/don't add it to result lists)
                    break

            # If not matched to any GT object, keep it (FP)
            if not matched:
                # print(f"Frame {frame.id} NO match for pred {track_id}")
                detections.append(detection)

        return detections

    def get_detections_for_frame(self, frame):
        if frame.frame in self.predictions_by_frame_id.keys():
            return self.predictions_by_frame_id[frame.frame]
        else:
            return None
    
    def annotate_frame_with_detections(self, frame_to_draw_on, frame, predictions_scale=1.0):
        """
        Annotates the frame with predictions, scaling the frame and bounding boxes if necessary.

        Parameters:
            frame_to_draw_on (np.ndarray): The frame to annotate.
            frame (np.ndarray): The scaled frame for obtaining predictions.
            scale (float): The scaling factor to resize the frame and bounding boxes. Default is 1.0 (no scaling).

        Returns:
            np.ndarray: The annotated and resized frame.
        """
        predictions = self.get_detections_for_frame(frame)

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
                frame_to_draw_on = cv2.rectangle(frame_to_draw_on, start_point, end_point, (255, 0, 255), 2)

                # Scale text position and add text
                text_position = (x, h + int(80 * predictions_scale))
                frame_to_draw_on = cv2.putText(frame_to_draw_on, f"Detection {pred_range} m", text_position, 
                                            cv2.FONT_HERSHEY_SIMPLEX, predictions_scale, (255, 0, 255), 2)

        return frame_to_draw_on