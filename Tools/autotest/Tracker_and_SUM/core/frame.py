from core.file_handler import FileHandler
import os
import imgaug as ia
import math
import cv2

class Frame:
    """
    This holds the information related to frame of the flight.

        Example for frame level data (multiple per frame):
        {
            'time': 1550844897919368155,
            'blob': {
                'frame': 480,
                'range_distance_m': nan
            },
            'id': 'Bird2',
            'bb': [1013.4, 515.8, 6.0, 6.0],
            'labels': {'is_above_horizon': 1},
            'flight_id': '280dc81adbb3420cab502fb88d6abf84',
            'img_name': '1550844897919368155280dc81adbb3420cab502fb88d6abf84.png'
        }
    """

    def __init__(self, entity, file_handler: FileHandler, flight):
        self.detected_object_locations = {}
        self.detected_objects = {}
        self.detected_object_angular_rates = {}
        self.id = entity['blob']['frame']
        self.timestamp = entity['time']
        self.file_handler = file_handler
        self.flight = flight
        self.dt = self.flight.metadata.dt # 10fps
        self.cam_angular_res = self.flight.metadata.cam_angular_res # deg/px

    def register_object_location(self, obj_location):
        if obj_location.object.id not in self.detected_objects:
            self.detected_objects[obj_location.object.id] = obj_location.object
        self.detected_object_locations[obj_location.object.id] = obj_location

        # angular rate calcs

        # only compute if 2 consecutive frames
        if len(self.detected_objects[obj_location.object.id].location) > 1:
            if (obj_location.frame.frame - self.detected_objects[obj_location.object.id].location[-2].frame.frame) <= 1:
                
                prev_obj_location = self.detected_objects[obj_location.object.id].location[-2].bb.get_center()
                curr_obj_location = obj_location.bb.get_center()

                dx = curr_obj_location[0] - prev_obj_location[0]
                dy = curr_obj_location[1] - prev_obj_location[1]
                
                vx = dx/self.dt
                vy = dy/self.dt

                angular_rate_ground_truth = self.cam_angular_res*math.sqrt(vx*vx + vy*vy)
                
                # save in a dict where key is the obj id and value is the ang rate
                self.detected_object_angular_rates[obj_location.object.id] = angular_rate_ground_truth


    def __str__(self):
        return "Frame#%s(time=%s, num_detected_objects=%s)" % (self.id, self.time, self.num_detected_objects)
    
    def get_obj_angular_rate(self, obj):
        if obj in self.detected_object_angular_rates.keys():
            return self.detected_object_angular_rates[obj]
        else:
            return None
        
    def get_planned_object_locations(self):
        result = {}

        for obj_id, obj_loc in self.detected_object_locations.items():
            if obj_loc.planned:
                result[obj_id] = obj_loc
        
        return result
    
    def get_unplanned_object_locations(self):
        result = {}

        for obj_id, obj_loc in self.detected_object_locations.items():
            if obj_loc.unplanned:
                result[obj_id] = obj_loc
        
        return result
    
    def get_all_object_locations(self):
        return self.detected_object_locations

    def get_above_horizon_object_locations(self):
        result = {}

        for obj_id, obj_loc in self.detected_object_locations.items():
            if obj_loc.above_horizon:
                result[obj_id] = obj_loc
        
        return result
    
    def get_above_horizon_planned_object_locations(self):
        result = {}

        for obj_id, obj_loc in self.detected_object_locations.items():
            if obj_loc.above_horizon and obj_loc.planned:
                result[obj_id] = obj_loc
        
        return result
    
    def get_below_horizon_object_locations(self):
        result = {}

        for obj_id, obj_loc in self.detected_object_locations.items():
            if not obj_loc.above_horizon:
                result[obj_id] = obj_loc
        
        return result
    
    def get_below_horizon_planned_object_locations(self):
        result = {}

        for obj_id, obj_loc in self.detected_object_locations.items():
            if not obj_loc.above_horizon and obj_loc.planned:
                result[obj_id] = obj_loc
        
        return result
    
    def get_objects_in_range(self, lower_range, upper_range):
        result = {}

        for obj_id, obj_loc in self.detected_object_locations.items():
            if obj_loc.planned:
                if lower_range <= obj_loc.range < upper_range:
                    result[obj_id] = obj_loc
        
        return result
    
    def get_below_horizon_objects_in_range(self, lower_range, upper_range):
        result = {}

        for obj_id, obj_loc in self.detected_object_locations.items():
            if not obj_loc.above_horizon and obj_loc.planned:
                if lower_range <= obj_loc.range < upper_range:
                    result[obj_id] = obj_loc
        
        return result
    
    def get_above_horizon_objects_in_range(self, lower_range, upper_range):
        result = {}

        for obj_id, obj_loc in self.detected_object_locations.items():
            if obj_loc.above_horizon and obj_loc.planned:
                if lower_range <= obj_loc.range < upper_range:
                    result[obj_id] = obj_loc
        
        return result


    @property
    def time(self):
        return self.timestamp

    @property
    def frame(self):
        return self.id
    
    @property
    def frame_name(self):
        return str(self.timestamp) + self.flight.id + '.png'

    @property
    def num_detected_objects(self):
        return len(self.detected_objects)

    def image_path(self):
        flight_id = self.flight.id
        if self.flight.prefix:
            flight_id = self.flight.id.split(self.flight.prefix)[1]
        return os.path.join('Images', self.flight.id, (str(self.timestamp) + flight_id + '.png'))

    def image_s3_path(self):
        flight_id = self.flight.id
        if self.flight.prefix:
            flight_id = self.flight.id.split(self.flight.prefix)[1]
        return os.path.join('Images', flight_id, (str(self.timestamp) + flight_id + '.png'))

    def image(self, type='pil', download_if_required=False):
        """
        Read the image of this frame in whichever type you want.

        type: It can be one of ["pil", "cv2", "numpy"]

        By default, normal PIL.Image is returned
        """
        return self.file_handler.get_file_content(s3_path=self.image_s3_path(), type=type, local_path=self.image_path(), download_if_required=download_if_required)

    def image_annotated(self, scale=1.0, download_if_required=False):
        """
        Annotates the image with bounding boxes, keypoints, and labels.

        Parameters:
            scale (float): The scaling factor to resize the image and bounding boxes. Default is 1.0 (no scaling).

        Returns:
            np.ndarray: The annotated and resized image.
        """
        img = self.image(type='cv2', download_if_required=download_if_required)

        # Resize the image if scaling is not 1.0
        if scale != 1.0:
            height, width = img.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))

        obj_locations_bbox_traditional = []
        obj_locations_centers = []

        for obj_name in self.detected_object_locations:
            # Scale bounding boxes
            bbox = self.detected_object_locations[obj_name].bb.get_bbox_traditional()
            scaled_bbox = [int(coord * scale) for coord in bbox]
            obj_locations_bbox_traditional.append(ia.BoundingBox(*scaled_bbox))

            # Scale keypoints (centers)
            center = self.detected_object_locations[obj_name].bb.get_center()
            scaled_center = [int(coord * scale) for coord in center]
            obj_locations_centers.append(ia.Keypoint(*scaled_center))

        bbsoi = ia.BoundingBoxesOnImage(obj_locations_bbox_traditional, shape=img.shape)
        img_annotated = bbsoi.draw_on_image(img, size=3, color=(255, 255, 0))

        kpsoi = ia.KeypointsOnImage(obj_locations_centers, shape=img.shape)
        img_annotated = kpsoi.draw_on_image(img_annotated, size=3, color=(255, 0, 255))

        for obj_name in self.detected_object_locations:
            # Scale text position
            top = int(self.detected_object_locations[obj_name].bb.top * scale)
            left = int(self.detected_object_locations[obj_name].bb.left * scale)

            img_annotated = ia.draw_text(img_annotated, top - 70, left, obj_name)
            if self.detected_object_locations[obj_name].range != self.detected_object_locations[obj_name].range:  # Check for NaN
                img_annotated = ia.draw_text(img_annotated, top - 35, left, "Range N/A")
            else:
                img_annotated = ia.draw_text(img_annotated, top - 35, left, f"{round(self.detected_object_locations[obj_name].range)} m")

        return img_annotated
