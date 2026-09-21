import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

class FlightTransformPredictions:
    """
    This class holds frame transform prediction data related to a flight.
    """

    def __init__(self, filepath, flight):
        self.filepath = filepath
        self.flight = flight

        self.transforms_by_image_name = {}
        self.transforms_by_frame_id = {}

        self.tr_global = np.eye(4)

        self.load_transform_file()

        self.register_transforms()
    
    def load_transform_file(self):
        # Opening JSON prediction file

        self.file_transforms = pd.read_json(self.filepath)

    def register_transforms(self):
        
        frames = self.flight.frames # All the frames of the flight
        frame_ids = list(frames.keys())

        for frame_id in frame_ids:
            frame = self.flight.get_frame(frame_id)
            img_name = frame.frame_name

            # find all matching detections for particular frame
            transform = self.file_transforms.loc[self.file_transforms['curr_img_name'] == img_name]["prev_to_curr_tr"]

            if img_name not in self.transforms_by_image_name:
                self.transforms_by_image_name[img_name] = transform
            
            if frame_id not in self.transforms_by_frame_id:
                self.transforms_by_frame_id[frame_id] = transform

    def visualize_transform(self, frame):
        # Create a grid of points
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        xx, yy = np.meshgrid(x, y)
        points = np.vstack([xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())])

        # Define an affine transformation matrix
        A = self.get_2D_transform_for_frame(frame)
        

        # Apply transformation
        transformed_points = A @ points

        # Plot original and transformed points
        fig, ax = plt.subplots()
        ax.plot(points[0], points[1], 'bo', label="Original")
        ax.plot(transformed_points[0], transformed_points[1], 'ro', label="Transformed")
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.legend()
        ax.set_title("Affine Transformation (Scaling + Rotation + Translation)")
        plt.show()

    def convert_2d_transform_to_3d_transform(self, affine_2d, z_translation=0):
        """
        Convert a 3x3 2D affine matrix to 4x4 3D affine matrix.
        """
        # Ensure it's a valid 3x3 matrix
        assert affine_2d.shape == (3, 3), "Each input matrix must be 3x3."

        # Create 4x4 matrix
        affine_3d = np.eye(4)  # Start with an identity matrix
        affine_3d[:2, :2] = affine_2d[:2, :2]  # Copy rotation/shear
        affine_3d[:2, 3] = affine_2d[:2, 2]    # Copy translation in X, Y
        affine_3d[2, 3] = z_translation        # Optional Z translation
        
        return affine_3d
    
    def get_2D_transform_for_frame(self, frame):
        if frame.frame in self.transforms_by_frame_id.keys():
            transform = self.transforms_by_frame_id[frame.frame]
            for tr in transform:
                tr = np.array(tr)
                if tr.shape == (3, 3):
                    return tr
                else: return None
        return None
        
    def get_3D_transform_for_frame(self, frame):
        if frame.frame in self.transforms_by_frame_id.keys():
            transform = self.transforms_by_frame_id[frame.frame]
            for tr in transform:
                tr = np.array(tr)
                if tr.shape == (3, 3):
                    return self.convert_2d_transform_to_3d_transform(tr)
                else: return None
        
        return None
    
    def get_planned_object_transform_corrected_coordinates(self, obj_id):
        result = {}

        if self.flight.detected_objects[obj_id].planned:

            for frame_id in self.flight.frames.keys():
                frame = self.flight.frames[frame_id]
                if obj_id in frame.detected_object_locations:
                    obj_location = frame.detected_object_locations[obj_id]

                    bbox_cx = obj_location.bb.get_center()[0]
                    bbox_cy = obj_location.bb.get_center()[1]
                    bbox_range = obj_location.range
                    
                    # points_3d = obj_location.get_3D_coordinates_relative_to_camera()

                    # points_homogeneous = np.array([points_3d['x'], points_3d['y'], points_3d['z'], 1])
                    
                    points_homogeneous = np.array([bbox_cx, bbox_cy, bbox_range, 1])

                    affine_matrix = self.get_3D_transform_for_frame(frame)

                    if affine_matrix is not None:

                        # self.visualize_transform(frame)

                        self.tr_global = self.tr_global @ np.linalg.inv(affine_matrix)

                        # Apply the affine transformation
                        # print(f'Before tr: {points_homogeneous}')
                        transformed_points = self.tr_global @ points_homogeneous  # Matrix multiplication
                        # print(f'After tr: {transformed_points}')

                        # Extract the adjusted 3D coordinates
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

        else:
            return None
        
        return result