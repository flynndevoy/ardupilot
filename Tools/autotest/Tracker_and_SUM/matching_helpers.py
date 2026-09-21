import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each box is represented as [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

def dilate_bbox_to_min_area(box, min_area):
    """
    Dilate a bounding box to ensure it meets the minimum area.
    The dilation keeps the aspect ratio of the bounding box.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    current_area = width * height

    if current_area >= min_area:
        return box  # No dilation needed

    scale_factor = np.sqrt(min_area / current_area)
    new_width = width * scale_factor
    new_height = height * scale_factor

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    new_x1 = x_center - new_width / 2
    new_x2 = x_center + new_width / 2
    new_y1 = y_center - new_height / 2
    new_y2 = y_center + new_height / 2

    return [new_x1, new_y1, new_x2, new_y2]

def calculate_eiou(gt_bbox, det_bbox, min_object_area, is_match_min_iou_thresh, is_no_match_max_iou_thresh):
    """
    Calculate extended IoU (eIoU) and determine match and false positive conditions.
    """
    # Calculate ground truth area
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

    # Determine if dilation is needed
    if gt_area < min_object_area:
        gt_bbox = dilate_bbox_to_min_area(gt_bbox, min_object_area)
        det_bbox = dilate_bbox_to_min_area(det_bbox, min_object_area)

    # Calculate eIoU
    eiou = calculate_iou(gt_bbox, det_bbox)

    # Determine match and false positive status
    is_match = eiou >= is_match_min_iou_thresh
    is_false_positive = eiou < is_no_match_max_iou_thresh

    return eiou, is_match, is_false_positive