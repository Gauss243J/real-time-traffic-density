# motion_utils.py
import cv2
import numpy as np

def init_motion_detector(history=500, varThreshold=16, detectShadows=True):
    """
    Initialize a background subtractor and morphology kernels.
    Returns backSub, op_kernel, cl_kernel.
    """
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=varThreshold,
        detectShadows=detectShadows
    )
    op_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cl_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    return backSub, op_kernel, cl_kernel


def filter_by_motion(fgMask, boxes, op_kernel, cl_kernel, motion_thresh=0.05):
    """
    Filter YOLO bounding boxes by actual motion.
    Keep boxes where fraction of moving pixels > motion_thresh.
    """
    fg = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, op_kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cl_kernel)
    _, fg = cv2.threshold(fg, 254, 255, cv2.THRESH_BINARY)

    filtered = []
    for (x1, y1, x2, y2) in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        roi = fg[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        motion_ratio = (roi > 0).sum() / float(roi.size)
        if motion_ratio > motion_thresh:
            filtered.append((x1, y1, x2, y2))
    return filtered