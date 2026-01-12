import numpy as np
import cv2
import math
import os
from Utils.ImageUtils import *
from Utils.MiscUtils import *
from Utils.GeometryUtils import *
from Utils.MovingAverage import *
import random
import argparse
import yaml
import signal
import sys
from moviepy.editor import VideoFileClip

global_state = {
    'old_left_fit': [],
    'old_right_fit': [],
    'left_lane_detected': False,
    'right_lane_detected': False,
    'moving_average_left': MovingAverage(window_size=10),
    'moving_average_right': MovingAverage(window_size=10),
    'moving_average_R_r': MovingAverage(window_size=10),
    'moving_average_R_l': MovingAverage(window_size=10),
    'R_l_old': 0,
    'turn': "Go Straight",
    'left_avg_x': None,
    'right_avg_x': None
}


def calculate_histogram_uniformity(hist):
    hist_normalized = hist / np.sum(hist)
    entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-7))
    return entropy


def enhanced_clahe_processing(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    enhanced_l1 = clahe1.apply(l)
    enhanced_l2 = clahe2.apply(l)

    hist1 = cv2.calcHist([enhanced_l1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([enhanced_l2], [0], None, [256], [0, 256])

    uniformity1 = calculate_histogram_uniformity(hist1)
    uniformity2 = calculate_histogram_uniformity(hist2)

    enhanced_l = enhanced_l1 if uniformity1 > uniformity2 else enhanced_l2

    enhanced_lab = cv2.merge([enhanced_l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


def gradient_based_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

    direction = np.arctan2(np.abs(sobely), np.abs(sobelx))

    sobel_mask = np.uint8((sobel / np.max(sobel)) * 255)
    _, sobel_binary = cv2.threshold(sobel_mask, 50, 255, cv2.THRESH_BINARY)

    return sobel_binary.astype(np.uint8) * 255


def extractWhiteYellow_robust(image):
    enhanced_image = enhanced_clahe_processing(image)

    hls = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)

    white_hls = cv2.inRange(hls, (0, 200, 0), (180, 255, 255))
    white_lab = cv2.inRange(lab, (200, 120, 120), (255, 140, 140))

    yellow_hsv = cv2.inRange(hsv, (15, 50, 50), (35, 255, 255))
    yellow_hls = cv2.inRange(hls, (10, 50, 80), (35, 255, 255))

    l_channel = hls[:, :, 1]
    adaptive_thresh = np.percentile(l_channel, 85)
    adaptive_white = cv2.inRange(l_channel, adaptive_thresh - 40, 255)

    gradient_mask = gradient_based_detection(image)

    white_combined = cv2.bitwise_or(white_hls, white_lab)
    white_combined = cv2.bitwise_or(white_combined, adaptive_white)

    yellow_combined = cv2.bitwise_or(yellow_hsv, yellow_hls)

    combined = cv2.bitwise_or(white_combined, yellow_combined)
    combined = cv2.bitwise_and(combined, gradient_mask)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined


def extractWhiteYellow_enhanced(image):
    original_mask = extractWhiteYellow_robust(image)

    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h_channel = image_hls[:, :, 0]
    l_channel = image_hls[:, :, 1]
    s_channel = image_hls[:, :, 2]

    l_mask_enhanced = cv2.inRange(l_channel, 86, 134)
    s_mask_enhanced = cv2.inRange(s_channel, 53, 77)

    white_mask_enhanced = cv2.inRange(image_hls, (0, 80, 10), (180, 255, 255))

    enhanced_image = image.copy()
    lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    enhanced_lab = cv2.merge([enhanced_l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_hls = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HLS)
    enhanced_white_mask = cv2.inRange(enhanced_hls, (0, 160, 20), (180, 255, 255))

    combined_mask = cv2.bitwise_or(original_mask, white_mask_enhanced)
    combined_mask = cv2.bitwise_or(combined_mask, enhanced_white_mask)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if np.sum(combined_mask) < 300:
        kernel_large = np.ones((5, 5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel_large, iterations=1)

    return combined_mask


def extractWhite_enhanced(image, threshold=220):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(image_gray)

    _, image_thresh = cv2.threshold(enhanced_gray, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)

    return image_thresh


def getLeftRightPoints_enhanced(image):
    h_w, w_w = image.shape
    strip_width = 15
    left_indexes = []
    right_indexes = []

    thresh = 10

    iy_r_old = 0
    iy_l_old = 0

    thresh_iy = 10.0

    left_found = False
    right_found = False

    for h in range(int(h_w / strip_width), 1, -1):
        left_strip = image[(h - 1) * strip_width: h * strip_width, 0: int(w_w / 2.8)]
        right_strip = image[(h - 1) * strip_width: h * strip_width, int(2.2 * w_w / 4): w_w]

        if np.sum(left_strip) / 255 > thresh:
            ix_l, iy_l = np.where(left_strip == 255)
            if len(ix_l) > 0:
                ix_l = int(np.median(ix_l))
                iy_l = int(np.median(iy_l))
                ix_l = ix_l + (h - 1) * strip_width
                iy_l = iy_l

                if left_found:
                    del_y = np.abs(float(iy_l) - float(iy_l_old))
                    if del_y < thresh_iy:
                        index = (ix_l, iy_l)
                        left_indexes.append(index)
                        iy_l_old = iy_l
                else:
                    index = (ix_l, iy_l)
                    left_indexes.append(index)
                    left_found = True
                    iy_l_old = iy_l

        if np.sum(right_strip) / 255 > thresh:
            ix_r, iy_r = np.where(right_strip == 255)
            if len(ix_r) > 0:
                ix_r = int(np.median(ix_r))
                iy_r = int(np.median(iy_r))
                ix_r = ix_r + (h - 1) * strip_width
                iy_r = iy_r + int(2.2 * w_w / 4)

                if right_found:
                    del_y = np.abs(float(iy_r) - float(iy_r_old))
                    if del_y < thresh_iy:
                        index = (ix_r, iy_r)
                        right_indexes.append(index)
                        iy_r_old = iy_r
                else:
                    index = (ix_r, iy_r)
                    right_indexes.append(index)
                    right_found = True
                    iy_r_old = iy_r

    if len(left_indexes) < 3 or len(right_indexes) < 3:
        left_indexes, right_indexes = getLeftRightPoints_supplement(image, left_indexes, right_indexes)

    return left_indexes, right_indexes


def getLeftRightPoints_supplement(image, existing_left, existing_right):
    h_w, w_w = image.shape
    strip_width = 20
    thresh = 5

    for h in range(int(h_w / strip_width), 1, -1):
        left_strip = image[(h - 1) * strip_width: h * strip_width, 0: int(w_w / 2.5)]
        right_strip = image[(h - 1) * strip_width: h * strip_width, int(2 * w_w / 3): w_w]

        if len(existing_left) < 8 and np.sum(left_strip) / 255 > thresh:
            ix_l, iy_l = np.where(left_strip == 255)
            if len(ix_l) > 0:
                ix_l = int(np.mean(ix_l))
                iy_l = int(np.mean(iy_l))
                ix_l = ix_l + (h - 1) * strip_width
                existing_left.append((ix_l, iy_l))

        if len(existing_right) < 8 and np.sum(right_strip) / 255 > thresh:
            ix_r, iy_r = np.where(right_strip == 255)
            if len(ix_r) > 0:
                ix_r = int(np.mean(ix_r))
                iy_r = int(np.mean(iy_r))
                ix_r = ix_r + (h - 1) * strip_width
                iy_r = iy_r + int(2 * w_w / 3)
                existing_right.append((ix_r, iy_r))

    return existing_left, existing_right


def getWarpedLane(image, src_points_file):
    h, w = image.shape[:2]

    with open(src_points_file, 'r') as f:
        data = yaml.safe_load(f)

    left_region = np.array(data['left_region'], dtype=np.float32)
    right_region = np.array(data['right_region'], dtype=np.float32)

    image_size_x = 300
    image_size_y = 500

    left_dst_points = np.array([
        [0, image_size_y],
        [0, 0],
        [image_size_x // 2, 0],
        [image_size_x // 2, image_size_y]
    ], dtype=np.float32)

    right_dst_points = np.array([
        [image_size_x // 2, image_size_y],
        [image_size_x // 2, 0],
        [image_size_x, 0],
        [image_size_x, image_size_y]
    ], dtype=np.float32)

    H_left = cv2.getPerspectiveTransform(left_region, left_dst_points)
    H_right = cv2.getPerspectiveTransform(right_region, right_dst_points)

    warped_left = cv2.warpPerspective(image, H_left, (image_size_x, image_size_y))
    warped_right = cv2.warpPerspective(image, H_right, (image_size_x, image_size_y))

    warped_image = warped_left.copy()
    warped_image[:, image_size_x // 2:image_size_x] = warped_right[:, image_size_x // 2:image_size_x]

    H = H_left

    return warped_image, H


def addBasePoint(points, max_x):
    pts = np.array(points)
    y = pts[:, 1]
    y_avg = y[0]
    y_avg = int(y_avg)
    points.append((max_x, y_avg))
    return points


def getCurve(points, order=2):
    indexes = np.array(points)
    x = indexes[:, 0]
    y = indexes[:, 1]
    try:
        fit = np.polyfit(x, y, order)
        return fit
    except:
        return np.array([0, 0, 0])


def findCurvature(coef, x):
    if len(coef) < 3:
        return 0

    a, b, c = coef[0], coef[1], coef[2] if len(coef) > 2 else 0

    dy = 2 * a * x + b
    d2y = 2 * a

    if np.abs(d2y) < 1e-6:
        return 10000.0

    R = (1 + dy ** 2) ** (3 / 2)
    R = R / d2y

    R = np.abs(R)
    R = np.nan_to_num(R, nan=10000.0, posinf=10000.0, neginf=10000.0)

    return np.min(R)


def finalDisplay(image_undistorted, image_bin, image_warped, display_image, image_overlay, left_curvature,
                 right_curvature, old_turn, shift_status, stream_bad=False):
    image_undistorted_resized = cv2.resize(image_undistorted, (300, 168))
    image_bin_resized = cv2.resize(image_bin, (300, 168))
    image_bin_resized = cv2.merge((image_bin_resized, image_bin_resized, image_bin_resized))
    image_warped_colored = cv2.merge((image_warped, image_warped, image_warped))

    image_undistorted_resized = cv2.putText(image_undistorted_resized, '(1)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 0, 0), 2,
                                            cv2.LINE_AA)
    image_bin_resized = cv2.putText(image_bin_resized, '(2)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                    cv2.LINE_AA)
    image_warped_colored = cv2.putText(image_warped_colored, '(3)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                       2,
                                       cv2.LINE_AA)
    display_image_resized = cv2.putText(cv2.resize(display_image, (300, 500)), '(4)', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                        cv2.LINE_AA)

    side_pannel = np.zeros((720, 600, 3), np.uint8)
    side_pannel[0:168, 0:300, :] = image_undistorted_resized
    side_pannel[0:168, 300:600, :] = image_bin_resized
    side_pannel[168:668, 0:300, :] = cv2.resize(image_warped_colored, (300, 500))
    side_pannel[168:668, 300:600, :] = display_image_resized

    image_overlay_resized = cv2.resize(image_overlay, (1280, 720))

    full_pannel = np.zeros((900, 1880, 3), np.uint8)
    full_pannel[0:720, 0:1280, :] = image_overlay_resized
    full_pannel[0:720, 1280:1880, :] = side_pannel

    info_pannel = np.zeros((180, 1880, 3), np.uint8)
    info_pannel[:, :, 2] = 200
    info_pannel[:, :, 0] = 255
    info_pannel[:, :, 1] = 200
    info_pannel = cv2.putText(info_pannel,
                              '(1) : Undistorted image, (2) : Detected white and yellow lane markings, (3) : Warped image, (4) : Detected points and curve fitting',
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    curve_string = "Left Curvature : " + str(round(left_curvature, 2)) + ", Right Curvature : " + str(
        round(right_curvature, 2))
    info_pannel = cv2.putText(info_pannel, curve_string, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                              cv2.LINE_AA)

    turn_curvature = "Curvature not found!"
    turn = old_turn

    if not (np.isnan(left_curvature) or np.isnan(right_curvature)) and left_curvature != 0 and right_curvature != 0:
        left_sign = 1 if left_curvature > 0 else -1
        right_sign = 1 if right_curvature > 0 else -1

        if left_sign == right_sign:
            av = 0.3 * left_curvature + 0.7 * right_curvature
            turn_curvature = "Average Curvature : " + str(round(av, 2))
            if av < 4000 and av > 0:
                turn = "Turn Right"
            elif np.abs(av) > 4000:
                turn = "Go Straight"
            elif av < 0 and av > -4000:
                turn = "Turn Left"
            else:
                turn = "Go Straight"
        else:
            global_state_ref = globals().get('global_state', {'left_avg_x': None, 'right_avg_x': None})
            left_avg_x = global_state_ref['left_avg_x']
            right_avg_x = global_state_ref['right_avg_x']
            if left_avg_x is not None and right_avg_x is not None:
                shift_threshold = 20
                if abs(left_avg_x - right_avg_x) > shift_threshold:
                    if left_avg_x < right_avg_x:
                        turn = "Shift Left"
                    else:
                        turn = "Shift Right"
                else:
                    turn = "Go Straight"
            else:
                turn = "Go Straight"
    else:
        turn = "Go Straight"

    if stream_bad:
        turn += " - Stream bad.."

    info_pannel = cv2.putText(info_pannel, turn_curvature, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                              cv2.LINE_AA)

    full_pannel = cv2.putText(full_pannel, turn, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    full_pannel = np.vstack((full_pannel, info_pannel))

    return full_pannel, turn


def process_frame_for_moviepy(frame_bgr):
    """
    Process a single BGR frame using the existing logic.
    Returns the processed BGR frame.
    """
    global global_state
    old_left_fit = global_state['old_left_fit']
    old_right_fit = global_state['old_right_fit']
    moving_average_left = global_state['moving_average_left']
    moving_average_right = global_state['moving_average_right']
    moving_average_R_r = global_state['moving_average_R_r']
    moving_average_R_l = global_state['moving_average_R_l']
    R_l_old = global_state['R_l_old']
    turn = global_state['turn']

    frame_with_points = frame_bgr.copy()
    PerspectiveConfigFile = 'perspective_points.yaml'  # Assuming this is available

    try:
        with open(PerspectiveConfigFile, 'r') as f:
            data = yaml.safe_load(f)

        left_region_points = [tuple(point) for point in data['left_region']]
        right_region_points = [tuple(point) for point in data['right_region']]
        all_points = [tuple(point) for point in data['all_points']]
    except:
        all_points = [(1347, 1520), (1685, 1232), (2146, 1224), (2565, 1539)]
        left_region_points = [(1347, 1520), (1685, 1232), (1685, 1232), (1347, 1520)]
        right_region_points = [(2146, 1224), (2565, 1539), (2565, 1539), (2146, 1224)]

    for i, (x, y) in enumerate(left_region_points):
        if 0 <= x < frame_bgr.shape[1] and 0 <= y < frame_bgr.shape[0]:
            cv2.circle(frame_with_points, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(frame_with_points, f'L{i + 1}', (x + 15, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    left_pts_array = np.array(left_region_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame_with_points, [left_pts_array], isClosed=True, color=(0, 0, 255), thickness=2)

    for i, (x, y) in enumerate(right_region_points):
        if 0 <= x < frame_bgr.shape[1] and 0 <= y < frame_bgr.shape[0]:
            cv2.circle(frame_with_points, (x, y), 10, (0, 255, 255), -1)
            cv2.putText(frame_with_points, f'R{i + 1}', (x + 15, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    right_pts_array = np.array(right_region_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame_with_points, [right_pts_array], isClosed=True, color=(255, 0, 255), thickness=2)

    cv2.putText(frame_with_points,
                'Left Region: L1-BL, L2-UL, L3-UR, L4-BR | Right Region: R1-BL, R2-UL, R3-UR, R4-BR',
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

    # cv2.imshow('Raw Frame with Calibration Points', resized_frame_with_points) # Remove for moviepy

    image = frame_bgr
    h, w, _ = image.shape
    K, D = getCamera('../Data/Project2_Dataset2/data_2/cam_params.yaml')  # Load once or pass
    image_undistorted = cv2.undistort(image, K, D)
    image_overlay = image_undistorted.copy()

    image_roi, cropped_image = getROI(image_undistorted, 0.1)

    image_bin = extractWhiteYellow_enhanced(image_roi)

    kernel = np.ones((3, 3), np.uint8)
    image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    if np.sum(image_bin) < 500:
        kernel_large = np.ones((5, 5), np.uint8)
        image_bin = cv2.dilate(image_bin, kernel_large, iterations=1)

    image_warped, H = getWarpedLane(image_bin, PerspectiveConfigFile)

    left_indexes, right_indexes = getLeftRightPoints_enhanced(image_warped)
    display_image = drawDetections(image_warped, left_indexes, right_indexes)

    draw_points_left = None
    draw_points_right = None

    left_avg_x, right_avg_x = None, None
    if len(left_indexes) > 0:
        left_avg_x = np.mean([p[1] for p in left_indexes])
    if len(right_indexes) > 0:
        right_avg_x = np.mean([p[1] for p in right_indexes])
    global_state['left_avg_x'] = left_avg_x
    global_state['right_avg_x'] = right_avg_x

    left_lane_detected = False
    right_lane_detected = False

    if (len(left_indexes) > 2):
        left_indexes = addBasePoint(left_indexes, image_warped.shape[0])
        left_fit = getCurve(left_indexes, order=2)
        if len(left_fit) > 0 and not np.allclose(left_fit, [0, 0, 0]):
            old_left_fit = left_fit
            moving_average_left.addMarkers(left_fit)
            left_fit = moving_average_left.getAverage()
            left_lane_detected = True
    else:
        if len(old_left_fit) > 0 and not np.allclose(old_left_fit, [0, 0, 0]):
            left_fit = old_left_fit
            left_lane_detected = True
        else:
            left_fit = np.array([0, 0, 0])
            left_lane_detected = False

    if (len(right_indexes) > 2):
        right_indexes = addBasePoint(right_indexes, image_warped.shape[0])
        right_fit = getCurve(right_indexes, order=2)
        if len(right_fit) > 0 and not np.allclose(right_fit, [0, 0, 0]):
            moving_average_right.addMarkers(right_fit)
            right_fit = moving_average_right.getAverage()
            old_right_fit = right_fit
            right_lane_detected = True
    else:
        if len(old_right_fit) > 0 and not np.allclose(old_right_fit, [0, 0, 0]):
            right_fit = old_right_fit
            right_lane_detected = True
        else:
            right_fit = np.array([0, 0, 0])
            right_lane_detected = False

    if left_lane_detected:
        try:
            display_image, draw_points_left = drawCurve(display_image, left_fit, (0, 0, 255))
        except:
            left_lane_detected = False

    if right_lane_detected:
        try:
            display_image, draw_points_right = drawCurve(display_image, right_fit, (0, 255, 255))
        except:
            right_lane_detected = False

    if left_lane_detected and draw_points_left is not None:
        try:
            draw_points_left = draw_points_left.reshape(-1, 1, 2).astype(np.float32)
            draw_points_left_transformed = cv2.perspectiveTransform(draw_points_left, np.linalg.inv(H))
            draw_points_left_transformed = (draw_points_left_transformed.reshape(-1, 2)).astype(np.int32)
            image_overlay = cv2.polylines(image_overlay, [draw_points_left_transformed], False, (0, 0, 255), 4)
        except:
            pass

    if right_lane_detected and draw_points_right is not None:
        try:
            draw_points_right = draw_points_right.reshape(-1, 1, 2).astype(np.float32)

            with open(PerspectiveConfigFile, 'r') as f:
                data = yaml.safe_load(f)

            right_region = np.array(data['right_region'], dtype=np.float32)
            image_size_x = 300
            image_size_y = 500

            right_dst_points = np.array([
                [image_size_x // 2, image_size_y],
                [image_size_x // 2, 0],
                [image_size_x, 0],
                [image_size_x, image_size_y]
            ], dtype=np.float32)

            H_right = cv2.getPerspectiveTransform(right_region, right_dst_points)

            draw_points_right_transformed = cv2.perspectiveTransform(draw_points_right, np.linalg.inv(H_right))
            draw_points_right_transformed = (draw_points_right_transformed.reshape(-1, 2)).astype(np.int32)
            image_overlay = cv2.polylines(image_overlay, [draw_points_right_transformed], False, (0, 255, 255), 4)
        except:
            pass

    if (left_lane_detected and draw_points_left is not None and
            right_lane_detected and draw_points_right is not None):
        try:
            corners = np.vstack((draw_points_left_transformed, draw_points_right_transformed[::-1]))
            cv2.fillPoly(image_overlay, pts=[corners], color=(0, 0, 255), lineType=cv2.LINE_AA)
        except:
            pass

    cv2.addWeighted(image_overlay, 0.4, image_undistorted, 0.6, 0, image_overlay)

    h_w = image_warped.shape[0]
    x = np.linspace(0, h_w - 1, h_w)

    if left_lane_detected and len(old_left_fit) > 0:
        avg_x = np.mean(x) if len(x) > 0 else h_w // 2
        R_l = findCurvature(left_fit, avg_x)
        moving_average_R_l.addMarkers(R_l)
        R_l = moving_average_R_l.getAverage()
    else:
        R_l = 10000.0

    if right_lane_detected and len(old_right_fit) > 0:
        avg_x = np.mean(x) if len(x) > 0 else h_w // 2
        R_r = findCurvature(right_fit, avg_x)
        moving_average_R_r.addMarkers(R_r)
        R_r = moving_average_R_r.getAverage()
    else:
        R_r = 10000.0

    full_display, turn = finalDisplay(image_undistorted, image_bin, image_warped, display_image, image_overlay, R_l,
                                      R_r, turn, "", stream_bad=False)

    # Update global state
    global_state['old_left_fit'] = old_left_fit if left_lane_detected else global_state['old_left_fit']
    global_state['old_right_fit'] = old_right_fit if right_lane_detected else global_state['old_right_fit']
    global_state['turn'] = turn

    # Convert BGR to RGB for moviepy
    full_display_rgb = cv2.cvtColor(full_display, cv2.COLOR_BGR2RGB)
    return full_display_rgb


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='./', help='Base path of project2')
    Parser.add_argument('--VideoFilePath', default='../Data/Project2_Dataset2/data_2/harder_challenge_video.mp4',
                        help='relative image files path')
    Parser.add_argument('--CamConfigFile', default='../Data/Project2_Dataset2/data_2/cam_params.yaml',
                        help='.yaml config file name')
    Parser.add_argument('--PerspectiveConfigFile', default='perspective_points.yaml',
                        help='YAML file with 8 perspective points (2 regions of 4 points each)')
    Parser.add_argument('--SaveFileName', default='lane_result_2.mp4',
                        help='Saved video file name')

    Args = Parser.parse_args()
    BasePath = Args.BasePath
    VideoFilePath = Args.VideoFilePath
    SaveFileName = BasePath + Args.SaveFileName

    cap = cv2.VideoCapture(VideoFilePath)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame_for_moviepy(frame)

        # Convert back to BGR for display
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Lane Detection', processed_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()