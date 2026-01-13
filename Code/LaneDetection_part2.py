from Utils.ImageUtils import *
from Utils.MiscUtils import *
from Utils.GeometryUtils import *
from Utils.MovingAverage import *
import argparse
import yaml
import glob
import os
import numpy as np
import cv2

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
    'right_avg_x': None,
    'white_dashed_history': None,
    'left_lane_type': 'Unknown',
    'right_lane_type': 'Unknown',
    'left_line_pattern': 'Unknown',
    'right_line_pattern': 'Unknown',
    'type_confidence_left': 0.0,
    'type_confidence_right': 0.0
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


def gradient_based_detection_adaptive(image, base_thresh=16):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_mask = np.uint8((sobel / np.max(sobel)) * 255)
    mean_grad = np.mean(sobel_mask)
    adaptive_thresh = max(base_thresh, int(mean_grad * 0.3))
    _, sobel_binary = cv2.threshold(sobel_mask, adaptive_thresh, 255, cv2.THRESH_BINARY)
    return sobel_binary.astype(np.uint8)


def connect_dashed_lines(binary_mask, max_gap=80):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return binary_mask

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    if len(centers) == 0:
        return binary_mask

    centers.sort(key=lambda x: x[1], reverse=True)
    result = binary_mask.copy()

    for i in range(len(centers) - 1):
        cx1, cy1 = centers[i]
        cx2, cy2 = centers[i + 1]
        dy = cy1 - cy2
        if dy <= 0:
            continue
        dist = np.hypot(cx2 - cx1, cy2 - cy1)
        if dist <= max_gap:
            cv2.line(result, (cx1, cy1), (cx2, cy2), 255, thickness=2)

    return result


def enhance_yellow_sensitivity(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_b = clahe.apply(b)
    enhanced_lab = cv2.merge([l, a, enhanced_b])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_img


def post_process_yellow_detection(yellow_mask, min_area_threshold=50):
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(yellow_mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area_threshold:
            cv2.fillPoly(filtered_mask, [contour], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    filtered_mask = cv2.dilate(filtered_mask, kernel_dilate, iterations=1)
    return filtered_mask


def analyze_line_pattern(mask, points):
    if len(points) < 2:
        return 'Unknown', 0.0

    y_coords = [p[0] for p in points]

    sorted_points = sorted(points, key=lambda x: x[0])

    gaps = []
    for i in range(len(sorted_points) - 1):
        gap = sorted_points[i + 1][0] - sorted_points[i][0]
        gaps.append(gap)

    if len(gaps) == 0:
        return 'Solid', 0.8

    avg_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    unique_y_values = len(set(y_coords))
    total_y_range = max(y_coords) - min(y_coords) if max(y_coords) != min(y_coords) else 1
    density = unique_y_values / total_y_range

    GAP_THRESHOLD = 10
    STD_THRESHOLD = 5
    AREA_THRESHOLD = 150

    if avg_gap > GAP_THRESHOLD and std_gap > STD_THRESHOLD:
        confidence = min(0.95, 0.3 + 0.7 * (avg_gap / (avg_gap + 5)))
        return 'Dashed', confidence
    elif density > 0.8:
        confidence = min(0.95, 0.3 + 0.7 * density)
        return 'Solid', confidence
    else:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels > 3:
            avg_area = np.mean(stats[1:, cv2.CC_STAT_AREA])
            if avg_area < AREA_THRESHOLD:
                return 'Dashed', 0.8
            else:
                return 'Solid', 0.7
        else:
            return 'Solid', 0.9


def prioritize_solid_over_dashed(points, binary_mask, detected_pattern):
    """
    当同时检测到实线和虚线时，优先选择实线
    """
    if detected_pattern == 'Solid':
        return 'Solid'

    # 检查是否有足够的连续像素来确认实线
    if len(points) > 0:
        # 计算点的密度
        y_coords = [p[0] for p in points]
        x_coords = [p[1] for p in points]

        # 检查纵向连续性
        sorted_points = sorted(points, key=lambda x: x[0])
        consecutive_count = 1
        max_consecutive = 1

        for i in range(1, len(sorted_points)):
            if abs(sorted_points[i][0] - sorted_points[i - 1][0]) <= 2:  # 允许间隔1-2个像素
                consecutive_count += 1
            else:
                max_consecutive = max(max_consecutive, consecutive_count)
                consecutive_count = 1

        max_consecutive = max(max_consecutive, consecutive_count)

        # 如果有足够的连续像素，优先认定为实线
        if max_consecutive > 10:  # 阈值可以根据实际情况调整
            return 'Solid'

    return detected_pattern


def extractWhiteYellow_separate(image):
    enhanced_image = enhance_yellow_sensitivity(image)
    hls = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    white_by_hls = cv2.inRange(hls, (0, 131, 0), (180, 255, 107))
    white_by_hsv = cv2.inRange(hsv, (0, 0, 210), (180, 255, 255))

    white_combined = cv2.bitwise_or(white_by_hls, white_by_hsv)

    yellow_ranges = [
        (15, 100, 100, 35, 255, 255),
        (10, 50, 50, 40, 255, 255),
        (20, 80, 80, 30, 255, 255),
    ]
    yellow_combined = np.zeros_like(hls[:, :, 0])
    for h_min, s_min, v_min, h_max, s_max, v_max in yellow_ranges:
        temp_mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
        yellow_combined = cv2.bitwise_or(yellow_combined, temp_mask)
    yellow_hls_extra = cv2.inRange(hls, (10, 50, 120), (30, 255, 255))
    yellow_combined = cv2.bitwise_or(yellow_combined, yellow_hls_extra)
    yellow_combined = post_process_yellow_detection(yellow_combined)

    if np.sum(yellow_combined) < 200:
        yellow_loose = cv2.inRange(hsv, (12, 40, 40), (42, 255, 255))
        yellow_combined = cv2.bitwise_or(yellow_combined, yellow_loose)

    gradient_mask = gradient_based_detection_adaptive(enhanced_image, base_thresh=49)
    white_final = cv2.bitwise_and(white_combined, gradient_mask)

    kernel_horizontal = np.ones((1, 15), np.uint8)
    white_connected = cv2.morphologyEx(white_final, cv2.MORPH_CLOSE, kernel_horizontal, iterations=1)
    kernel_vertical = np.ones((3, 1), np.uint8)
    white_cleaned = cv2.morphologyEx(white_connected, cv2.MORPH_OPEN, kernel_vertical, iterations=1)
    white_final = connect_dashed_lines(white_cleaned, max_gap=50)
    kernel = np.ones((3, 3), np.uint8)
    white_final = cv2.morphologyEx(white_final, cv2.MORPH_CLOSE, kernel, iterations=1)
    white_final = cv2.morphologyEx(white_final, cv2.MORPH_OPEN, kernel, iterations=1)

    if np.sum(white_final) < 500:
        kernel_large = np.ones((3, 20), np.uint8)
        white_final = cv2.dilate(white_final, kernel_large, iterations=1)

    if np.sum(yellow_combined) < 300:
        yellow_combined = cv2.dilate(yellow_combined, np.ones((5, 5)), iterations=3)

    return white_final, yellow_combined


def getWarpedBinary(binary_mask, src_points_file):
    h, w = binary_mask.shape[:2]
    with open(src_points_file, 'r') as f:
        data = yaml.safe_load(f)
    left_region = np.array(data['left_region'], dtype=np.float32)
    right_region = np.array(data['right_region'], dtype=np.float32)
    image_size_x = 300
    image_size_y = 500
    left_dst_points = np.array([[0, image_size_y], [0, 0], [image_size_x // 2, 0], [image_size_x // 2, image_size_y]],
                               dtype=np.float32)
    right_dst_points = np.array(
        [[image_size_x // 2, image_size_y], [image_size_x // 2, 0], [image_size_x, 0], [image_size_x, image_size_y]],
        dtype=np.float32)
    H_left = cv2.getPerspectiveTransform(left_region, left_dst_points)
    H_right = cv2.getPerspectiveTransform(right_region, right_dst_points)
    warped_left = cv2.warpPerspective(binary_mask, H_left, (image_size_x, image_size_y))
    warped_right = cv2.warpPerspective(binary_mask, H_right, (image_size_x, image_size_y))
    warped = warped_left.copy()
    warped[:, image_size_x // 2:] = warped_right[:, image_size_x // 2:]
    return warped, H_left, H_right


def getLeftRightPoints_enhanced(image):
    h_w, w_w = image.shape
    strip_width = 15
    left_indexes = []
    right_indexes = []
    thresh = 6
    iy_r_old = 0
    iy_l_old = 0
    thresh_iy = 20.0
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
    thresh = 2
    for h in range(int(h_w / strip_width), 1, -1):
        left_strip = image[(h - 1) * strip_width: h * strip_width, 0: int(w_w / 2.5)]
        right_strip = image[(h - 1) * strip_width: h * strip_width, int(2 * w_w / 3): w_w]
        if len(existing_left) < 10 and np.sum(left_strip) / 255 > thresh:
            ix_l, iy_l = np.where(left_strip == 255)
            if len(ix_l) > 0:
                ix_l = int(np.mean(ix_l))
                iy_l = int(np.mean(iy_l))
                ix_l = ix_l + (h - 1) * strip_width
                existing_left.append((ix_l, iy_l))
        if len(existing_right) < 10 and np.sum(right_strip) / 255 > thresh:
            ix_r, iy_r = np.where(right_strip == 255)
            if len(ix_r) > 0:
                ix_r = int(np.mean(ix_r))
                iy_r = int(np.mean(iy_r))
                ix_r = ix_r + (h - 1) * strip_width
                iy_r = iy_r + int(2 * w_w / 3)
                existing_right.append((ix_r, iy_r))
    return existing_left, existing_right


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
    # 移除绝对值，保留符号信息
    R = np.nan_to_num(R, nan=10000.0, posinf=10000.0, neginf=-10000.0)
    return np.min(R)


def drawDetections(image, left_indexes, right_indexes):
    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in left_indexes:
        cv2.circle(display_image, (point[1], point[0]), 3, (255, 0, 0), -1)
    for point in right_indexes:
        cv2.circle(display_image, (point[1], point[0]), 3, (0, 255, 0), -1)
    return display_image


def drawCurve(image, fit, color):
    h, w = image.shape[:2]
    ploty = np.linspace(0, h - 1, h)
    if len(fit) == 3:
        plotx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    elif len(fit) == 2:
        plotx = fit[1] * ploty + fit[0]
    else:
        plotx = np.full_like(ploty, w // 2)
    plotx = np.clip(plotx, 0, w - 1).astype(np.int32)
    ploty = ploty.astype(np.int32)
    points = np.vstack((plotx, ploty)).T
    points = points.reshape((-1, 1, 2))
    for i in range(len(points) - 1):
        pt1 = tuple(points[i][0])
        pt2 = tuple(points[i + 1][0])
        cv2.line(image, pt1, pt2, color, 2)
    return image, points


def determine_lane_type_from_warped_masks(points, warped_white, warped_yellow):
    if len(points) == 0:
        return 'Unknown', 0.0

    ys = np.array([p[0] for p in points])
    xs = np.array([p[1] for p in points])

    h_w, w_w = warped_white.shape

    ys = np.clip(ys, 0, h_w - 1)
    xs = np.clip(xs, 0, w_w - 1)

    white_pixels = warped_white[ys, xs]
    yellow_pixels = warped_yellow[ys, xs]

    white_hits = np.sum(white_pixels > 0)
    yellow_hits = np.sum(yellow_pixels > 0)

    total = len(points)

    if total <= 5:
        if white_hits >= 1 and white_hits >= yellow_hits:
            confidence = min(0.9, 0.3 + 0.1 * white_hits)
            return 'White', confidence
        elif yellow_hits >= 1 and yellow_hits > white_hits:
            confidence = min(0.9, 0.3 + 0.1 * yellow_hits)
            return 'Yellow', confidence
        else:
            return 'Unknown', 0.0

    white_ratio = white_hits / total
    yellow_ratio = yellow_hits / total

    if white_ratio > 0.2 and white_ratio >= yellow_ratio:
        confidence = min(0.95, 0.4 + 0.6 * white_ratio)
        return 'White', confidence
    elif yellow_ratio > 0.2 and yellow_ratio > white_ratio:
        confidence = min(0.95, 0.4 + 0.6 * yellow_ratio)
        return 'Yellow', confidence
    else:
        return 'Unknown', 0.0


def finalDisplay(image_undistorted, image_bin, image_warped_colored, display_image, image_overlay, left_curvature,
                 right_curvature, old_turn, shift_status, left_lane_type='Unknown', right_lane_type='Unknown',
                 left_line_pattern='Unknown', right_line_pattern='Unknown',
                 left_confidence=0.0, right_confidence=0.0, stream_bad=False):
    image_undistorted_resized = cv2.resize(image_undistorted, (300, 168))
    image_bin_resized = cv2.resize(image_bin, (300, 168))
    image_bin_resized = cv2.merge((image_bin_resized, image_bin_resized, image_bin_resized))
    image_undistorted_resized = cv2.putText(image_undistorted_resized, '(1)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 0, 0), 2, cv2.LINE_AA)
    image_bin_resized = cv2.putText(image_bin_resized, '(2)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                    cv2.LINE_AA)
    image_warped_resized = cv2.resize(image_warped_colored, (300, 500))
    image_warped_resized = cv2.putText(image_warped_resized, '(3)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                       2, cv2.LINE_AA)
    display_image_resized = cv2.putText(cv2.resize(display_image, (300, 500)), '(4)', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    side_pannel = np.zeros((720, 600, 3), np.uint8)
    side_pannel[0:168, 0:300, :] = image_undistorted_resized
    side_pannel[0:168, 300:600, :] = image_bin_resized
    side_pannel[168:668, 0:300, :] = image_warped_resized
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

    # 修改这部分代码 - 反转左右转向判断
    if not (np.isnan(left_curvature) or np.isnan(right_curvature)) and left_curvature != 0 and right_curvature != 0:
        left_sign = 1 if left_curvature > 0 else -1
        right_sign = 1 if right_curvature > 0 else -1
        if left_sign == right_sign:
            av = 0.3 * left_curvature + 0.7 * right_curvature
            # 反转转向判断逻辑
            if av < 4000 and av > 0:
                turn = "Turn Right"  # 原来是Turn Left，现在改为Turn Right
            elif np.abs(av) > 4000:
                turn = "Go Straight"
            elif av < 0 and av > -4000:
                turn = "Turn Left"  # 原来是Turn Right，现在改为Turn Left
            else:
                turn = "Go Straight"
        else:
            # 移除平移检测部分，只保留基本转向判断
            turn = "Go Straight"
    else:
        turn = "Go Straight"
    if stream_bad:
        turn += " - Stream bad.."
    info_pannel = cv2.putText(info_pannel, turn_curvature, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                              cv2.LINE_AA)
    full_pannel = cv2.putText(full_pannel, turn, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    turn_info_y = 80
    lane_type_below_turn = f"Left: {left_lane_type}({left_confidence:.2f})-{left_line_pattern}, Right: {right_lane_type}({right_confidence:.2f})-{right_line_pattern}"
    full_pannel = cv2.putText(full_pannel, lane_type_below_turn, (50, turn_info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (0, 0, 255), 1, cv2.LINE_AA)
    full_pannel = np.vstack((full_pannel, info_pannel))
    return full_pannel, turn


def process_frame_for_moviepy(frame_bgr):
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
    PerspectiveConfigFile = 'perspective_points.yaml'
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
    image = frame_bgr
    enhanced_image = enhance_yellow_sensitivity(image)
    h, w, _ = image.shape
    K, D = getCamera('../Data/Project2_Dataset2/data_2/cam_params.yaml')
    image_undistorted = cv2.undistort(enhanced_image, K, D)
    image_overlay = image_undistorted.copy()
    image_roi, cropped_image = getROI(image_undistorted, 0.1)
    white_mask, yellow_mask = extractWhiteYellow_separate(image_roi)
    image_bin = cv2.bitwise_or(white_mask, yellow_mask)
    kernel = np.ones((3, 3), np.uint8)
    image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    if np.sum(image_bin) < 500:
        kernel_large = np.ones((5, 5), np.uint8)
        image_bin = cv2.dilate(image_bin, kernel_large, iterations=1)
    image_warped_binary, H_left, H_right = getWarpedBinary(image_bin, PerspectiveConfigFile)
    warped_white, _, _ = getWarpedBinary(white_mask, PerspectiveConfigFile)
    warped_yellow, _, _ = getWarpedBinary(yellow_mask, PerspectiveConfigFile)
    colored_warped = np.zeros((500, 300, 3), dtype=np.uint8)
    colored_warped[warped_white == 255] = [255, 255, 255]
    colored_warped[warped_yellow == 255] = [0, 255, 255]
    overlap = cv2.bitwise_and(warped_white, warped_yellow)
    colored_warped[overlap == 255] = [255, 255, 255]
    left_indexes, right_indexes = getLeftRightPoints_enhanced(image_warped_binary)
    display_image = drawDetections(image_warped_binary, left_indexes, right_indexes)
    draw_points_left = None
    draw_points_right = None
    left_avg_x, right_avg_x = None, None
    if len(left_indexes) > 0:
        left_avg_x = np.mean([p[1] for p in left_indexes])
    if len(right_indexes) > 0:
        right_avg_x = np.mean([p[1] for p in right_indexes])
    global_state['left_avg_x'] = left_avg_x
    global_state['right_avg_x'] = right_avg_x

    current_left_lane_type = 'Unknown'
    current_right_lane_type = 'Unknown'
    current_left_line_pattern = 'Unknown'
    current_right_line_pattern = 'Unknown'
    current_type_confidence_left = 0.0
    current_type_confidence_right = 0.0

    if len(left_indexes) > 0:
        left_type, left_conf = determine_lane_type_from_warped_masks(left_indexes, warped_white, warped_yellow)
        if left_conf > current_type_confidence_left:
            current_left_lane_type = left_type
            current_type_confidence_left = left_conf
        left_pattern, left_pattern_conf = analyze_line_pattern(image_warped_binary, left_indexes)
        if left_pattern_conf > 0.6:
            current_left_line_pattern = prioritize_solid_over_dashed(left_indexes, image_warped_binary, left_pattern)

    if len(right_indexes) > 0:
        right_type, right_conf = determine_lane_type_from_warped_masks(right_indexes, warped_white, warped_yellow)
        if right_conf > current_type_confidence_right:
            current_right_lane_type = right_type
            current_type_confidence_right = right_conf
        right_pattern, right_pattern_conf = analyze_line_pattern(image_warped_binary, right_indexes)
        if right_pattern_conf > 0.6:
            current_right_line_pattern = prioritize_solid_over_dashed(right_indexes, image_warped_binary, right_pattern)

    left_lane_detected = False
    right_lane_detected = False
    if (len(left_indexes) > 2):
        left_indexes = addBasePoint(left_indexes, image_warped_binary.shape[0])
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
        right_indexes = addBasePoint(right_indexes, image_warped_binary.shape[0])
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
            display_image, draw_points_left = drawCurve(display_image, left_fit, (255, 0, 0))
        except:
            left_lane_detected = False
    if right_lane_detected:
        try:
            display_image, draw_points_right = drawCurve(display_image, right_fit, (0, 255, 0))
        except:
            right_lane_detected = False
    if left_lane_detected and draw_points_left is not None:
        try:
            draw_points_left = draw_points_left.reshape(-1, 1, 2).astype(np.float32)
            draw_points_left_transformed = cv2.perspectiveTransform(draw_points_left, np.linalg.inv(H_left))
            draw_points_left_transformed = (draw_points_left_transformed.reshape(-1, 2)).astype(np.int32)

            if current_left_lane_type == 'Yellow':
                color = (0, 255, 255)
            elif current_left_lane_type == 'White':
                color = (255, 255, 255)
            else:
                color = (255, 0, 0)

            image_overlay = cv2.polylines(image_overlay, [draw_points_left_transformed], False, color, 4)
        except:
            pass
    if right_lane_detected and draw_points_right is not None:
        try:
            draw_points_right = draw_points_right.reshape(-1, 1, 2).astype(np.float32)
            draw_points_right_transformed = cv2.perspectiveTransform(draw_points_right, np.linalg.inv(H_right))
            draw_points_right_transformed = (draw_points_right_transformed.reshape(-1, 2)).astype(np.int32)

            if current_right_lane_type == 'Yellow':
                color = (0, 255, 255)
            elif current_right_lane_type == 'White':
                color = (255, 255, 255)
            else:
                color = (0, 255, 0)

            image_overlay = cv2.polylines(image_overlay, [draw_points_right_transformed], False, color, 4)
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
    h_w = image_warped_binary.shape[0]
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
    full_display, turn = finalDisplay(image_undistorted, image_bin, colored_warped,
                                      display_image, image_overlay, R_l,
                                      R_r, turn, "", current_left_lane_type, current_right_lane_type,
                                      current_left_line_pattern, current_right_line_pattern,
                                      current_type_confidence_left, current_type_confidence_right, stream_bad=False)

    global_state['old_left_fit'] = old_left_fit if left_lane_detected else global_state['old_left_fit']
    global_state['old_right_fit'] = old_right_fit if right_lane_detected else global_state['old_right_fit']
    global_state['turn'] = turn
    global_state['left_lane_type'] = current_left_lane_type
    global_state['right_lane_type'] = current_right_lane_type
    global_state['left_line_pattern'] = current_left_line_pattern
    global_state['right_line_pattern'] = current_right_line_pattern
    global_state['type_confidence_left'] = current_type_confidence_left
    global_state['type_confidence_right'] = current_type_confidence_right

    full_display_rgb = cv2.cvtColor(full_display, cv2.COLOR_BGR2RGB)
    return full_display_rgb


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--InputDir', default='../Data/Project2_Dataset2/data_2/images/',
                        help='Directory containing input .png images')
    Parser.add_argument('--OutputDir', default='output_images',
                        help='Directory to save processed .png results')
    Parser.add_argument('--CamConfigFile', default='../Data/Project2_Dataset2/data_2/cam_params.yaml',
                        help='Camera calibration YAML file')
    Parser.add_argument('--PerspectiveConfigFile', default='perspective_points.yaml',
                        help='YAML file with perspective points')
    Args = Parser.parse_args()
    os.makedirs(Args.OutputDir, exist_ok=True)
    png_files = glob.glob(os.path.join(Args.InputDir, "*.png")) + \
                glob.glob(os.path.join(Args.InputDir, "*.PNG"))
    if not png_files:
        print(f"No .png files found in {Args.InputDir}")
        return
    png_files.sort()
    for input_path in png_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(Args.OutputDir, filename)
        print(f"Processing: {filename}")
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            print(f"Warning: Could not load {input_path}. Skipping.")
            continue
        try:
            processed_rgb = process_frame_for_moviepy(image_bgr)
            processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(output_path, processed_bgr)
            if not success:
                print(f"Failed to write {output_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            continue
    print(f"Done! Processed {len(png_files)} images. Results saved to {Args.OutputDir}")


if __name__ == "__main__":
    main()