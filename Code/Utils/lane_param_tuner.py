import cv2
import numpy as np


def adjust_lane_params_broken_lines(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image from {image_path}")
        return

    # Resize for display
    height, width = image.shape[:2]
    scale = min(800 / width, 600 / height, 1.0)
    image = cv2.resize(image, (int(width * scale), int(height * scale)))

    # Precompute color spaces
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1].astype(np.float32)

    # Gradient magnitude (for white line structure)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mag = np.uint8(255 * grad_mag / (np.max(grad_mag) + 1e-6))

    cv2.namedWindow('Lane Parameter Tuning (White + Yellow)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lane Parameter Tuning (White + Yellow)', 1000, 700)

    # === White lane trackbars (robust to broken lines) ===
    cv2.createTrackbar('White_Perc_L', 'Lane Parameter Tuning (White + Yellow)', 85, 100, lambda x: None)
    cv2.createTrackbar('White_Delta_L', 'Lane Parameter Tuning (White + Yellow)', 40, 100, lambda x: None)
    cv2.createTrackbar('White_Grad_Thresh', 'Lane Parameter Tuning (White + Yellow)', 50, 150, lambda x: None)
    cv2.createTrackbar('White_S_max', 'Lane Parameter Tuning (White + Yellow)', 60, 255, lambda x: None)

    # === Yellow lane trackbars (standard HLS) ===
    cv2.createTrackbar('Yellow_H_min', 'Lane Parameter Tuning (White + Yellow)', 10, 180, lambda x: None)
    cv2.createTrackbar('Yellow_H_max', 'Lane Parameter Tuning (White + Yellow)', 35, 180, lambda x: None)
    cv2.createTrackbar('Yellow_L_min', 'Lane Parameter Tuning (White + Yellow)', 50, 255, lambda x: None)
    cv2.createTrackbar('Yellow_S_min', 'Lane Parameter Tuning (White + Yellow)', 80, 255, lambda x: None)

    print("Tuning guide:")
    print("- White: uses adaptive brightness + edge constraint (good for broken lines)")
    print("- Yellow: uses standard HLS hue/saturation")
    print("- Press 'q' to quit and get parameters")

    while True:
        # --- White parameters ---
        w_perc_l = cv2.getTrackbarPos('White_Perc_L', 'Lane Parameter Tuning (White + Yellow)')
        w_delta_l = cv2.getTrackbarPos('White_Delta_L', 'Lane Parameter Tuning (White + Yellow)')
        w_grad_thresh = cv2.getTrackbarPos('White_Grad_Thresh', 'Lane Parameter Tuning (White + Yellow)')
        w_s_max = cv2.getTrackbarPos('White_S_max', 'Lane Parameter Tuning (White + Yellow)')

        # Adaptive white mask
        adaptive_thresh = np.percentile(l_channel, w_perc_l)
        l_min_white = max(0, int(adaptive_thresh - w_delta_l))
        white_mask_color = cv2.inRange(hls, (0, l_min_white, 0), (180, 255, w_s_max))

        # Gradient mask
        _, grad_mask = cv2.threshold(grad_mag, w_grad_thresh, 255, cv2.THRESH_BINARY)

        # Combine: must be bright AND have edge
        white_mask = cv2.bitwise_and(white_mask_color, grad_mask)

        # --- Yellow parameters ---
        y_h_min = cv2.getTrackbarPos('Yellow_H_min', 'Lane Parameter Tuning (White + Yellow)')
        y_h_max = cv2.getTrackbarPos('Yellow_H_max', 'Lane Parameter Tuning (White + Yellow)')
        y_l_min = cv2.getTrackbarPos('Yellow_L_min', 'Lane Parameter Tuning (White + Yellow)')
        y_s_min = cv2.getTrackbarPos('Yellow_S_min', 'Lane Parameter Tuning (White + Yellow)')

        yellow_mask = cv2.inRange(hls, (y_h_min, y_l_min, y_s_min), (y_h_max, 255, 255))

        # --- Combined result ---
        combined = cv2.bitwise_or(white_mask, yellow_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        # Display
        cv2.imshow('Original', image)
        cv2.imshow('White Mask (Adaptive + Edge)', white_mask)
        cv2.imshow('Yellow Mask (HLS)', yellow_mask)
        cv2.imshow('Combined Lane Mask', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Print final parameters
    print("\n" + "="*50)
    print("FINAL PARAMETERS FOR YOUR CODE")
    print("="*50)
    print("# For WHITE lane (robust to broken lines):")
    print(f"percentile_L = {w_perc_l}")
    print(f"delta_L = {w_delta_l}")
    print(f"grad_threshold = {w_grad_thresh}")
    print(f"white_S_max = {w_s_max}\n")

    print("# For YELLOW lane (HLS range):")
    print(f"yellow_hls_lower = ({y_h_min}, {y_l_min}, {y_s_min})")
    print(f"yellow_hls_upper = ({y_h_max}, 255, 255)")

    return {
        'white': {
            'percentile_L': w_perc_l,
            'delta_L': w_delta_l,
            'grad_threshold': w_grad_thresh,
            'S_max': w_s_max
        },
        'yellow': {
            'lower': (y_h_min, y_l_min, y_s_min),
            'upper': (y_h_max, 255, 255)
        }
    }


# Usage
image_path = '/home/xtjiang/Documents/Project/Lane Features Detection/Lane-Detection/Data/Project2_Dataset2/data_2/images/YW_3.png'
params = adjust_lane_params_broken_lines(image_path)