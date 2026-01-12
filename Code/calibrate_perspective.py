# calibrate_perspective.py
import cv2
import numpy as np
import argparse
import os
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Manually select 8 points for perspective transform (2 regions of 4 points each).")
    parser.add_argument('--VideoFilePath', default='../Data/Project2_Dataset2/data_2/harder_challenge_video.mp4',
                        help='Path to input video file')
    parser.add_argument('--ImageFilePath', default='',
                        help='Optional: Use an image instead of video')
    parser.add_argument('--OutputFile', default='perspective_points.yaml',
                        help='Output YAML file to save the 8 points')
    args = parser.parse_args()

    # Load original frame
    if args.ImageFilePath and os.path.exists(args.ImageFilePath):
        orig_img = cv2.imread(args.ImageFilePath)
        print(f"Loaded image: {args.ImageFilePath}")
    else:
        cap = cv2.VideoCapture(args.VideoFilePath)
        if not cap.isOpened():
            print(f"Error: Cannot open video {args.VideoFilePath}")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error: Cannot read first frame from video.")
            return
        orig_img = frame.copy()
        print(f"Loaded first frame from video: {args.VideoFilePath}")

    h_orig, w_orig = orig_img.shape[:2]
    print(f"Original image size: {w_orig} x {h_orig}")

    # Compute scale for display
    MAX_WIDTH, MAX_HEIGHT = 1000, 800
    scale = min(MAX_WIDTH / w_orig, MAX_HEIGHT / h_orig, 1.0)
    disp_w = int(w_orig * scale)
    disp_h = int(h_orig * scale)
    disp_img = cv2.resize(orig_img, (disp_w, disp_h))

    clone_disp = disp_img.copy()
    points_orig = []  # Store points in original coordinate system

    def click_and_crop(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_orig) < 8:
            # Convert display coordinate back to original image coordinate
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            points_orig.append((x_orig, y_orig))  # Still tuple internally
            # Draw on display image
            cv2.circle(disp_img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select 8 Points: [Left Region: BL, UL, UR, BR], [Right Region: BL, UL, UR, BR]", disp_img)
            print(f"Point {len(points_orig)}: ({x_orig}, {y_orig}) in original image")

    print("Please click 8 points in this order:")
    print("LEFT REGION (4 points):")
    print("  1. Bottom-left  (BL)")
    print("  2. Top-left     (UL)")
    print("  3. Top-right    (UR)")
    print("  4. Bottom-right (BR)")
    print("RIGHT REGION (4 points):")
    print("  5. Bottom-left  (BL)")
    print("  6. Top-left     (UL)")
    print("  7. Top-right    (UR)")
    print("  8. Bottom-right (BR)")
    print("Press 'r' to reset, 'q' to quit.")

    cv2.namedWindow("Select 8 Points: [Left Region: BL, UL, UR, BR], [Right Region: BL, UL, UR, BR]",
                    cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Select 8 Points: [Left Region: BL, UL, UR, BR], [Right Region: BL, UL, UR, BR]",
                         click_and_crop)

    while True:
        cv2.imshow("Select 8 Points: [Left Region: BL, UL, UR, BR], [Right Region: BL, UL, UR, BR]", disp_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            disp_img[:] = clone_disp[:]
            points_orig = []
            print("Points reset.")
        elif key == ord("q") or len(points_orig) == 8:
            break

    cv2.destroyAllWindows()

    if len(points_orig) == 8:
        print("\nSelected points (original coordinates):")
        for i, pt in enumerate(points_orig):
            region = "Left" if i < 4 else "Right"
            corner = ["BL", "UL", "UR", "BR"][i % 4]
            print(f"  Point {i + 1} ({region} {corner}): ({pt[0]}, {pt[1]})")

        # Convert tuples to lists for clean YAML output
        src_points_list = [list(pt) for pt in points_orig]

        data = {
            'left_region': src_points_list[:4],
            'right_region': src_points_list[4:8],
            'all_points': src_points_list,
            'image_size': [w_orig, h_orig]
        }

        # Write YAML with clean format
        with open(args.OutputFile, 'w') as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)
        print(f"\nSaved to {args.OutputFile}")

        # Visualize on original image (small window)
        visual = orig_img.copy()

        # Draw left region
        for i in range(4):
            pt = points_orig[i]
            cv2.circle(visual, pt, 10, (0, 255, 0), -1)
            cv2.putText(visual, f"L{i + 1}", (pt[0] + 15, pt[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        left_pts = np.array(points_orig[:4], np.int32).reshape((-1, 1, 2))
        cv2.polylines(visual, [left_pts], isClosed=True, color=(0, 0, 255), thickness=3)

        # Draw right region
        for i in range(4, 8):
            pt = points_orig[i]
            cv2.circle(visual, pt, 10, (0, 255, 0), -1)
            cv2.putText(visual, f"R{i - 3}", (pt[0] + 15, pt[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        right_pts = np.array(points_orig[4:], np.int32).reshape((-1, 1, 2))
        cv2.polylines(visual, [right_pts], isClosed=True, color=(255, 0, 255), thickness=3)

        # Resize final viz for display
        vis_disp = cv2.resize(visual, (disp_w, disp_h))
        cv2.imshow("Selected Points (Original Coordinates)", vis_disp)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not enough points selected. Exiting.")


if __name__ == "__main__":
    main()