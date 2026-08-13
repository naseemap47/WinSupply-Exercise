import cv2
import csv
import os
import argparse


def draw_polygon_and_save(image_path, csv_path):
    """
    Draw polygon on a large image.

    - Image is resized to fit the screen.
    - Mouse coordinates are converted back to original-image coordinates.
    - CSV contains original-resolution coordinates.
    
    Controls:
        Left click  -> Add point
        Right click -> Remove last point
        ENTER       -> Finish and save
        ESC         -> Exit
    """

    # Load original image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original_h, original_w = image.shape[:2]

    print(f"Original image size: {original_w} x {original_h}")

    # Get screen resolution
    screen_width = 1920
    screen_height = 1080

    # Create a temporary fullscreen window to get actual screen size
    cv2.namedWindow("Draw Polygon", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Draw Polygon",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # Calculate scaling factor
    scale_x = screen_width / original_w
    scale_y = screen_height / original_h

    # Use the smaller scale so the entire image fits
    scale = min(scale_x, scale_y)

    # Don't enlarge small images
    scale = min(scale, 1.0)

    display_w = int(original_w * scale)
    display_h = int(original_h * scale)

    print(f"Display image size: {display_w} x {display_h}")
    print(f"Scale: {scale:.6f}")

    # Resize image for display
    display_image = cv2.resize(
        image,
        (display_w, display_h),
        interpolation=cv2.INTER_AREA
    )

    original_display = display_image.copy()

    # Free original image memory as it is not needed during drawing
    del image

    # Points stored in ORIGINAL image coordinates
    points = []
    need_redraw = True

    # Mouse callback
    def mouse_callback(event, x, y, flags, param):

        nonlocal display_image, need_redraw

        if event == cv2.EVENT_LBUTTONDOWN:

            # Convert display coordinates -> original coordinates
            original_x = int(round(x / scale))
            original_y = int(round(y / scale))

            # Make sure coordinates stay inside image
            original_x = max(0, min(original_x, original_w - 1))
            original_y = max(0, min(original_y, original_h - 1))

            # Save ORIGINAL coordinates
            points.append((original_x, original_y))

            print(
                f"Point {len(points)}: "
                f"display=({x}, {y}) "
                f"original=({original_x}, {original_y})"
            )

            # Draw using DISPLAY coordinates
            dx = int(round(original_x * scale))
            dy = int(round(original_y * scale))

            cv2.circle(
                display_image,
                (dx, dy),
                5,
                (0, 0, 255),
                -1
            )

            # Draw line
            if len(points) > 1:
                prev_x = int(round(points[-2][0] * scale))
                prev_y = int(round(points[-2][1] * scale))

                cv2.line(
                    display_image,
                    (prev_x, prev_y),
                    (dx, dy),
                    (0, 255, 0),
                    2
                )

            need_redraw = True

        # Right click -> remove last point
        elif event == cv2.EVENT_RBUTTONDOWN:

            if len(points) > 0:

                removed = points.pop()
                print(f"Removed point: {removed}")

                # Redraw everything
                display_image = original_display.copy()

                for i, (px, py) in enumerate(points):

                    dx = int(round(px * scale))
                    dy = int(round(py * scale))

                    cv2.circle(
                        display_image,
                        (dx, dy),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    if i > 0:
                        prev_px = int(round(points[i - 1][0] * scale))
                        prev_py = int(round(points[i - 1][1] * scale))

                        cv2.line(
                            display_image,
                            (prev_px, prev_py),
                            (dx, dy),
                            (0, 255, 0),
                            2
                        )

                need_redraw = True

    # Register mouse callback
    cv2.setMouseCallback(
        "Draw Polygon",
        mouse_callback
    )

    print("\nControls:")
    print("  LEFT CLICK  -> Add point")
    print("  RIGHT CLICK -> Remove last point")
    print("  ENTER       -> Save polygon")
    print("  ESC         -> Exit")

    # Main loop
    while True:

        if need_redraw:
            cv2.imshow(
                "Draw Polygon",
                display_image
            )
            need_redraw = False

        key = cv2.waitKey(20) & 0xFF

        # ENTER -> save
        if key == 13:

            if len(points) < 3:
                print("Polygon requires at least 3 points.")
                continue

            # Close polygon visually
            first_x = int(round(points[0][0] * scale))
            first_y = int(round(points[0][1] * scale))

            last_x = int(round(points[-1][0] * scale))
            last_y = int(round(points[-1][1] * scale))

            cv2.line(
                display_image,
                (last_x, last_y),
                (first_x, first_y),
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Draw Polygon",
                display_image
            )

            cv2.waitKey(300)

            # Save ORIGINAL coordinates to CSV
            with open(
                csv_path,
                "w",
                newline=""
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    "point_id",
                    "x",
                    "y"
                ])

                for i, (x, y) in enumerate(points):

                    writer.writerow([
                        i + 1,
                        x,
                        y
                    ])

            print("\nPolygon saved!")
            print(f"CSV: {csv_path}")

            print("\nOriginal image coordinates:")

            for i, (x, y) in enumerate(points):

                print(
                    f"Point {i + 1}: "
                    f"x={x}, y={y}"
                )

            break

        # ESC -> exit without saving
        elif key == 27:

            print("Exited without saving.")
            break

    cv2.destroyAllWindows()



def load_roi(csv_path):
    """
    Reads polygon coordinates from a CSV file.
    Expects columns: point_id, x, y (or x, y).
    Returns an np.ndarray of shape (N, 2) with dtype int32.
    """
    import os
    import numpy as np

    if not os.path.exists(csv_path):
        return None

    points = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = int(float(row["x"]))
            y = int(float(row["y"]))
            points.append([x, y])

    if not points:
        return None

    return np.array(points, dtype=np.int32)


def load_all_rois(rois_dir_or_csv="rois"):
    """
    Loads ROI definitions from a directory containing <sheet_name>.csv files
    or from a single CSV file.
    Returns a dictionary mapping sheet_name -> np.ndarray of shape (N, 2).
    """
    import os
    import glob

    rois = {}
    if os.path.isdir(rois_dir_or_csv):
        csv_files = glob.glob(os.path.join(rois_dir_or_csv, "*.csv"))
        for csv_file in csv_files:
            sheet_name = os.path.splitext(os.path.basename(csv_file))[0]
            pts = load_roi(csv_file)
            if pts is not None:
                rois[sheet_name] = pts
    elif os.path.isfile(rois_dir_or_csv):
        sheet_name = os.path.splitext(os.path.basename(rois_dir_or_csv))[0]
        pts = load_roi(rois_dir_or_csv)
        if pts is not None:
            rois[sheet_name] = pts

    return rois


# Alias for compatibility
load_rois = load_all_rois


def apply_roi_mask(image, points, crop_to_bbox=True, save_crop_path=None):
    """
    Applies a ROI polygon mask to an image.
    Pixels outside the polygon are set to 0.

    Parameters:
        image: np.ndarray (2D grayscale/binary or 3D BGR)
        points: list or np.ndarray of (x, y) coordinates
        crop_to_bbox: bool, if True crops the image to the bounding box of polygon
        save_crop_path: str, optional path to save the cropped masked image

    Returns:
        masked_image: np.ndarray
    """
    import os
    import numpy as np

    if points is None or len(points) < 3:
        return image

    pts = np.array(points, dtype=np.int32)
    h, w = image.shape[:2]

    # Clip points to image boundaries
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    if crop_to_bbox:
        cropped = image[y_min:y_max + 1, x_min:x_max + 1].copy()
        crop_pts = pts.copy()
        crop_pts[:, 0] -= x_min
        crop_pts[:, 1] -= y_min

        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [crop_pts], 255)

        masked = cv2.bitwise_and(cropped, cropped, mask=mask)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(image, image, mask=mask)

    if save_crop_path:
        os.makedirs(os.path.dirname(save_crop_path), exist_ok=True)
        cv2.imwrite(save_crop_path, masked)

    return masked


if __name__ == "__main__":
    os.makedirs("rois", exist_ok=True)
    args = argparse.ArgumentParser(
        description="Draw polygon on image"
    )
    args.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to image"
    )
    args = args.parse_args()
    csv_path = "rois/" + args.image.split("/")[-1].split(".")[0] + ".csv"
    draw_polygon_and_save(
        args.image,
        csv_path
    )