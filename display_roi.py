import cv2
import numpy as np
import csv


def display_cropped_polygon(
    image_path,
    csv_path,
    fullscreen=True,
):
    """
    Display only the cropped polygon region.

    The polygon coordinates in CSV must be in
    ORIGINAL image coordinates.

    CSV format:
        point_id,x,y
        1,100,200
        2,500,200
        3,500,600
        4,100,600

    Parameters
    ----------
    image_path : str
        Path to original image.

    csv_path : str
        Path to polygon CSV.

    padding : int
        Extra pixels around the polygon.

    fullscreen : bool
        Display crop in fullscreen if True.

    Controls
    --------
    ESC -> Close window
    """

    # ---------------------------------------------------------
    # Load image
    # ---------------------------------------------------------
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    image_h, image_w = image.shape[:2]

    # ---------------------------------------------------------
    # Read polygon points
    # ---------------------------------------------------------
    points = []

    with open(csv_path, "r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            x = int(float(row["x"]))
            y = int(float(row["y"]))

            points.append([x, y])

    if len(points) < 3:
        raise ValueError(
            "Polygon must contain at least 3 points."
        )

    points = np.array(
        points,
        dtype=np.int32
    )

    # ---------------------------------------------------------
    # Make sure points are inside image
    # ---------------------------------------------------------
    points[:, 0] = np.clip(
        points[:, 0],
        0,
        image_w - 1
    )

    points[:, 1] = np.clip(
        points[:, 1],
        0,
        image_h - 1
    )

    # ---------------------------------------------------------
    # Get polygon bounding box
    # ---------------------------------------------------------
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])

    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    # ---------------------------------------------------------
    # Crop image
    # ---------------------------------------------------------
    cropped = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ].copy()

    # ---------------------------------------------------------
    # Convert polygon coordinates to crop coordinates
    # ---------------------------------------------------------
    crop_points = points.copy()

    crop_points[:, 0] -= x_min
    crop_points[:, 1] -= y_min

    # ---------------------------------------------------------
    # Create mask
    # ---------------------------------------------------------
    mask = np.zeros(
        cropped.shape[:2],
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [crop_points],
        255
    )

    # ---------------------------------------------------------
    # Keep only polygon area
    # ---------------------------------------------------------
    polygon_crop = cv2.bitwise_and(
        cropped,
        cropped,
        mask=mask
    )

    # ---------------------------------------------------------
    # Calculate display size
    # ---------------------------------------------------------
    crop_h, crop_w = polygon_crop.shape[:2]

    screen_width = 1920
    screen_height = 1080

    scale = min(
        screen_width / crop_w,
        screen_height / crop_h
    )

    # Don't enlarge if not desired
    scale = min(scale, 1.0)

    display_w = int(crop_w * scale)
    display_h = int(crop_h * scale)

    if scale != 1.0:

        display = cv2.resize(
            polygon_crop,
            (display_w, display_h),
            interpolation=cv2.INTER_AREA
        )

    else:

        display = polygon_crop
    
    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------
    window_name = "Cropped Polygon"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    if fullscreen:

        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    cv2.imshow(
        window_name,
        display
    )

    print(
        f"Original image: {image_w} x {image_h}"
    )

    print(
        f"Polygon bounding box: "
        f"({x_min}, {y_min}) -> ({x_max}, {y_max})"
    )

    print(
        f"Cropped size: "
        f"{crop_w} x {crop_h}"
    )

    print("\nPress ESC to close.")

    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cv2.destroyAllWindows()


display_cropped_polygon(
    image_path="sheets/sheet_02.png",
    csv_path="rois/sheet_02.csv",
)