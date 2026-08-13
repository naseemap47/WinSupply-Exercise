import json
from pathlib import Path
import cv2
import csv
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx


def cropped_polygon(
    image_path,
    csv_path,
):
    """
    Crop an image using polygon points stored in CSV.

    Returns:
        polygon_crop:
            Original-resolution BGR image.
            Pixels outside the polygon are white.

        crop_points:
            Polygon points in crop-local coordinates.
    """

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    image_h, image_w = image.shape[:2]

    # Read polygon points
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

    # Keep points inside image
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

    # Bounding box
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])

    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    # Crop
    cropped = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ].copy()

    # Convert polygon points to crop coordinates
    crop_points = points.copy()

    crop_points[:, 0] -= x_min
    crop_points[:, 1] -= y_min

    # Create polygon mask
    mask = np.zeros(
        cropped.shape[:2],
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [crop_points],
        255
    )

    # Keep polygon area
    polygon_crop = np.full_like(
        cropped,
        255
    )

    polygon_crop[mask == 255] = cropped[mask == 255]

    return polygon_crop, crop_points


def preprocess(
    gray,
    threshold=220,
):
    """
    Preprocess grayscale engineering drawing.

    Returns:
        binary image:
            pipelines/lines = 255
            background = 0
    """

    # Reduce small scanning noise
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Threshold dark engineering lines
    _, binary = cv2.threshold(
        blurred,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Remove tiny isolated noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    return binary


def skeletonize_pipeline(
    pipeline_mask,
):
    """
    Convert pipeline mask into one-pixel-wide skeleton.
    """

    binary = (
        pipeline_mask > 0
    )

    skeleton = skeletonize(
        binary
    )

    return skeleton


def calculate_pixel_degree(
    skeleton,
):
    """
    Calculate 8-neighbor degree for every skeleton pixel.
    """

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8
    )

    degree = cv2.filter2D(
        skeleton.astype(np.uint8),
        cv2.CV_16U,
        kernel
    )

    return degree


def detect_graph_nodes(skeleton, degree):
    """
    Detect graph nodes from a skeleton.

    A node is:
        degree == 1  -> endpoint
        degree >= 3  -> junction

    Adjacent endpoint/junction pixels are grouped into one
    logical graph node.

    Returns
    -------
    node_labels : dict
        {
            (x, y): node_id
        }

    endpoints : list
        [(x, y), ...]

    junctions : list
        [(x, y), ...]
    """

    skeleton = (skeleton > 0).astype(np.uint8)

    # 1. Endpoint mask
    endpoint_mask = (
        (skeleton > 0) &
        (degree == 1)
    ).astype(np.uint8)

    # 2. Junction mask
    junction_mask = (
        (skeleton > 0) &
        (degree >= 3)
    ).astype(np.uint8)

    # 3. Group connected endpoint pixels
    num_endpoints, endpoint_labels, endpoint_stats, endpoint_centroids = \
        cv2.connectedComponentsWithStats(
            endpoint_mask,
            connectivity=8
        )

    # 4. Group connected junction pixels
    num_junctions, junction_labels, junction_stats, junction_centroids = \
        cv2.connectedComponentsWithStats(
            junction_mask,
            connectivity=8
        )

    # 5. Output structures
    node_labels = {}

    endpoints = []
    junctions = []

    node_id = 0

    # Find nearest skeleton pixel to centroid
    skeleton_yx = np.column_stack(
        np.where(skeleton > 0)
    )

    def nearest_skeleton_pixel(cx, cy):

        if len(skeleton_yx) == 0:
            return None

        # skeleton_yx contains:
        # [y, x]

        dy = skeleton_yx[:, 0] - cy
        dx = skeleton_yx[:, 1] - cx

        distances = (
            dx * dx +
            dy * dy
        )

        index = np.argmin(distances)

        y = int(
            skeleton_yx[index, 0]
        )

        x = int(
            skeleton_yx[index, 1]
        )

        return (x, y)

    # 6. Endpoints
    for label in range(
        1,
        num_endpoints
    ):

        cx, cy = endpoint_centroids[label]

        pixel = nearest_skeleton_pixel(
            cx,
            cy
        )

        if pixel is None:
            continue

        node_labels[pixel] = node_id

        endpoints.append(pixel)

        node_id += 1

    # 7. Junctions
    for label in range(
        1,
        num_junctions
    ):

        cx, cy = junction_centroids[label]

        pixel = nearest_skeleton_pixel(
            cx,
            cy
        )

        if pixel is None:
            continue

        # Avoid duplicate node
        if pixel in node_labels:
            continue

        node_labels[pixel] = node_id

        junctions.append(pixel)

        node_id += 1

    return (
        node_labels,
        endpoints,
        junctions
    )


def build_pixel_graph(skeleton):
    """
    Build a pixel-level graph from a binary skeleton.

    Each foreground pixel is a graph node.

    8-connectivity is used so diagonal skeleton pixels are
    connected as well.

    Edge weight:
        1.0            horizontal / vertical
        sqrt(2)        diagonal
    """

    G = nx.Graph()

    # Get skeleton pixels
    ys, xs = np.where(skeleton > 0)

    skeleton_pixels = set(
        zip(xs, ys)
    )

    # Add nodes
    for x, y in skeleton_pixels:

        G.add_node(
            (x, y)
        )

    # 8-neighborhood
    # Only use forward neighbors to avoid adding
    # the same edge twice.
    neighbors = [
        (1, 0),
        (0, 1),
        (1, 1),
        (-1, 1),
    ]

    # Add edges
    for x, y in skeleton_pixels:

        for dx, dy in neighbors:

            nx_ = x + dx
            ny_ = y + dy

            neighbor = (
                nx_,
                ny_
            )

            if neighbor not in skeleton_pixels:
                continue

            # Euclidean pixel distance
            if dx == 0 or dy == 0:
                length = 1.0
            else:
                length = np.sqrt(2.0)

            G.add_edge(
                (x, y),
                neighbor,
                length=length
            )

    return G


def extract_directional_candidates(
    binary,
    horizontal_length=50,
    vertical_length=50,
):
    """
    Extract long horizontal and vertical line structures.

    Returns:
        horizontal
        vertical
    """

    # Horizontal lines
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        h_kernel
    )

    # Vertical lines
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length)
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        v_kernel
    )

    return horizontal, vertical


def make_single_centerline_mask(
    binary,
    horizontal_length=50,
    vertical_length=50,
    gap_size=30,
):
    """
    Convert double-line pipeline representations into
    filled regions suitable for skeletonization.
    """

    # 1. Extract horizontal structures
    horizontal, vertical = extract_directional_candidates(
        binary,
        horizontal_length=horizontal_length,
        vertical_length=vertical_length
    )

    # 2. Fill horizontal pipe boundaries
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, gap_size)
    )

    horizontal_filled = cv2.morphologyEx(
        horizontal,
        cv2.MORPH_CLOSE,
        h_kernel
    )

    # 3. Fill vertical pipe boundaries
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (gap_size, 3)
    )

    vertical_filled = cv2.morphologyEx(
        vertical,
        cv2.MORPH_CLOSE,
        v_kernel
    )

    # 4. Combine
    filled = cv2.bitwise_or(
        horizontal_filled,
        vertical_filled
    )

    # 5. Small morphological cleanup
    cleanup_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    filled = cv2.morphologyEx(
        filled,
        cv2.MORPH_CLOSE,
        cleanup_kernel
    )

    return filled

def calculate_pipeline_results(
    pixel_graph,
    node_labels
):
    """
    Calculate length and node count for every connected
    pipeline component.
    """

    results = {}

    components = list(
        nx.connected_components(
            pixel_graph
        )
    )

    # Map pixel -> logical node

    component_nodes = {}

    for pixel, node_id in node_labels.items():

        for component_index, component in enumerate(
            components
        ):

            if pixel in component:

                component_nodes.setdefault(
                    component_index,
                    set()
                ).add(
                    node_id
                )

                break

    # Calculate component lengths

    pipeline_id = 1

    for component_index, component in enumerate(
        components
    ):

        # Ignore tiny components
        if len(component) < 10:
            continue

        subgraph = pixel_graph.subgraph(
            component
        )

        total_length = 0.0

        for _, _, data in subgraph.edges(
            data=True
        ):

            total_length += data.get(
                "length",
                1.0
            )

        nodes = component_nodes.get(
            component_index,
            set()
        )

        results[
            f"pipeline_{pipeline_id}"
        ] = {
            "length": round(
                total_length,
                2
            ),
            "nodes": len(nodes)
        }

        pipeline_id += 1

    return results

def save_json(
    results,
    output_path,
):
    """
    Save pipeline results to JSON.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

def draw_skeleton_overlay(
    image,
    skeleton,
    endpoints=None,
    junctions=None,
):
    """
    Draw skeleton and graph nodes on original image.
    """

    overlay = image.copy()

    # Draw centerline
    ys, xs = np.where(skeleton)

    overlay[ys, xs] = (0, 255, 0)

    # Endpoints
    if endpoints is not None:

        for x, y in endpoints:

            cv2.circle(
                overlay,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

    # Junctions
    if junctions is not None:

        for x, y in junctions:

            cv2.circle(
                overlay,
                (x, y),
                8,
                (255, 0, 0),
                -1
            )

    return overlay

def main(
    image_path,
    csv_path,
    output_json,
    debug_dir
):

    # Crop ROI
    plan, crop_points = cropped_polygon(
        image_path,
        csv_path
    )

    # Grayscale
    gray = cv2.cvtColor(
        plan,
        cv2.COLOR_BGR2GRAY
    )

    # Preprocessing
    binary = preprocess(
        gray,
        threshold=220
    )

    filled = make_single_centerline_mask(
        binary,
        horizontal_length=50,
        vertical_length=50,
        gap_size=30
    )

    # Skeletonization
    skeleton = skeletonize_pipeline(
        filled
    )

    # Pixel graph
    pixel_graph = build_pixel_graph(
        skeleton
    )

    # Pixel degree
    degree = calculate_pixel_degree(
        skeleton
    )

    # Detect graph nodes
    node_labels, endpoints, junctions = detect_graph_nodes(
        skeleton,
        degree
    )

    results = calculate_pipeline_results(
        pixel_graph,
        node_labels
    )

    # Save JSON
    save_json(results, output_json)

    # Debug image
    overlay = draw_skeleton_overlay(
        plan,
        skeleton,
        endpoints,
        junctions
        )

    # Save debug images
    if debug_dir:

        debug_dir = Path(debug_dir)
        debug_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        stem = Path(
            image_path
        ).stem

        cv2.imwrite(
            str(
                debug_dir /
                f"01_{stem}_binary.png"
            ),
            binary
        )

        cv2.imwrite(
            str(
                debug_dir /
                f"02_{stem}_filled.png"
            ),
            filled
        )

        cv2.imwrite(
            str(
                debug_dir /
                f"03_{stem}_skeleton.png"
            ),
            skeleton.astype(
                np.uint8
            ) * 255
        )

        cv2.imwrite(
            str(
                debug_dir /
                f"04_{stem}_overlay.png"
            ),
            overlay
        )

if __name__ == "__main__":
    main(
        image_path = "sheets/sheet_02.png",
        csv_path="rois/sheet_02.csv",
        output_json="runs/sheet_02.json",
        debug_dir="debug"
    )