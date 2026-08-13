# Pipeline Extraction Challenge — Notes

## 1. Approach

The goal was to extract pipelines from engineering drawing images, convert them into a graph, and calculate:

- Total pipeline length in pixels
- Number of graph nodes per connected pipeline
- One JSON output per sheet

The solution uses classical computer vision only:

- OpenCV
- NumPy
- scikit-image
- NetworkX

No deep learning models or external vision APIs are used.

The processing pipeline is:

PNG → ROI cropping → Grayscale conversion → Binary thresholding → Pipeline candidate extraction → Connect/fill double pipeline boundaries → Morphological cleanup → Skeletonization → 8-connected pixel graph → Endpoint/junction detection → Connected-component analysis → Pipeline length calculation → JSON output


## 2. ROI Cropping

The engineering drawings contain a large amount of information that is not relevant to pipeline extraction, including:

- Title blocks
- Legends
- Notes
- Key maps
- Dimension information
- Text annotations

A polygon ROI is therefore used to restrict processing to the relevant plan/profile region.

The existing `cropped_polygon()` function:

1. Loads the image.
2. Reads polygon coordinates from CSV.
3. Clips coordinates to the image boundaries.
4. Calculates the polygon bounding box.
5. Crops the image.
6. Creates a polygon mask.
7. Keeps only the selected polygon area.

This substantially reduces the amount of irrelevant information passed to the pipeline extraction stage.


## 3. Image Preprocessing

The cropped image is converted to grayscale.

A binary representation is then generated so that drawing lines become foreground pixels.

The preprocessing is intentionally simple because the drawings are mostly high-contrast black/white engineering drawings.

Morphological operations are used later to remove small gaps and connect nearby line segments.


## 4. Pipeline Candidate Extraction

Engineering drawings often represent a pipe using two approximately parallel lines rather than a single centerline.

For example:

    -------------------------
    -------------------------

If the image is skeletonized directly, these become two separate skeleton lines.

Therefore, the two boundaries need to be connected before skeletonization.

Directional morphology is used to detect long horizontal and vertical structures.

Long horizontal structures are extracted using a horizontal morphological kernel, and vertical structures are extracted using a vertical kernel.


## 5. Converting Double Pipeline Lines to a Single Centerline

One of the main challenges encountered was that the initial skeleton contained two parallel lines for a single pipe.

Instead of skeletonizing these boundaries directly, morphological closing is used to connect the two boundaries.

Conceptually:

Before:

    -------------------------
    -------------------------

After filling:

    █████████████████████████
    █████████████████████████

Skeletonization then produces approximately:

    -------------------------


This provides a single centerline that can be used for graph construction.

The gap size is controlled by the morphology parameters and is intended to be large enough to connect the two boundaries while avoiding excessive merging of unrelated drawing elements.


## 6. Skeletonization

The filled pipeline mask is skeletonized using `skimage.morphology.skeletonize`.

The purpose of skeletonization is to reduce each pipeline region to approximately one-pixel-wide centerlines.

The resulting skeleton is suitable for graph construction because each foreground pixel represents a point along the pipeline.


## 7. 8-Connected Pixel Graph

The skeleton is converted into a NetworkX graph.

Each skeleton pixel becomes a graph node.

8-connectivity is used instead of 4-connectivity because skeletons can contain diagonal connections.

The neighborhood is:

    (-1,-1)   (0,-1)   (1,-1)
    (-1, 0)      X     (1, 0)
    (-1, 1)   (0, 1)   (1, 1)

Edges are created between neighboring skeleton pixels.

The edge length is:

- Horizontal/vertical connection = 1 pixel
- Diagonal connection = sqrt(2) pixels

Therefore the pipeline length is based on Euclidean pixel distance rather than simply counting skeleton pixels.


## 8. Endpoint and Junction Detection

The number of neighboring skeleton pixels is used to determine the degree of each skeleton pixel.

The basic rules are:

    degree == 1
        → endpoint

    degree == 2
        → normal pipeline pixel

    degree >= 3
        → junction candidate

Adjacent junction pixels are grouped into a single logical junction.

Similarly, endpoint pixels are grouped where necessary.

The final graph-node representation contains:

    {
        (x, y): node_id
    }

This allows the logical graph nodes to be associated with their corresponding skeleton coordinates.


## 9. Connected Pipeline Detection

The skeleton graph is divided into connected components using NetworkX.

Each connected component represents a candidate connected pipeline network.

For every component:

1. Collect all graph edges.
2. Sum their pixel lengths.
3. Count logical endpoints and junctions belonging to that component.
4. Store the result as one pipeline entry.

For example:

```json
{
    "pipeline_1": {
        "length": 534.2,
        "nodes": 18
    },
    "pipeline_2": {
        "length": 288.4,
        "nodes": 9
    }
}
```
## Failed Cases

1. On sheet 3, I am able to extract the curved part, but for the 45 degree angle pipe, it breaks into multiple small segments.
2. Need to find best threshold values to improve the extraction accuracy. but not sure same values will works for all sheets.
3. If the pipe are in different angles other that 90 or 180, the extraction may not be accurate.
4. If the ROI image is not cropped properly, it may extract other elements in the image.
5. If we pass full images to the pipeline extraction algorithm, it will extract other elements in the image.
6. If this solution need roi marking manually, it will not be a good solution for a real time application.
7. for joint angles, we can use angle detection algorithm to detect the angle between two lines, but it requires tuning threshold values to improve the accuracy.

## Improvements
If we have more time for R&D and experiments, we could explore the following options:

1. Train detection/segmentation models for find the ROI of the sheets. so that we can crop the ROI of the sheets automatically.
2. Improve the pipeline extraction accuracy by using deep learning models.
3. Tune the parameters of the pipeline extraction algorithm to improve the accuracy.
4. Add support for other formats of engineering drawings.
5. We can try keypoint detection model to detect pipelines, if this works we no need of skeletonization. 
    And also this is good for 45 degree angle pipes and curved pipes.
6. Try to use Hough lines for 45 degree angle pipes and vertical pipes, but it requires tuning threshold values to improve the accuracy.
