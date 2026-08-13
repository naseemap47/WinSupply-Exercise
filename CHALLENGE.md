# Pipeline Graph Challenge

**Time:** 1.5–2 hours  
**Format:** Take-home

---

## Problem

You’re given a folder containing **5 engineering drawing images (PNG)** under `sheets/`.

Each drawing contains:

* pipelines
* junctions
* valves
* text
* noise

Write a Python program that:

1. Detects all pipelines.
2. Finds every intersection/junction.
3. Converts the pipelines into a graph.
4. Calculates the total length of each connected pipeline.
5. Outputs (one JSON file per sheet):

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

`length` is in **pixels** (sum of skeleton/edge lengths for that connected component).  
`nodes` is the number of graph nodes (junctions + endpoints) in that component.

---

## Constraints

You may use:

* OpenCV
* NumPy
* NetworkX
* scikit-image

**No deep learning required.**  
No hosted vision APIs / foundation models.

Prefer **one script** that runs on every sheet (global thresholds OK). Do not hardcode per-sheet pixel seed corridors or `if sheet == …` geometry branches.

---

## Deliverables

1. **`runs/`** — one JSON file per sheet you attempted:

```
runs/sheet_01.json
runs/sheet_02.json
…
```

Shape as shown above (`pipeline_1`, `pipeline_2`, …).

2. **`NOTES.md`** — short write-up: preprocessing choices, how you skeletonize / build the graph, what failed per sheet, what you’d do with more time.

3. Preferred: your script (e.g. `trace_pipelines.py`) and optional overlay images.

Optional helpers: `starter.py` (load images, write JSON). You can ignore it.

**Format preview (not the answer):** `example_runs.json`.

---

## What we’re looking for

* Image preprocessing  
* Skeletonization  
* Graph construction  
* Geometry  
* Clean code  
