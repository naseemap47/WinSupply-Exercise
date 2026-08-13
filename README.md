# Project Setup & Execution Guide

This repository contains the image processing and line extraction script to analyze sheet diagrams, extract ROI polygons, crop regions, skeletonize line paths, build line graphs, and export structured JSON outputs.

---

## 🛠️ Environment Requirements & Prerequisites

- **Python**: `>= 3.13`
- Tooling recommendation: [`uv`](https://github.com/astral-sh/uv) (fast Python package installer and resolver) or `pip` / `venv`.

### Core Dependencies
- `opencv-python >= 5.0.0.93`
- `scikit-image >= 0.26.0`
- `networkx >= 3.6.1`
- `numpy`

---

## 🚀 Installation & Setup

### Option 1: Using `uv` (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/naseemap47/WinSupply-Exercise.git
   cd exercise
   ```

2. **Install dependencies and sync environment:**
   ```bash
   uv sync
   ```

### Option 2: Using standard `pip` & `venv`

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install required packages:**
   ```bash
   pip install -r pyproject.toml
   # or explicitly:
   pip install "opencv-python>=5.0.0.93" "scikit-image>=0.26.0" "networkx>=3.6.1" numpy
   ```

---

## 🏃 Running the Application

`main.py` processes image sheets alongside corresponding ROI CSV files, exporting output JSON files and optional debug images.

### Command Line Interface (CLI) Arguments

| Parameter | Short | Required | Description |
|---|---|---|---|
| `--image-dir` | `-i` | **Yes** | Path to the directory containing input sheet images (`.png`). |
| `--csv-dir` | `-c` | **Yes** | Path to the directory containing ROI CSV polygon definitions. |
| `--output-dir` | `-o` | **Yes** | Path to directory where output JSON files will be saved. |
| `--debug-dir` | `-d` | No | Optional directory path to save debug/visualization outputs. |

---

### Example Command

To run the processing pipeline on the default dataset:

#### With `uv`:
```bash
uv run main.py -i sheets/ -c rois/ -o runs/ -d debug/
```

#### With activated Virtual Environment:
```bash
python main.py -i sheets/ -c rois/ -o runs/ -d debug/
```

---

## 🎯 Creating ROI Polygons (`roi.py`)

`roi.py` is an interactive tool for annotating and creating custom region-of-interest (ROI) polygon CSV files for image sheets.

### Interactive Window Controls

| Action | Control |
|---|---|
| **Add Polygon Point** | `Left Click` |
| **Remove Last Point** | `Right Click` |
| **Save & Finish** | `ENTER` key |
| **Exit without saving** | `ESC` key |

### Usage

Run `roi.py` by providing the path to a sheet image:

#### With `uv`:
```bash
uv run roi.py -i sheets/sheet_01.png
```

#### With `python`:
```bash
python roi.py -i sheets/sheet_01.png
```

This will launch an interactive GUI window scaled to your display. Clicking points on the image records polygon vertices, and pressing `ENTER` exports the original-resolution coordinates automatically to `rois/<image_name>.csv`.

---

## 📁 Directory Structure & Expectations

- **Input Sheets (`sheets/`)**: Contains `.png` files (e.g., `sheet_01.png`, `sheet_02.png`).
- **ROI CSVs (`rois/`)**: Contains corresponding `.csv` files defining polygon points (`sheet_01.csv`, `sheet_02.csv`). Each image file must have a matching CSV filename.
- **Output JSON (`runs/`)**: Stores extracted line topology and metadata in JSON format.
- **Debug Directory (`debug/`)**: Stores intermediate crops, masks, and skeleton visualizations when `-d` is specified.

