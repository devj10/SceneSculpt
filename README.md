# SceneSculpt

Turn a phone video into an editable 3D scene. SceneSculpt reconstructs real-world environments using 3D Gaussian Splatting and lets you select, delete, move, and recolor objects directly in 3D space — no 3D modeling experience required.

---

## What it does

You point your phone at something, walk around it, and SceneSculpt turns that video into a live 3D scene you can edit in your browser. Click on any object to select it, then delete it, paint it a new color, or move it somewhere else. The edits are saved back to a `.ply` file you can reload at any time.

---

## How the pipeline works

```
phone video
    │
    ▼  1. Frame extraction
    │  Sample frames at ~2-4 fps, skip blurry ones
    │
    ▼  2. Structure from Motion  (COLMAP)
    │  Estimate where the camera was for every frame
    │  Output: camera poses + sparse 3D point cloud
    │
    ▼  3. Gaussian Splatting training  (gsplat)
    │  Represent the scene as millions of 3D Gaussians
    │  Each Gaussian has: position, color, opacity, shape
    │  Train 30k iterations to match every input view
    │  Output: splats.ply
    │
    ▼  4. Object segmentation  (SAM)
    │  Run Segment Anything on training frames
    │  Project 2D masks into 3D — majority vote assigns
    │  each Gaussian an object ID
    │  Output: splats_segmented.ply
    │
    ▼  5. Interactive editor  (viser)
       Browser-based 3D viewer
       Click → select entire object
       Delete / Recolor / Translate
```

---

## Quick start

**Requirements:** Python 3.11, conda, [Modal](https://modal.com) account (free tier works), COLMAP on PATH.

```bash
git clone https://github.com/devj10/SceneSculpt
cd SceneSculpt
conda create -n scenesculpt python=3.11 -y && conda activate scenesculpt
pip install viser scipy plyfile numpy Pillow tqdm rich tyro natsort pycolmap opencv-python modal
pip install -e . --no-deps

modal setup   # one-time browser auth
```

**1. Train** (runs on Modal GPU, ~45 min)
```bash
modal run modal_train.py --video my_video.mp4 --scene my_scene --fps 4 --max-frames 150
```

**2. Segment objects** (~10 min, downloads SAM checkpoint once)
```bash
modal run modal_segment.py --scene my_scene
```

**3. Download and edit**
```bash
modal volume get scenesculpt-scenes my_scene/splats_segmented.ply ./splats.ply
python scripts/launch_editor.py --scene splats.ply
# open http://localhost:8080
```

---

## Editor controls

| Action | How |
|---|---|
| Select object | Click on it in the viewport |
| Expand selection | Increase Radius slider, click again |
| Delete | Select → "Delete Selected" |
| Recolor | Select → pick color → "Recolor Selected" |
| Move | Select → set Δx/Δy/Δz → "Translate Selected" |
| Save | "Save Scene" (overwrites the .ply) |

---

## Capture tips

The quality of the reconstruction depends entirely on the video:

- **Walk all the way around** the subject — the more angles, the better
- **Slow and steady** — fast motion causes blur and confuses COLMAP
- **Good lighting** — avoid harsh shadows or overexposed areas
- **30–60 seconds** of footage is enough; longer doesn't help much
- **Static scene** — moving objects (people walking through, leaves blowing) hurt reconstruction

---

## Tech stack

| Component | Library |
|---|---|
| Gaussian Splatting training | [gsplat](https://github.com/nerfstudio-project/gsplat) |
| Structure from Motion | [COLMAP](https://colmap.github.io) + [pycolmap](https://github.com/colmap/pycolmap) |
| Object segmentation | [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) |
| 3D viewer / editor | [viser](https://github.com/nerfstudio-project/viser) |
| Cloud GPU | [Modal](https://modal.com) |
