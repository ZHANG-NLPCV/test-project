# Adapting this repo to the SC-Track / pcnaDeep benchmark

This connects our strict binary baseline to the Chan lab evaluation protocol
(`chan-labsite/SC-Track`, `chan-labsite/SC-Track-evaluation`) so results are
comparable on **their** metrics (IDF1, MOTA, CDF1, CTC SEG/TRA) instead of
binary Dice.

## The good news: their data == our data, and the GT is public (CC-BY 4.0)

The published GT tracking tables use the exact samples we train on
(`MCF10A_copy02`, `MCF10A_copy11`, `copy_of_1_xy01`, `copy_of_1_xy19`, `src06`).

- **GT tracking + analysis** — Zenodo record **10441055** (DOI 10.5281/zenodo.8284986):
  `tracking results.zip`, `deep learning models.zip`, `evaluate_results.zip`, `demo.zip`.
- **pcnaDeep pretrained + MCF10A** — Zenodo record **5515771**:
  `mrcnn_sat_rot_aug.pth`, `MCF10A.rar`, `demo.rar`.
- **CTC benchmark** — celltrackingchallenge.net/2d-datasets/ (silver masks + GT tracking).

So the benchmark line needs **no data request to Chan** — only download.

## The track-table schema (verbatim from SC-Track-evaluation source)

Both GT and prediction are per-detection CSV rows. The tracker (`cell_id`)
carries identity across frames.

| Use | Required columns |
| --- | --- |
| `evaluate-MOT.py` (IDF1/MOTA) | `track_id, cell_id, frame_index, center_x, center_y` — matched per frame by centroid Euclidean distance via `motmetrics` |
| `prepare_TRA_compute_data.py` (CTC SEG/TRA) | `frame_index, cell_id, parent_id, mask_of_x_points, mask_of_y_points` — `parent_id == cell_id` marks a track root; polygons are rasterised to CTC `man_track###.tif` + lineage `.txt` |

`track.csv` with these columns is exactly what **SC-Track emits**. We do not
hand-build it — we feed instance masks to SC-Track and it produces `track.csv`.

## Pipeline

```text
[this repo]                          [SC-Track]            [SC-Track-evaluation]
patch binary preds                    track.csv             IDF1 / MOTA  (evaluate-MOT.py)
   |  export_sctrack_masks.py            ^                   CTC SEG/TRA  (prepare_TRA_compute_data.py
   v  (stitch -> full-frame instances)   |                                + CTC-EvaluationSoftware)
<sample>.tif  --------------------------/
(grayscale multi-TIFF, instances per frame)
```

### Step 1 — export SC-Track input from our model (this repo)

`scripts/export_sctrack_masks.py` runs the trained FrameUNet over every patch,
stitches the binary foreground back to full frames, instance-labels each full
frame (watershed by default), and writes one `<sample_id>.tif` per movie under
`<out-dir>/sctrack_input/`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_sctrack_masks.py \
  --checkpoint /home/hjk4090d/microscopy_project/runs/b0_frameunet_strict_baseline/frameunet_strict_best.pt \
  --splits train val test \
  --method watershed --min-area 20 \
  --out-dir /home/hjk4090d/microscopy_project/runs/b0_frameunet_strict_baseline
```
Ends with `EXPORT_SCTRACK_MASKS_OK`. (Use `--method cc` for the merge-everything
floor; compare the two to see how much watershed separation buys.)

### Step 2 — run SC-Track (their repo)

```bash
pip install SC-Track          # or clone chan-labsite/SC-Track
sctrack -p <out-dir>/sctrack_input/MCF10A_copy02.tif   # -> track.csv
```

### Step 3 — evaluate against public GT

Get GT tables from Zenodo 10441055 `tracking results.zip` (per sample), then:

```bash
git clone https://github.com/chan-labsite/SC-Track-evaluation
pip install motmetrics imagesize          # added to this repo's requirements.txt
python SC-Track-evaluation/evaluate-MOT.py            # IDF1 / MOTA  vs track-GT.csv
python SC-Track-evaluation/prepare_TRA_compute_data.py # build CTC RES/GT, then run CTC-EvaluationSoftware
```

## Known gaps / decisions before the numbers are meaningful

- **Identity comes from SC-Track, not from us.** Our model is per-frame; IDF1/MOTA
  measure the tracker. To claim a Plan B contribution, compare
  `our-masks -> SC-Track` against `pcnaDeep/Cellpose -> SC-Track` on the same GT.
- **Phase / CDF1 not yet possible.** No cell-cycle phase head in the binary model.
  Needs a phase output (the interior+boundary / multi-task step) before
  `evaluate-CDF1.py` / `evaluate-phase-classification.py` apply.
- **Cross-movie split.** The strict split is temporal within the same movies;
  for a generalization claim, evaluate on held-out movies / CTC datasets.
- **Watershed is a stopgap** for instance separation. If `cc` vs `watershed`
  (see `eval_instance_strict.py`) shows watershed does not close the high-density
  gap, that motivates a real instance head rather than post-processing.
