# HEQC

HEQC detects five types of artifacts in H&E whole-slide images: foreign material, out-of-focus regions, air bubbles and slide-edge artifacts, tissue folds and wrinkles, and pen marks. It assigns multiple labels to each image patch and produces class-specific heatmaps, an overall heatmap, and prediction summaries.

The inference pipeline supports ResNet-18, EfficientNet-B0, and MobileNetV3-Large. Use the supplied weights and thresholds; no retraining or threshold fitting is required.

## Installation

Install the dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

For NVIDIA GPU inference, first install a compatible CUDA-enabled PyTorch and torchvision pair using the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then run the command above. OpenSlide is included through `openslide-python` and `openslide-bin`.

## Models

Place the supplied model files and their corresponding threshold files in `model/`:

| Model | Weights | Thresholds |
| --- | --- | --- |
| EfficientNet-B0 | `efficientnet_b0.pt` | `threshold_efficientnet_b0.json` |
| ResNet-18 | `resnet18.pt` | `threshold_resnet18.json` |
| MobileNetV3-Large | `mobilenet_v3_large.pt` | `threshold_mobilenet_v3_large.json` |

Each model has two threshold profiles, `balanced` and `recall`. Keep each weight file paired with its supplied threshold JSON.

## Usage

1. Place input slides directly in `wsi/`. Supported extensions are `.svs`, `.ndpi`, `.mrxs`, `.tif`, and `.tiff`; the files must also be readable by OpenSlide. Slide filenames must have unique stems.
2. Select a model and threshold profile in `src/config.py`:

   ```python
   MODEL_NAME = "efficientnet_b0"
   THRESHOLD_PROFILE = "balanced"
   ```

3. Run from the project root:

   ```bash
   python scripts/run_pipeline.py
   ```

Each run processes the slides in `wsi/` using the selected model and profile. Subdirectories are not searched.

## Configuration

Common settings in `src/config.py` are listed below. Other settings can be adjusted in the same file.

| Setting | Default | Description |
| --- | --- | --- |
| `TARGET_MAGNIFICATION` | `10.0` | Target magnification for selecting a WSI pyramid level |
| `STRIDE` | `256` | Patch spacing at the selected level |
| `DEVICE` | `"auto"` | Select CUDA, Apple MPS, or CPU automatically |
| `BATCH_SIZE` | `64` | Patches per inference batch; reduce if GPU memory is insufficient |
| `NUM_WORKERS` | `8` | Data-loading workers; set to `0` when troubleshooting worker errors |
| `HEATMAP_MAGNIFICATION` | `1.25` | Heatmap image resolution |
| `THUMBNAIL_MAGNIFICATION` | `1.25` | Standalone thumbnail resolution |
| `OVERWRITE` | `False` | Allow replacement of existing slide results |

Patches are 512 × 512 pixels. The default stride of 256 gives overlapping patches; use 512 for non-overlapping patches. The pipeline selects the available pyramid level closest to the target magnification. It warns when the selected MPP is outside the recommended range of 0.85–1.15 µm/px and continues processing.

If both objective-power and MPP metadata are unavailable, set `MANUAL_OBJECTIVE_POWER` or `MANUAL_BASE_MPP` from known slide information. Increasing thumbnail magnification affects only the standalone thumbnail and can substantially increase memory use and file size.

## Outputs

Results are saved to `outputs/<model_name>/<slide_id>/`:

| Output | Contents |
| --- | --- |
| `images/thumbnail_1p25x.png` | WSI thumbnail at the default magnification |
| `images/heatmap_01_foreign.png` | Foreign-material heatmap |
| `images/heatmap_02_out_of_focus.png` | Out-of-focus heatmap |
| `images/heatmap_03_air_bubble.png` | Air/bubble heatmap |
| `images/heatmap_04_fold_wrinkle.png` | Fold/wrinkle heatmap |
| `images/heatmap_05_pen.png` | Pen/ink heatmap |
| `images/heatmap_overall.png` | Combined heatmap |
| `label_statistics.csv` | Per-class prediction statistics |
| `statistics.json` | Slide information, settings, statistics, and timing |

`run_summary.csv` is saved in the model's output directory. Set `SAVE_PATCH_PREDICTIONS = True` to also save patch coordinates and probabilities in `patch_predictions.npz`.

Both profiles use the same output path. To retain results from both, move the first run's results outside `outputs/` before switching profiles. With `OVERWRITE = False`, matching completed results are reused and conflicting results are not replaced. With `OVERWRITE = True`, the previous generated directory for that slide is replaced.

## Notes

- Heat intensity is relative to each class threshold. Heatmaps summarize patch predictions and do not represent exact artifact boundaries.
- Statistics are calculated over the full slide grid, including background. Reported fractions are not measurements of affected tissue area.
- The study WSIs and annotations are not publicly available. Training code is not included.
- HEQC is intended for research use, not clinical diagnosis.

## License

The code, model weights, and threshold files are available for non-commercial research under the [HEQC Non-Commercial Research License](LICENSE). Commercial use requires prior written permission.

For commercial licensing inquiries, open an issue in the original HEQC repository to request contact with the project author.
