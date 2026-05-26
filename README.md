# BHSD 3D hemorrhage segmentation templates

PyTorch templates for **multi-class 3D CT segmentation** on the [BHSD](https://github.com/WonderLandxR/BHSD) dataset layout: **Mamba 3D U-Net** vs **pure 3D U-Net** baseline.

## Files

| File | Description |
|------|-------------|
| `mamba3d_unet_template.py` / `.ipynb` | Tri-oriented Mamba + 3D U-Net |
| `pure3d_unet_template.py` / `.ipynb` | Pure 3D U-Net (conv only) |
| `test_mamba_unweighted_dice.py` | TEST eval, unweighted Dice → `mamba_test_unweighted_dice_results.txt` |
| `test_pure_unweighted_dice.py` | Same for Pure → `pure_test_unweighted_dice_results.txt` |

## Dataset layout

Set `SEG_DATASET_ROOT` to a folder containing:

```
SEG_DATASET_ROOT/
  train_images/  train_masks/
  val_images/    val_masks/
  test_images/   test_masks/
```

NIfTI volumes (`.nii`). Masks are integer labels `0` (background) and `1–5` (EDH, ICH, IVH, SAH, SDH).

## Dependencies

```
torch
numpy
nibabel
patchify
scikit-image
tqdm
```

## Training

```powershell
cd github_templates
set SEG_DATASET_ROOT=C:\path\to\BHSD_split
set CHECKPOINT_DIR=.\checkpoints_mamba
python mamba3d_unet_template.py
```

Pure baseline:

```powershell
set CHECKPOINT_DIR=.\checkpoints_pure_unet
python pure3d_unet_template.py
```

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `SEG_DATASET_ROOT` | `./data/segmentation` | BHSD root (see layout above) |
| `CHECKPOINT_DIR` | `./checkpoints_mamba` or `./checkpoints_pure_unet` | Saves `best_model.pt`, `latest_model.pt`, `training_history.pkl` |
| `TRAINING_MODE` | `new` | `new` or `continue` (resume from `latest_model.pt`) |
| `EPOCHS` | `5000` | Max epochs (early stopping usually ends sooner) |
| `PATIENCE` | `20` | Early stopping on validation loss |
| `BASE_CHANNELS` | `16` | Model width |
| `MAX_PATIENTS` | all | Limit patients for a quick run |

## TEST evaluation (unweighted Dice)

Paper-comparable **unweighted macro foreground Dice** (merged full volume, `INFER_STEP`):

```powershell
python test_mamba_unweighted_dice.py --checkpoint C:\path\to\best_model.pt --data-root C:\path\to\BHSD_split
python test_pure_unweighted_dice.py --checkpoint C:\path\to\best_model.pt --data-root C:\path\to\BHSD_split
```

Omit flags to be prompted for paths. Results are written next to this README.

## Patch tiling

| Phase | Step | Typical value |
|-------|------|----------------|
| Train | `TRAIN_STEP` | `(128, 128, 4)` |
| Val (merged) | `VAL_STEP` | `(64, 64, 8)` |
| Test (merged) | `INFER_STEP` | `(64, 64, 4)` |

Patch size: `(128, 128, 8, 4)` — four CT windows per voxel.
