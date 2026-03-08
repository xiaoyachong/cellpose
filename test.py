import os
import numpy as np
from pathlib import Path
from cellpose import models
import tifffile


# Configuration
input_folder       = "/pscratch/sd/x/xchong/sam3_finetune/seg_annotation_pipeline2/data/images"
output_folder      = "./original_cellpose"
model_type         = "cyto3"
channels           = [0, 0]
diameter           = None
flow_threshold     = 0.4
cellprob_threshold = 0.0

os.makedirs(output_folder, exist_ok=True)

# ── collect files ──────────────────────────────────────────────────────────────
tiff_extensions = ['*.tiff', '*.tif', '*.TIF', '*.TIFF']
image_files = []
for ext in tiff_extensions:
    image_files.extend(Path(input_folder).glob(ext))

image_files = sorted([str(f) for f in image_files])
print(f"Found {len(image_files)} TIFF files")

target_file = "20260212_133951_petiole30_00100.tiff"
image_files = [f for f in image_files if os.path.basename(f) == target_file]

print(f"Processing {len(image_files)} file(s):")
for f in image_files:
    print(f"  - {os.path.basename(f)}")

# ── model ──────────────────────────────────────────────────────────────────────
model = models.CellposeModel(gpu=True, model_type=model_type)
print(f"Loaded {model_type} model | GPU: {model.gpu}")

# ── helper: sanitize image ─────────────────────────────────────────────────────
def prepare_image(img):
    """Ensure image is 2-D float32 grayscale with sane dimensions."""

    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.mean(axis=-1)

    if img.ndim == 3:
        img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]

    img = img.astype(np.float32)

    # pad to multiples of 8
    h, w = img.shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8

    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
        print(f"  Padded to {img.shape} (was {h}x{w})")

    return img, h, w


# ── inference ──────────────────────────────────────────────────────────────────
results = []

for i, img_path in enumerate(image_files):

    print(f"\nProcessing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")

    try:

        img = tifffile.imread(img_path)
        print(f"  Raw shape: {img.shape}, dtype: {img.dtype}")

        img, orig_h, orig_w = prepare_image(img)
        print(f"  Prepared shape: {img.shape}")

        masks, flows, styles = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

        masks = masks[:orig_h, :orig_w]

        n_cells = len(np.unique(masks)) - 1
        print(f"  Detected {n_cells} cells")

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        tifffile.imwrite(
            os.path.join(output_folder, f"{base_name}_masks.tif"),
            masks.astype(np.uint16)
        )

        results.append({
            'filename': os.path.basename(img_path),
            'n_cells': n_cells
        })

    except Exception as e:

        print(f"  ERROR: {e}")

        results.append({
            'filename': os.path.basename(img_path),
            'n_cells': 'ERROR'
        })

print("\n=== Processing Complete ===")
for r in results:
    print(r)