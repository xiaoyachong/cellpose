from cellpose import models
import tifffile
from pathlib import Path
import numpy as np

# Define paths
image_path = "./20260221_155821_petiole36_00005.tiff"
model_path = "/pscratch/sd/x/xchong/3RSE/832/cellpose/models/petiole_model_flow0"

# Load image
print("Loading image...")
img = tifffile.imread(image_path)
print(f"Loaded {Path(image_path).name}")

# Load model and run inference
print("Running inference...")
model = models.CellposeModel(gpu=True, pretrained_model=model_path)

masks, flows, styles = model.eval([img], batch_size=1)
mask   = masks[0]
flow   = flows[0]   # list: [flow_rgb, flow_xy, cell_prob, ...]
style  = styles[0]  # 1-D style vector (numpy array)

print("Inference complete!")
print(f"{Path(image_path).name}: {mask.max()} cells")

stem = "20260221_155821_petiole36_00005"

# --- Save mask ---
mask_path = f"{stem}_mask.tiff"
tifffile.imwrite(mask_path, mask.astype("uint16"))
print(f"Mask saved to: {mask_path}")

# --- Save flows ---
# flow[0] : RGB flow image  (H x W x 3, uint8)
# flow[1] : dX/dY flow      (2 x H x W, float32)
# flow[2] : cell probability (H x W,     float32)
flow_rgb_path  = f"{stem}_flow_rgb.tiff"
flow_xy_path   = f"{stem}_flow_xy.tiff"
flow_prob_path = f"{stem}_flow_cellprob.tiff"

tifffile.imwrite(flow_rgb_path,  flow[0].astype("uint8"))
tifffile.imwrite(flow_xy_path,   flow[1].astype("float32"))
tifffile.imwrite(flow_prob_path, flow[2].astype("float32"))
print(f"Flows saved to: {flow_rgb_path}, {flow_xy_path}, {flow_prob_path}")

# --- Save style vector ---
style_path = f"{stem}_style.npy"
np.save(style_path, style)
print(f"Style vector saved to: {style_path}")