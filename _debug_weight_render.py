"""Quick diagnostic for weight image rendering."""
import sys, torch
sys.path.insert(0, '.')
from pipeline.weight_map import render_weight_image

from wav_ml_models import TinyConvClassifier
model = TinyConvClassifier(num_classes=10, base_ch=96, max_ch=512, context_blocks=12)
sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
keys = list(sd.keys())

# Render with typical viewer target
H = 2 * 256 + 192 + 24  # 728
W = 256
rgb_tall, meta_tall = render_weight_image(
    sd,
    parameter_keys=keys,
    target_width=W,
    target_height=H,
    mode="architectural_tall",
)
rgb_groups, meta_groups = render_weight_image(
    sd,
    parameter_keys=keys,
    target_width=W,
    target_height=H,
    mode="parameter_groups",
)
print(f"tall: shape={rgb_tall.shape} meta={meta_tall}")
print(f"groups: shape={rgb_groups.shape} meta={meta_groups}")
