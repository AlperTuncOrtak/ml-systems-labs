from common import set_seeds, RESULTS
import torch
from torchinfo import summary
from models import get_models, PatchAttention

set_seeds()
x = torch.randn(2, 1, 28, 28)

# patch split sanity check: patch 5 (row 1, col 1) must equal pixels [7:14, 7:14]
pa = PatchAttention()
patches = pa.to_patches(x)
assert torch.equal(patches[0, 5], x[0, 0, 7:14, 7:14].reshape(-1)), "patch split wrong"
print("patch split OK:", tuple(patches.shape))

lines = []
for name, model in get_models().items():
    out = model(x)
    s = summary(model, input_size=(1, 1, 28, 28), depth=3, verbose=0,
                col_names=["output_size", "num_params"])
    print(f"{name:<10} output {tuple(out.shape)} | params {s.total_params:,}")
    lines.append(f"===== {name} =====\n{s}\n")

(RESULTS / "torchinfo_summary.txt").write_text("\n".join(lines))
print("Saved torchinfo_summary.txt")
