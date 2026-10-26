from common import set_seeds, RESULTS
import copy
import statistics
import time

import pandas as pd
import torch
from thop import profile
from torchinfo import summary
from models import get_models

set_seeds()
torch.set_num_threads(1)  # explicit; matches OMP_NUM_THREADS=1


def hand_flops(name):
    """Per-layer FLOPs (2 x multiply-adds) with the Ch. 4 formulas.
    Bias, activations, pooling, softmax and LayerNorm are ignored."""
    if name == "MLP":
        return [(f"Linear {a}->{b}", "Linear", 2 * a * b)
                for a, b in [(784, 256), (256, 128), (128, 10)]]
    if name == "CNN":
        return [
            ("Conv1 1->16 @28x28", "Conv2d", 2 * 3 * 3 * 1 * 16 * 28 * 28),
            ("Conv2 16->32 @14x14", "Conv2d", 2 * 3 * 3 * 16 * 32 * 14 * 14),
            ("Linear 1568->10", "Linear", 2 * 1568 * 10),
        ]
    seq, d, ff = 16, 64, 128
    rows = [("Patch proj 49->64 (x16 patches)", "Linear", 2 * seq * 49 * d)]
    for l in (1, 2):
        rows += [
            (f"L{l} QKV projections", "Attention", 2 * seq * d * 3 * d),
            (f"L{l} scores Q.K^T", "Attention", 2 * seq * seq * d),
            (f"L{l} weights x V", "Attention", 2 * seq * seq * d),
            (f"L{l} output projection", "Attention", 2 * seq * d * d),
            (f"L{l} FFN 64->128->64", "Linear", 2 * seq * d * ff * 2),
        ]
    rows.append(("Head 64->10", "Linear", 2 * d * 10))
    return rows


@torch.inference_mode()
def latency_ms(model, batch, repeats):
    x = torch.randn(batch, 1, 28, 28)
    model(x)  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


layer_rows, summary_rows = [], []
for name, model in get_models().items():
    model.eval()
    params = summary(model, input_size=(1, 1, 28, 28), verbose=0).total_params

    layers = hand_flops(name)
    for layer, kind, flops in layers:
        layer_rows.append({"model": name, "layer": layer, "type": kind, "hand_FLOPs": flops})
    hand_total = sum(f for _, _, f in layers)

    # thop on a copy (it attaches counters to modules); batch 1, grad enabled
    macs, _ = profile(copy.deepcopy(model), inputs=(torch.randn(1, 1, 28, 28),),
                      verbose=False)
    thop_flops = 2 * macs

    lat1 = latency_ms(model, 1, 100)
    lat128 = latency_ms(model, 128, 20)

    summary_rows.append({
        "model": name,
        "params": params,
        "hand_FLOPs": hand_total,
        "thop_MACs": int(macs),
        "thop_FLOPs_x2": int(thop_flops),
        "thop_vs_hand_pct": round(100 * thop_flops / hand_total, 1),
        "latency_b1_ms": round(lat1, 4),
        "latency_b128_ms": round(lat128, 4),
        "per_sample_b128_ms": round(lat128 / 128, 4),
    })

layers_df = pd.DataFrame(layer_rows)
layers_df.to_csv(RESULTS / "flops_by_layer.csv", index=False)
cost_df = pd.DataFrame(summary_rows)
cost_df.to_csv(RESULTS / "cost_summary.csv", index=False)

print("Hand-calculated FLOPs per layer (batch 1):")
print(layers_df.to_string(index=False))
print("\nCost summary (torch threads =", torch.get_num_threads(), "):")
print(cost_df.T.to_string())
