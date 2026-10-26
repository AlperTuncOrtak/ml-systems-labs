from common import set_seeds, RESULTS
import copy
import pandas as pd
import torch
from thop import profile
from models import PatchAttention

set_seeds()
model = PatchAttention().eval()
x = torch.randn(1, 1, 28, 28)
hand = 2_329_856

rows = []
for fast in [True, False]:
    torch.backends.mha.set_fastpath_enabled(fast)
    macs, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
    rows.append({"mha_fastpath_enabled": fast, "thop_MACs": int(macs),
                 "thop_FLOPs_x2": int(2 * macs),
                 "pct_of_hand": round(100 * 2 * macs / hand, 1)})
torch.backends.mha.set_fastpath_enabled(True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "thop_fastpath_check.csv", index=False)
print(df.to_string(index=False))
