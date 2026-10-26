import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchAttention(nn.Module):
    """28x28 image -> 16 patches of 7x7 -> Linear 49->64 -> 2 Transformer
    encoder layers -> mean pool -> Linear 64->10.
    A learnable positional embedding is added (not in the lab spec): without it,
    mean pooling makes the model blind to where each patch comes from."""

    def __init__(self, d_model=64, nhead=4, dim_ff=128, num_layers=2):
        super().__init__()
        self.patch = 7
        self.proj = nn.Linear(self.patch * self.patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 16, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_ff,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def to_patches(self, x):
        b = x.shape[0]
        p = self.patch
        x = x.reshape(b, 1, 4, p, 4, p)           # split rows and cols into 4 blocks of 7
        x = x.permute(0, 2, 4, 1, 3, 5)            # (b, 4, 4, 1, 7, 7)
        return x.reshape(b, 16, p * p)             # (b, 16 patches, 49 pixels)

    def forward(self, x):
        z = self.proj(self.to_patches(x)) + self.pos  # (b, 16, 64)
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


def get_models():
    return {"MLP": MLP(), "CNN": CNN(), "Attention": PatchAttention()}
