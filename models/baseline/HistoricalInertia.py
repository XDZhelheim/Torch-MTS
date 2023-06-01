import torch
import torch.nn as nn
from torchinfo import summary


class HistoricalInertia(nn.Module):
    def __init__(self, in_steps=12, out_steps=12):
        super().__init__()

        assert in_steps >= out_steps

        self.in_steps = in_steps
        self.out_steps = out_steps

        self.placeholder = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, T, N, C)

        self.placeholder.data[0] = 0 # kill backprop

        return x[:, -self.out_steps :, :, :] + self.placeholder


if __name__ == "__main__":
    model = HistoricalInertia()
    summary(model, [64, 12, 207, 1])
