import torch
import torch.nn as nn
from torchinfo import summary


class MLP(nn.Module):
    def __init__(
        self, in_steps=12, out_steps=12, input_dim=1, output_dim=1, hidden_dim=256
    ):
        super().__init__()

        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_steps * input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_steps * output_dim),
        )

    def forward(self, x):
        # x: (B, T, N, C)
        batch_size = x.shape[0]
        num_nodes = x.shape[2]

        x = x.transpose(1, 2).view(
            batch_size, num_nodes, self.in_steps * self.input_dim
        )  # (B, N, T*C)

        out = self.mlp(x)  # (B, N, out_steps*output_dim)
        out = out.view(
            batch_size, num_nodes, self.out_steps, self.output_dim
        ).transpose(1, 2)

        return out

if __name__ == "__main__":
    model = MLP()
    summary(model, [64, 12, 207, 1])
    