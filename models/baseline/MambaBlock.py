"""
My implementation of Mamba block,
a sequential version
"""

import torch
import torch.nn as nn
import math


# https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )

        return output


class MambaBlock(nn.Module):
    def __init__(
        self,
        model_dim,
        state_dim=16,
        expand=2,
        conv_kernel_size=4,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        rms_norm_eps=1e-5,
    ):
        super().__init__()

        self.model_dim = model_dim  # D in paper (model_dim=input_dim=output_dim)
        self.state_dim = state_dim  # N in paper (hidden state dim)
        self.conv_kernel_size = conv_kernel_size  # conv1d kernel size
        self.expand = expand  # E=2
        self.inner_dim = self.expand * self.model_dim  # E*D = ED in paper
        self.dt_rank = math.ceil(self.model_dim / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.rms_norm_eps = rms_norm_eps

        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()
        self.norm = RMSNorm(self.model_dim, self.rms_norm_eps)

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(self.model_dim, 2 * self.inner_dim, bias=False)

        self.conv = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.inner_dim,
            padding=self.conv_kernel_size - 1,
        )

        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(
            self.inner_dim, self.dt_rank + 2 * self.state_dim, bias=False
        )

        # projects delta from dt_rank to ED
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.inner_dim) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        self.A = nn.Parameter(
            -torch.arange(1, self.state_dim + 1, dtype=torch.float32).repeat(
                self.inner_dim, 1
            )
        )
        self.D = nn.Parameter(torch.ones(self.inner_dim, dtype=torch.float32))
        self.A._no_weight_decay = True
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(self.inner_dim, self.model_dim, bias=False)

    def forward(self, x):
        # x: (B, L, D)
        L = x.shape[1]
        residual = x

        x = self.norm(x)

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = torch.split(xz, self.inner_dim, dim=-1)  # both (B, L, ED)

        # x branch
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv(x)[:, :, :L]
        x = x.transpose(1, 2)  # (B, L, ED)

        x = self.silu(x)
        y = self.ssm(x)

        # z branch
        z = self.silu(z)

        out = y * z
        out = self.out_proj(out)  # (B, L, D)

        return out + residual

    def ssm(self, x):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        batch_size, L, _ = x.shape

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(
            deltaBC, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )  # (B, L, dt_rank), (B, L, N), (B, L, N)

        delta = self.softplus(self.dt_proj(delta))  # (B, L, ED)

        dA = torch.exp(torch.einsum("ble,en->blen", delta, self.A))  # (B, L, ED, N)
        dB = torch.einsum("ble,bln->blen", delta, B)  # (B, L, ED, N)

        y_list = []
        h = torch.zeros(
            size=(batch_size, self.inner_dim, self.state_dim), device=x.device
        )  # (B, ED, N)

        for t in range(L):
            # h_t = dA * h_t-1 + dB * x_t
            # y_t = C * h_t + D * x_t

            h = (
                dA[:, t, :, :] * h + dB[:, t, :, :] * x[:, t, :, None]
            )  # (B, ED, N)*(B, ED, N) + (B, ED, N)*(B, ED, 1)

            y = (
                torch.einsum("bn,ben->be", C[:, t, :], h) + self.D * x[:, t, :]
            )  # (B, ED)

            y_list.append(y)

        ys = torch.stack(y_list, dim=1)  # (B, L, ED)

        return ys


class MambaSeq(nn.Module):
    def __init__(
        self,
        num_nodes=207,
        seq_len=12,
        pred_len=12,
        input_dim=1,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(num_nodes * input_dim, hidden_dim)
        self.mamba_layer = nn.Sequential(
            *[MambaBlock(model_dim=hidden_dim) for _ in range(num_layers)]
        )
        self.time_proj = nn.Conv1d(
            in_channels=seq_len, out_channels=pred_len, kernel_size=(1,)
        )
        self.output_proj = nn.Linear(hidden_dim, num_nodes * output_dim)

    def forward(self, x):
        # x: (B, T, N, C)
        batch_size = x.shape[0]

        x = self.input_proj(
            x.view(batch_size, self.seq_len, self.num_nodes * self.input_dim)
        )  # (B, T, H)
        out = self.mamba_layer(x)  # (B, T, H)
        out = self.output_proj(out)  # (B, T, N*1)
        out = self.time_proj(out)  # (B, Tout, N*1)

        return out.unsqueeze(-1)


def print_model_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("%-40s\t%-30s\t%-30s" % (name, list(param.shape), param.numel()))
            param_count += param.numel()
    print("%-40s\t%-30s" % ("Total trainable params", param_count))


if __name__ == "__main__":
    from torchinfo import summary

    model = MambaSeq()
    summary(model, [64, 12, 207, 1], device="cpu")
    print_model_params(model)
