import torch.nn as nn
import torch
import numpy as np
from torchinfo import summary


class STMetaAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=4, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.out_proj = nn.Linear(model_dim, model_dim)

    def set_weights(self, W, b):
        self.W = W
        self.b = b

        self.WQ, self.WK, self.WV = torch.split(W, self.model_dim, dim=-1)
        self.bQ, self.bK, self.bV = torch.split(b, self.model_dim, dim=-1)

    def forward(self, query, key, value, spatial=False):
        # Q             (B, N, T, model_dim)
        # K, V          (B, N, T, model_dim)
        # WQ, WK, WV    (B, N, model_dim, model_dim)
        # bQ, bK, bV    (B, N, 1, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = query @ self.WQ + self.bQ
        key = key @ self.WK + self.bK
        value = value @ self.WV + self.bV

        if spatial:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        # Qhead, Khead, Vhead (num_heads * B, N, T, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)  # (num_heads * B, N, head_dim, T)

        attn_score = (query @ key) / self.head_dim ** 0.5  # (num_heads * B, N, T, T)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * B, N, T, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (B, N, T, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        if spatial:
            out = out.transpose(1, 2)

        return out


class STMetaSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        st_embedding_dim=95,
        learner_hidden_dim=128,
        z_dim=32,
        feed_forward_dim=128,
        num_heads=4,
        dropout=0.1,
        mask=False,
    ):
        super().__init__()

        self.model_dim = model_dim

        self.attn = STMetaAttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.learner_w = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 3 * model_dim * model_dim),
        )

        self.learner_b = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 3 * model_dim),
        )

    def forward(self, x, meta_input, spatial=False):
        # x            (B, T_in, N, model_dim)
        # meta_input   (B, N, st_embedding_dim+z_dim)
        batch_size = x.shape[0]
        num_nodes = x.shape[2]

        W = self.learner_w(meta_input).view(
            batch_size, num_nodes, self.model_dim, 3 * self.model_dim
        )
        b = self.learner_b(meta_input).view(
            batch_size, num_nodes, 1, 3 * self.model_dim
        )

        self.attn.set_weights(W, b)

        x = x.transpose(1, 2)
        residual = x
        out = self.attn(x, x, x, spatial)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        return out.transpose(1, 2)


class STMetaAttention(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_emb_file,
        device=None,
        in_steps=12,
        out_steps=12,
        input_dim=1,
        output_dim=1,
        model_dim=32,
        tod_embedding_dim=24,
        dow_embedding_dim=7,
        node_embedding_dim=64,
        learner_hidden_dim=128,
        z_dim=32,
        feed_forward_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        with_spatial=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dim = model_dim
        self.st_embedding_dim = (
            tod_embedding_dim + dow_embedding_dim + node_embedding_dim
        )
        self.learner_hidden_dim = learner_hidden_dim
        self.z_dim = z_dim
        self.feed_forward_dim = feed_forward_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.with_spatial = with_spatial

        self.node_embedding = torch.FloatTensor(np.load(node_emb_file)["data"]).to(
            device
        )

        self.tod_onehots = torch.eye(24, device=device)
        self.dow_onehots = torch.eye(7, device=device)

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.temporal_proj = nn.Linear(in_steps, out_steps)
        self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                STMetaSelfAttentionLayer(
                    self.model_dim,
                    self.st_embedding_dim,
                    learner_hidden_dim,
                    z_dim,
                    feed_forward_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        if with_spatial:
            self.attn_layers_s = nn.ModuleList(
                [
                    STMetaSelfAttentionLayer(
                        self.model_dim,
                        self.st_embedding_dim,
                        learner_hidden_dim,
                        z_dim,
                        feed_forward_dim,
                        num_heads,
                        dropout,
                    )
                    for _ in range(num_layers)
                ]
            )

        if self.z_dim > 0:
            self.mu = nn.Parameter(torch.randn(num_nodes, z_dim), requires_grad=True)
            self.logvar = nn.Parameter(
                torch.randn(num_nodes, z_dim), requires_grad=True
            )

            self.mu_estimator = nn.Sequential(
                nn.Linear(in_steps, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, z_dim),
            )

            self.logvar_estimator = nn.Sequential(
                nn.Linear(in_steps, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, z_dim),
            )

    def forward(self, x):
        """
        x: (B, T_in, N, input_dim+tod+dow=3)
        
        BTN3
        x: BTN1
        tod: BTN -> use last step (ref. STID) -> BN
        dow: BTN -> use last step (ref. STID) -> BN
        
        tod -> one-hot -> BN24
        dow -> one-hot -> BN7
        
        spatial: N64 -> broadcast -> BN64
        """
        batch_size = x.shape[0]

        tod = x[..., 1]  # (B, T_in, N)
        dow = x[..., 2]  # (B, T_in, N)
        x = x[..., :1]  # (B, T_in, N, 1)
        x_original = x

        x = self.input_proj(x)

        # use the last time step to represent the temporal location of the time seires
        tod_embedding = self.tod_onehots[(tod[:, -1, :] * 24).long()]  # (B, N, 24)
        dow_embedding = self.dow_onehots[dow[:, -1, :].long()]  # (B, N, 7)
        node_embedding = self.node_embedding.expand(
            batch_size, *self.node_embedding.shape
        )  # (B, N, node_emb_dim)

        meta_input = torch.concat(
            [node_embedding, tod_embedding, dow_embedding], dim=-1
        )  # (B, N, st_emb_dim)

        if self.z_dim > 0:
            z_input = x_original.squeeze(dim=-1).transpose(1, 2)

            mu = self.mu_estimator(z_input)  # (B, N, z_dim)
            logvar = self.logvar_estimator(z_input)  # (B, N, z_dim)

            z_data = self.reparameterize(mu, logvar)  # temporal z (B, N, z_dim)
            z_data = z_data + self.reparameterize(
                self.mu, self.logvar
            )  # temporal z + spatial z

            meta_input = torch.concat(
                [meta_input, z_data], dim=-1
            )  # (B, N, st_emb_dim+z_dim)

        for attn in self.attn_layers_t:
            x = attn(x, meta_input)
        if self.with_spatial:
            for attn in self.attn_layers_s:
                x = attn(x, meta_input, spatial=True)
        # (B, T_in, N, model_dim)

        out = x.transpose(1, 3)  # (B, model_dim, N, T_in)
        out = self.temporal_proj(out)  # (B, model_dim, N, T_out)
        out = self.output_proj(out.transpose(1, 3))  # (B, T_out, N, output_dim)

        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


if __name__ == "__main__":
    model = STMetaAttention(
        207,
        "../../data/METRLA/spatial_embeddings.npz",
        torch.device("cpu"),
        learner_hidden_dim=64,
        feed_forward_dim=64,
        model_dim=32,
        z_dim=32,
        num_layers=1,
        with_spatial=True,
    )
    summary(model, [64, 12, 207, 3], device="cpu")

