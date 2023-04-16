import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary


class STMetaGRUCell(nn.Module):
    def __init__(self, gru_hidden_dim=64):
        super().__init__()

        self.gru_hidden_dim = gru_hidden_dim

    def set_weights(self, Wx, Wh, b):
        self.Wx = Wx
        self.Wh = Wh
        self.b = b

        self.Wrx, self.Wzx, self.Wnx = torch.split(Wx, self.gru_hidden_dim, dim=-1)
        self.Wrh, self.Wzh, self.Wnh = torch.split(Wh, self.gru_hidden_dim, dim=-1)
        self.br, self.bz, self.bn = torch.split(b, self.gru_hidden_dim, dim=-1)

    def forward(self, xt, h):
        # xt             (B, N, input_dim)
        # h              (B, N, gru_hidden_dim)
        # Wrx Wzx Wnx    (B, N, input_dim, gru_hidden_dim)
        # Wrh Wzh Wnh    (B, N, gru_hidden_dim, gru_hidden_dim)
        # br bz bn       (B, N, gru_hidden_dim)

        xt = xt[:, :, None, :]  # (B, N, 1, input_dim)
        r = torch.sigmoid(xt @ self.Wrx + h @ self.Wrh + self.br)
        z = torch.sigmoid(xt @ self.Wzx + h @ self.Wzh + self.bz)
        n = torch.tanh(xt @ self.Wnx + (r * h) @ self.Wnh + self.bn)

        h = z * n + (1 - z) * h  # (B, N, 1, gru_hidden_dim)
        # 这里不能是 (1-z)n+zh, 否则z的导数是相反数, 越训loss越高
        # https://stats.stackexchange.com/questions/511642/gru-hidden-state-output-formula-difference

        return h


class STMetaGRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim=1,
        gru_hidden_dim=64,
        st_embedding_dim=95,
        learner_hidden_dim=128,
        z_dim=32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.st_embedding_dim = st_embedding_dim
        self.learner_hidden_dim = learner_hidden_dim
        self.z_dim = z_dim

        self.cell = STMetaGRUCell(gru_hidden_dim=gru_hidden_dim)

        self.learner_wx = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 3 * input_dim * gru_hidden_dim),
        )

        self.learner_wh = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 3 * gru_hidden_dim * gru_hidden_dim),
        )

        self.learner_b = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 3 * gru_hidden_dim),
        )

    def forward(self, x, meta_input):
        # x            (B, T_in, N, input_dim)
        # meta_input   (B, N, st_embedding_dim+z_dim)

        batch_size = x.shape[0]
        in_steps = x.shape[1]
        num_nodes = x.shape[2]

        Wx = self.learner_wx(meta_input).view(
            batch_size, num_nodes, self.input_dim, 3 * self.gru_hidden_dim
        )
        Wh = self.learner_wh(meta_input).view(
            batch_size, num_nodes, self.gru_hidden_dim, 3 * self.gru_hidden_dim
        )
        b = self.learner_b(meta_input).view(
            batch_size, num_nodes, 1, 3 * self.gru_hidden_dim
        )

        self.cell.set_weights(Wx, Wh, b)

        h = torch.zeros(batch_size, num_nodes, 1, self.gru_hidden_dim, device=x.device)

        h_each_step = []
        for t in range(in_steps):
            h = self.cell(x[:, t, ...], h)  # (B, N, 1, lstm_hidden_dim)
            h_each_step.append(h.squeeze(dim=2))  # T_in*(B, N, lstm_hidden_dim)

        h_each_step = torch.stack(
            h_each_step, dim=1
        )  # (B, T_in, N, lstm_hidden_dim) input for next layer

        return h_each_step, h.squeeze(dim=2)  # last step's h


class STMetaGRU(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_emb_file,
        device=None,
        in_steps=12,
        out_steps=12,
        input_dim=1,
        output_dim=1,
        gru_hidden_dim=64,
        tod_embedding_dim=24,
        dow_embedding_dim=7,
        node_embedding_dim=64,
        learner_hidden_dim=128,
        z_dim=32,
        num_layers=1,
        seq2seq=False,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.st_embedding_dim = (
            tod_embedding_dim + dow_embedding_dim + node_embedding_dim
        )
        self.learner_hidden_dim = learner_hidden_dim
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.seq2seq = seq2seq

        self.node_embedding = torch.FloatTensor(np.load(node_emb_file)["data"]).to(
            device
        )

        self.tod_onehots = torch.eye(24, device=device)
        self.dow_onehots = torch.eye(7, device=device)

        self.encoders = nn.ModuleList(
            [
                STMetaGRUEncoder(
                    input_dim,
                    gru_hidden_dim,
                    self.st_embedding_dim,
                    learner_hidden_dim,
                    z_dim,
                )
            ]
        )
        for _ in range(num_layers - 1):
            self.encoders.append(
                STMetaGRUEncoder(
                    gru_hidden_dim,
                    gru_hidden_dim,
                    self.st_embedding_dim,
                    learner_hidden_dim,
                    z_dim,
                )
            )

        self.decoder = nn.Linear(gru_hidden_dim, out_steps * output_dim)

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
            z_input = x.squeeze(dim=-1).transpose(1, 2)

            mu = self.mu_estimator(z_input)  # (B, N, z_dim)
            logvar = self.logvar_estimator(z_input)  # (B, N, z_dim)

            z_data = self.reparameterize(mu, logvar)  # temporal z (B, N, z_dim)
            z_data = z_data + self.reparameterize(
                self.mu, self.logvar
            )  # temporal z + spatial z

            meta_input = torch.concat(
                [meta_input, z_data], dim=-1
            )  # (B, N, st_emb_dim+z_dim)

        lstm_input = x  # (B, T_in, N, 1)
        h_each_layer = []  # last step's h of each layer
        for encoder in self.encoders:
            lstm_input, last_h = encoder(lstm_input, meta_input)

            h_each_layer.append(last_h)  # num_layers*(B, N, lstm_hidden_dim)

        # TODO seq2seq

        out = h_each_layer[-1]  # (B, N, lstm_hidden_dim) last layer last step's h

        out = self.decoder(out).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )  # (B, N, T_out, output_dim=1)

        return out.transpose(1, 2)  # (B, N, T_out, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


if __name__ == "__main__":
    model = STMetaGRU(
        207,
        "../data/METRLA/spatial_embeddings.npz",
        torch.device("cpu"),
        learner_hidden_dim=128,
        gru_hidden_dim=32,
        z_dim=32,
        num_layers=1,
    )
    summary(model, [64, 12, 207, 3], device="cpu")

