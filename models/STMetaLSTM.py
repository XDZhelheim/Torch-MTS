import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary


class STMetaLSTMCell(nn.Module):
    def __init__(
        self, input_dim, lstm_hidden_dim, st_embedding_dim, learner_hidden_dim, z_dim
    ):
        super().__init__()

        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.st_embedding_dim = st_embedding_dim
        self.learner_hidden_dim = learner_hidden_dim
        self.z_dim = z_dim

        self.learner_wx = nn.Sequential(
            nn.Linear(self.st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 4 * input_dim * lstm_hidden_dim),
        )

        self.learner_wh = nn.Sequential(
            nn.Linear(self.st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 4 * lstm_hidden_dim * lstm_hidden_dim),
        )

        self.learner_b = nn.Sequential(
            nn.Linear(self.st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 4 * lstm_hidden_dim),
        )

    def forward(self, xt, meta_input, h, c):
        # xt           (B, N, input_dim)
        # meta_input   (B, N, st_embedding_dim+z_dim)
        # h, c         (B, N, lstm_hidden_dim)
        batch_size = xt.shape[0]
        num_nodes = xt.shape[1]

        Wx = self.learner_wx(meta_input).view(
            batch_size, num_nodes, self.input_dim, 4 * self.lstm_hidden_dim
        )
        Wh = self.learner_wh(meta_input).view(
            batch_size, num_nodes, self.lstm_hidden_dim, 4 * self.lstm_hidden_dim
        )
        b = self.learner_b(meta_input).view(
            batch_size, num_nodes, 4 * self.lstm_hidden_dim
        )

        combined_gates = (
            torch.einsum("bnc,bncd->bnd", xt, Wx)
            + torch.einsum("bnh,bnhd->bnd", h, Wh)
            + b
        )  # (B, N, 4*lstm_hidden_dim)

        g, i, f, o = torch.split(combined_gates, self.lstm_hidden_dim, dim=-1)
        g = torch.tanh(g)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)

        c = g * i + c * f
        h = torch.tanh(c) * o

        return h, c


class STMetaLSTM(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_emb_file,
        device=None,
        in_steps=12,
        out_steps=12,
        input_dim=1,
        output_dim=1,
        lstm_hidden_dim=64,
        tod_embedding_dim=24,
        dow_embedding_dim=7,
        node_embedding_dim=64,
        learner_hidden_dim=64,
        z_dim=0,
        num_layers=1,
        seq2seq=False,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
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

        self.encoder = nn.ModuleList(
            [
                STMetaLSTMCell(
                    input_dim,
                    lstm_hidden_dim,
                    self.st_embedding_dim,
                    learner_hidden_dim,
                    z_dim,
                )
            ]
        )
        for _ in range(num_layers - 1):
            self.encoder.append(
                STMetaLSTMCell(
                    lstm_hidden_dim,
                    lstm_hidden_dim,
                    self.st_embedding_dim,
                    learner_hidden_dim,
                    z_dim,
                )
            )

        self.decoder = nn.Linear(lstm_hidden_dim, out_steps * output_dim)

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
            mu = self.mu_estimator(x.squeeze(dim=-1).transpose(1, 2))  # (B, N, z_dim)

            logvar = self.logvar_estimator(
                x.squeeze(dim=-1).transpose(1, 2)
            )  # (B, N, z_dim)

            z_data = self.reparameterize(mu, logvar)  # temporal z (B, N, z_dim)

            z_data = z_data + self.reparameterize(
                self.mu, self.logvar
            )  # temporal z + spatial z

            meta_input = torch.concat(
                [meta_input, z_data], dim=-1
            )  # (B, N, st_emb_dim+z_dim)

        lstm_input = x  # (B, T_in, N, 1)
        h_each_layer, c_each_layer = [], []  # last step's h, c of each layer
        for lstm_cell in self.encoder:
            # temp_weight = next(lstm_cell.parameters()).data
            # # use new_zeros to generate zero tensor, keeping the same device and requires_grad option
            # h = temp_weight.new_zeros(batch_size, self.num_nodes, self.lstm_hidden_dim)
            # c = temp_weight.new_zeros(batch_size, self.num_nodes, self.lstm_hidden_dim)
            h = torch.zeros(
                batch_size, self.num_nodes, self.lstm_hidden_dim, device=x.device
            )
            c = torch.zeros(
                batch_size, self.num_nodes, self.lstm_hidden_dim, device=x.device
            )

            h_each_step = []  # every step's h
            for t in range(self.in_steps):
                h, c = lstm_cell(lstm_input[:, t, ...], meta_input, h, c)
                h_each_step.append(h)  # T_in*(B, N, lstm_hidden_dim)
            lstm_input = torch.stack(
                h_each_step, dim=1
            )  # (B, T_in, N, lstm_hidden_dim) input for next layer

            h_each_layer.append(h)  # num_layers*(B, N, lstm_hidden_dim)
            c_each_layer.append(c)

        out = h_each_layer[-1]  # (B, N, lstm_hidden_dim) last layer last step's h

        out = self.decoder(out).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )  # (B, N, T_out, output_dim=1)

        return out.transpose(1, 2)  # (B, N, T_out, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class STMetaLSTM2(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_emb_file,
        device=None,
        in_steps=12,
        out_steps=12,
        input_dim=1,
        output_dim=1,
        lstm_hidden_dim=64,
        tod_embedding_dim=24,
        dow_embedding_dim=7,
        node_embedding_dim=64,
        learner_hidden_dim=64,
        z_dim=0,
        num_layers=1,
        seq2seq=False,
    ):
        super(STMetaLSTM, self).__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
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

        self.learner_wx = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.st_embedding_dim + z_dim, learner_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(learner_hidden_dim, input_dim * lstm_hidden_dim),
                )
                for _ in range(4)
            ]
        )

        self.learner_wh = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.st_embedding_dim + z_dim, learner_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(learner_hidden_dim, lstm_hidden_dim * lstm_hidden_dim),
                )
                for _ in range(4)
            ]
        )

        self.learner_b = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.st_embedding_dim + z_dim, learner_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(learner_hidden_dim, lstm_hidden_dim),
                )
                for _ in range(4)
            ]
        )

        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(lstm_hidden_dim // 2, out_steps * output_dim),
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.shape[0]

        tod = x[..., 1]  # (B, T_in, N)
        dow = x[..., 2]  # (B, T_in, N)
        x = x[..., :1].transpose(1, 2)  # (B, N, T_in, 1)

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
            mu = self.mu_estimator(x.squeeze(dim=-1))  # (B, N, z_dim)

            logvar = self.logvar_estimator(x.squeeze(dim=-1))  # (B, N, z_dim)

            z_data = self.reparameterize(mu, logvar)  # temporal z (B, N, z_dim)

            z_data = z_data + self.reparameterize(
                self.mu, self.logvar
            )  # temporal z + spatial z

            meta_input = torch.concat(
                [meta_input, z_data], dim=-1
            )  # (B, N, st_emb_dim+z_dim)

        # input and 3 gates
        Wgx = self.learner_wx[0](meta_input).view(
            batch_size, self.num_nodes, self.input_dim, self.lstm_hidden_dim
        )
        Wgh = self.learner_wh[0](meta_input).view(
            batch_size, self.num_nodes, self.lstm_hidden_dim, self.lstm_hidden_dim
        )
        bg = self.learner_b[0](meta_input).view(
            batch_size, self.num_nodes, 1, self.lstm_hidden_dim
        )

        Wix = self.learner_wx[1](meta_input).view(
            batch_size, self.num_nodes, self.input_dim, self.lstm_hidden_dim
        )
        Wih = self.learner_wh[1](meta_input).view(
            batch_size, self.num_nodes, self.lstm_hidden_dim, self.lstm_hidden_dim
        )
        bi = self.learner_b[1](meta_input).view(
            batch_size, self.num_nodes, 1, self.lstm_hidden_dim
        )

        Wfx = self.learner_wx[2](meta_input).view(
            batch_size, self.num_nodes, self.input_dim, self.lstm_hidden_dim
        )
        Wfh = self.learner_wh[2](meta_input).view(
            batch_size, self.num_nodes, self.lstm_hidden_dim, self.lstm_hidden_dim
        )
        bf = self.learner_b[2](meta_input).view(
            batch_size, self.num_nodes, 1, self.lstm_hidden_dim
        )

        Wox = self.learner_wx[3](meta_input).view(
            batch_size, self.num_nodes, self.input_dim, self.lstm_hidden_dim
        )
        Woh = self.learner_wh[3](meta_input).view(
            batch_size, self.num_nodes, self.lstm_hidden_dim, self.lstm_hidden_dim
        )
        bo = self.learner_b[3](meta_input).view(
            batch_size, self.num_nodes, 1, self.lstm_hidden_dim
        )

        for timestep in range(self.in_steps):
            xt = x[:, :, timestep, :].unsqueeze(2)  # (bs, num_nodes, 1, input_dim)

            if timestep == 0:
                g = torch.tanh(
                    xt @ Wgx + bg
                )  # (bs, num_nodes, 1, input_dim) * (bs, num_nodes, input_dim, hidden_dim) = (bs, num_nodes, 1, hidden_dim)
                i = torch.sigmoid(xt @ Wix + bi)
                f = torch.sigmoid(xt @ Wfx + bf)
                o = torch.sigmoid(xt @ Wox + bo)

                c = g * i
                h = torch.tanh(c) * o
            else:
                g = torch.tanh(xt @ Wgx + h @ Wgh + bg)

                i = torch.sigmoid(xt @ Wix + h @ Wih + bi)
                f = torch.sigmoid(xt @ Wfx + h @ Wfh + bf)
                o = torch.sigmoid(xt @ Wox + h @ Woh + bo)

                c = g * i + c * f
                h = torch.tanh(c) * o

        out = h.squeeze(dim=2)  # (bs, num_nodes, hidden_dim)

        out = self.fc(out).view(
            batch_size, self.num_nodes, self.out_steps, self.input_dim
        )  # (batch_size, num_nodes, out_steps, input_dim=1)

        return out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, 1)


if __name__ == "__main__":
    model = STMetaLSTM(
        207, "../data/METRLA/spatial_embeddings.npz", torch.device("cpu"), z_dim=32
    )
    summary(model, [64, 12, 207, 3], device="cpu")
