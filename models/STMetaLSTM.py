import torch
import torch.nn as nn
from torchinfo import summary


class STMetaLSTM(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps,
        out_steps,
        lstm_input_dim,
        lstm_hidden_dim,
        st_embedding_dim,
        learner_hidden_dim,
        z_dim=0,
        towards=False,
    ):
        super(STMetaLSTM, self).__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.st_embedding_dim = st_embedding_dim
        self.learner_hidden_dim = learner_hidden_dim
        self.z_dim = z_dim
        self.towards = towards

        self.learner_wx = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(learner_hidden_dim, lstm_input_dim * lstm_hidden_dim),
                )
                for _ in range(4)
            ]
        )

        self.learner_wh = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(learner_hidden_dim, lstm_hidden_dim * lstm_hidden_dim),
                )
                for _ in range(4)
            ]
        )

        self.learner_b = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
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
            nn.Linear(lstm_hidden_dim // 2, out_steps * lstm_input_dim),
        )

        if self.towards:
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
        """
        x: (batch_size, in_steps, num_nodes, input_dim+s_meta_input_dim+t_meta_input_dim=1+64+32)
        meta_input: (batch_size, in_steps, num_nodes, s_meta_input_dim+t_meta_input_dim=64+32)
        """

        meta_input = x[..., 1:]  # (batch_size, in_steps, num_nodes, 96)
        x = x[..., :1].transpose(1, 2)  # (batch_size, num_nodes, in_steps, 1)

        meta_input = meta_input.mean(dim=1)  # (batch_size, num_nodes, 96)

        batch_size = x.shape[0]
        in_steps = x.shape[2]

        if self.towards:
            mu = self.mu_estimator(x.squeeze())  # (batch_size, num_nodes, z_dim=32)

            logvar = self.logvar_estimator(
                x.squeeze()
            )  # (batch_size, num_nodes, z_dim)

            z_data = self.reparameterize(mu, logvar)  # (batch_size, num_nodes, z_dim)

            z_data = z_data + self.reparameterize(self.mu, self.logvar)

            meta_input = torch.concat(
                (meta_input, z_data), dim=2
            )  # (batch_size, num_nodes, 128)

        # input and 3 gates
        Wgx = self.learner_wx[0](meta_input).view(
            batch_size, self.num_nodes, self.lstm_input_dim, self.lstm_hidden_dim
        )
        Wgh = self.learner_wh[0](meta_input).view(
            batch_size, self.num_nodes, self.lstm_hidden_dim, self.lstm_hidden_dim
        )
        bg = self.learner_b[0](meta_input).view(
            batch_size, self.num_nodes, 1, self.lstm_hidden_dim
        )

        Wix = self.learner_wx[1](meta_input).view(
            batch_size, self.num_nodes, self.lstm_input_dim, self.lstm_hidden_dim
        )
        Wih = self.learner_wh[1](meta_input).view(
            batch_size, self.num_nodes, self.lstm_hidden_dim, self.lstm_hidden_dim
        )
        bi = self.learner_b[1](meta_input).view(
            batch_size, self.num_nodes, 1, self.lstm_hidden_dim
        )

        Wfx = self.learner_wx[2](meta_input).view(
            batch_size, self.num_nodes, self.lstm_input_dim, self.lstm_hidden_dim
        )
        Wfh = self.learner_wh[2](meta_input).view(
            batch_size, self.num_nodes, self.lstm_hidden_dim, self.lstm_hidden_dim
        )
        bf = self.learner_b[2](meta_input).view(
            batch_size, self.num_nodes, 1, self.lstm_hidden_dim
        )

        Wox = self.learner_wx[3](meta_input).view(
            batch_size, self.num_nodes, self.lstm_input_dim, self.lstm_hidden_dim
        )
        Woh = self.learner_wh[3](meta_input).view(
            batch_size, self.num_nodes, self.lstm_hidden_dim, self.lstm_hidden_dim
        )
        bo = self.learner_b[3](meta_input).view(
            batch_size, self.num_nodes, 1, self.lstm_hidden_dim
        )

        for timestep in range(in_steps):
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
            batch_size, self.num_nodes, self.out_steps, self.lstm_input_dim
        )  # (batch_size, num_nodes, out_steps, lstm_input_dim=1)

        return out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, 1)


if __name__ == "__main__":
    model = STMetaLSTM(207, 12, 12, 1, 64, 96, 64, 32, True)
    summary(model, [8, 12, 207, 97])
