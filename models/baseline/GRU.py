import torch.nn as nn
import torch
from torchinfo import summary


class GRU(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps,
        out_steps,
        input_dim=1,
        output_dim=1,
        gru_hidden_dim=64,
        num_layers=3,
        seq2seq=True,
    ):
        super(GRU, self).__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_layers = num_layers
        self.seq2seq = seq2seq

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=not seq2seq,  # when not using seq2seq, use bidirectional LSTM
            dropout=0,
        )

        if self.seq2seq:
            self.decoder = nn.GRU(
                input_size=output_dim,
                hidden_size=gru_hidden_dim,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=False,
                dropout=0,
            )
            self.proj = nn.Linear(gru_hidden_dim, output_dim)
        else:
            self.decoder = nn.Linear(gru_hidden_dim * 2, out_steps * output_dim)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim=1)
        x = x.transpose(1, 2).contiguous()  # (batch_size, num_nodes, in_steps, 1)
        batch_size = x.shape[0]
        x = x.view(batch_size * self.num_nodes, self.in_steps, self.input_dim)

        out, h = self.encoder(x)

        if self.seq2seq:
            decoder_input = torch.zeros(
                (batch_size * self.num_nodes, 1, self.output_dim), device=x.device
            )  # (batch_size * num_nodes, 1, output_dim)
            out = []

            for _ in range(self.out_steps):
                decoder_input, h = self.decoder(
                    decoder_input, h
                )  # (batch_size * num_nodes, 1, hidden_dim)
                decoder_input = self.proj(
                    decoder_input
                )  # (batch_size * num_nodes, 1, output_dim)
                out.append(decoder_input)

            out = torch.cat(
                out, dim=1
            )  # (batch_size * num_nodes, out_steps, output_dim)
        else:
            out = out[
                :, -1, :
            ]  # (batch_size * num_nodes, hidden_dim) use last step's output
            out = self.decoder(out)  # (batch_size * num_nodes, out_steps * output_dim)

        out = out.view(batch_size, self.num_nodes, self.out_steps, self.output_dim)

        return out.transpose(1, 2)


if __name__ == "__main__":
    model = GRU(207, 12, 12, 1, 1, 64, seq2seq=True)
    summary(model, [128, 207, 12, 1])
