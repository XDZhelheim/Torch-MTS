import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, num_nodes, in_steps, out_steps, lstm_input_dim, lstm_hidden_dim):
        super(LSTMPredictor, self).__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.use_all_h = False

        bi = True
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            num_layers=2,
            bidirectional=bi,
            dropout=0,
        )

        self.fc_input_dim = lstm_hidden_dim
        if bi:
            self.fc_input_dim *= 2
        if self.use_all_h:
            self.fc_input_dim *= in_steps
        self.fc = nn.Linear(self.fc_input_dim, out_steps * lstm_input_dim)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, lstm_input_dim=1)
        batch_size = x.shape[0]
        x = x.view(batch_size * self.num_nodes, self.in_steps, self.lstm_input_dim)

        out, _ = self.lstm(x)
        if self.use_all_h:
            out = out.contiguous().view(
                batch_size * self.num_nodes, self.fc_input_dim
            )  # (batch_size * num_nodes, in_steps * hidden_dim) use all step's output
        else:
            out = out[
                :, -1, :
            ]  # (batch_size * num_nodes, hidden_dim) use last step's output

        out = self.fc(out).view(
            batch_size, self.out_steps, self.num_nodes, self.lstm_input_dim
        )

        return out
    