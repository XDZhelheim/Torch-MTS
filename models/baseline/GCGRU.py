import torch
import torch.nn as nn
from torchinfo import summary
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import pickle
import pandas as pd
import numpy as np


def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (
        adj.dot(d_mat_inv_sqrt)
        .transpose()
        .dot(d_mat_inv_sqrt)
        .astype(np.float32)
        .todense()
    )


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = (
        sp.eye(adj.shape[0])
        - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format="csr", dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_adj(filepath, filetype, adjtype):
    print(filepath, filetype, adjtype)
    if filetype == "pkl":
        try:
            # METRLA and PEMSBAY
            _, _, adj_mx = load_pickle(filepath)
        except ValueError:
            # PEMS3478
            adj_mx = load_pickle(filepath)
    elif filetype == "csv":
        adj_mx = pd.read_csv(filepath).values
    else:
        error = 0
        assert error, "adj file type not defined"

    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adjtype == None:
        adj = adj_mx
    else:
        error = 0
        assert error, "adj type not defined"
    return adj


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.dim_in = dim_in
        self.W = nn.Parameter(torch.empty(cheb_k * dim_in, dim_out), requires_grad=True)
        self.b = nn.Parameter(torch.empty(dim_out), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        nn.init.constant_(self.b, val=0)

    def forward(self, G, x):
        """
        :param x: graph feature/signal          -   [B, N, C + H_in] input concat last step's hidden
        :param G: support adj matrices          -   [K, N, N]
        :return output: hidden representation   -   [B, N, H_out]
        """
        support_list = []
        for k in range(self.cheb_k):
            support = torch.einsum(
                "ij,bjp->bip", G[k, :, :], x
            )  # [B, N, C + H_in] perform GCN
            support_list.append(support)  # k * [B, N, C + H_in]
        support_cat = torch.cat(support_list, dim=-1)  # [B, N, k * (C + H_in)]
        output = (
            torch.einsum("bip,pq->biq", support_cat, self.W) + self.b
        )  # [B, N, H_out]
        return output


class GRUCell(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_hidden, cheb_k):
        super(GRUCell, self).__init__()
        self.num_nodes = num_nodes
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden

        self.gate = GCN(
            cheb_k=cheb_k, dim_in=dim_in + dim_hidden, dim_out=2 * dim_hidden
        )
        self.update = GCN(cheb_k=cheb_k, dim_in=dim_in + dim_hidden, dim_out=dim_hidden)

    def forward(self, G, x_t, h_pre):
        """
        :param G: support adj matrices      -   [K, N, N]
        :param x_t: graph feature/signal    -   [B, N, C]
        :param h_pre: previous hidden state -   [B, N, H]
        :return h_t: current hidden state   -   [B, N, H]
        """
        combined = torch.cat([x_t, h_pre], dim=-1)  # concat input and last hidden
        z_r = torch.sigmoid(self.gate(G, combined))  # [B, N, 2 * H]
        z, r = torch.split(z_r, self.dim_hidden, dim=-1)
        candidate = torch.cat([x_t, r * h_pre], dim=-1)
        n = torch.tanh(self.update(G, candidate))
        h_t = z * n + (1 - z) * h_pre  # [B, N, H]

        return h_t

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        h = weight.new_zeros(batch_size, self.num_nodes, self.dim_hidden)
        return h


class Encoder(nn.Module):
    """
    First iter on each timestep, then iter on each layer.
    """

    def __init__(self, num_nodes, dim_in, dim_hidden, cheb_k, num_layers=1):
        super(Encoder, self).__init__()
        self.num_nodes = num_nodes
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.dim_in if i == 0 else self.dim_hidden
            self.cell_list.append(
                GRUCell(
                    num_nodes=num_nodes,
                    dim_in=cur_input_dim,
                    dim_hidden=self.dim_hidden,
                    cheb_k=cheb_k,
                )
            )

    def forward(self, G, x_seq, init_h):
        """
        :param G: support adj matrices                          -   [K, N, N]
        :param x_seq: graph feature/signal                      -   [B, T, N, C]
        :param init_h: init hidden state                        -   num_layers * [B, N, H]
        :return output_h: the last hidden state                 -   num_layers * [B, N, H]
        """
        batch_size, seq_len = x_seq.shape[:2]
        if init_h is None:
            init_h = self._init_hidden(batch_size)  # each layer's init h

        current_inputs = x_seq
        output_h = []  # each layer's last h
        for i in range(self.num_layers):
            h = init_h[i]
            h_lst = []  # each step's h for this layer
            for t in range(seq_len):
                h = self.cell_list[i](G, current_inputs[:, t, :, :], h)
                h_lst.append(h)  # T * [B, N, H]
            output_h.append(h)  # num_layers * [B, N, H]
            current_inputs = torch.stack(
                h_lst, dim=1
            )  # [B, T, N, H] input for the next layer
        return output_h

    def _init_hidden(self, batch_size: int):
        h_l = []
        for i in range(self.num_layers):
            h = self.cell_list[i].init_hidden(batch_size)
            h_l.append(h)
        return h_l


class Decoder(nn.Module):
    """
    First iter on each layer, get cur step's pred, then feed back to pred next step.
    """

    def __init__(self, num_nodes, dim_out, dim_hidden, cheb_k, num_layers=1):
        super(Decoder, self).__init__()
        self.num_nodes = num_nodes
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.dim_out if i == 0 else self.dim_hidden
            self.cell_list.append(
                GRUCell(
                    num_nodes=num_nodes,
                    dim_in=cur_input_dim,
                    dim_hidden=self.dim_hidden,
                    cheb_k=cheb_k,
                )
            )

    def forward(self, G, x_t, h):
        """
        :param G: support adj matrices                              -   [K, N, N]
        :param x_t: graph feature/signal                            -   [B, N, C]
        :param h: previous hidden state from the last encoder cell  -   num_layers * [B, N, H]
        :return output: the last hidden state                       -   [B, N, H]
        :return h_lst: hidden state of each layer                   -   num_layers * [B, N, H]
        """
        current_inputs = x_t
        h_lst = []  # each layer's h for this step
        for i in range(self.num_layers):
            h_t = self.cell_list[i](G, current_inputs, h[i])
            h_lst.append(h_t)  # num_layers * [B, N, H]
            current_inputs = h_t  # input for next layer
        output = current_inputs
        return output, h_lst


class GCGRU(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        adj_path,
        adj_type,
        input_dim,
        output_dim,
        horizon,
        rnn_units,
        num_layers=1,
        cheb_k=3,
    ):
        super(GCGRU, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.decoder_dim = self.rnn_units

        adj = load_adj(adj_path, "pkl", adj_type)
        self.P = self.compute_cheby_poly(adj).to(device)

        self.encoder = Encoder(
            num_nodes=self.num_nodes,
            dim_in=self.input_dim,
            dim_hidden=self.rnn_units,
            cheb_k=self.P.shape[0],
            num_layers=self.num_layers,
        )
        self.decoder = Decoder(
            num_nodes=self.num_nodes,
            dim_out=self.input_dim,
            dim_hidden=self.decoder_dim,
            cheb_k=self.P.shape[0],
            num_layers=self.num_layers,
        )
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim))

    def compute_cheby_poly(self, P: list):
        P_k = []
        for p in P:
            p = torch.from_numpy(p).float().T
            T_k = [torch.eye(p.shape[0]), p]  # order 0, 1
            for k in range(2, self.cheb_k):
                T_k.append(2 * torch.mm(p, T_k[-1]) - T_k[-2])  # recurrent to order K
            P_k += T_k
        return torch.stack(P_k, dim=0)  # (K, N, N) or (2*K, N, N) for bidirection

    def forward(self, x):
        init_h = None

        h_lst = self.encoder(self.P, x, init_h)

        deco_input = torch.zeros(
            (x.shape[0], x.shape[2], x.shape[3]), device=x.device
        )  # original initialization [B, N, C], go
        outputs = []
        for t in range(self.horizon):
            output, h_lst = self.decoder(self.P, deco_input, h_lst)
            deco_input = self.proj(output)  # update decoder input
            outputs.append(deco_input)  # T * [B, N, C]

        outputs = torch.stack(outputs, dim=1)  # [B, T, N, C]
        return outputs


if __name__ == "__main__":
    model = GCGRU(
        torch.device("cpu"),
        num_nodes=207,
        adj_path="../../data/METRLA/adj_mx.pkl",
        adj_type="doubletransition",
        input_dim=1,
        output_dim=1,
        horizon=12,
        rnn_units=64,
        num_layers=1,
        cheb_k=3,
    ).cpu()
    summary(model, [64, 12, 207, 1], device="cpu")
