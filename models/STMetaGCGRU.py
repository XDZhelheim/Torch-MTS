import torch
import torch.nn as nn
import numpy as np
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


class STMetaGCN(nn.Module):
    def __init__(self, cheb_k):
        super(STMetaGCN, self).__init__()
        self.cheb_k = cheb_k

    def set_weights(self, W, b):
        # W   (B, N, k*gcn_input_dim, gcn_hidden_dim)
        # b   (B, N, gcn_hidden_dim)
        self.W = W
        self.b = b

    def forward(self, G, x):
        # G   (K, N, N)
        # x   (B, N, gcn_input_dim)

        support_list = []
        for k in range(self.cheb_k):
            support = torch.einsum(
                "nm,bmi->bni", G[k, :, :], x
            )  # (B, N, gcn_input_dim)
            support_list.append(support)
        support_cat = torch.cat(support_list, dim=-1)  # (B, N, k*gcn_input_dim)

        output = (
            torch.einsum("bni,bnih->bnh", support_cat, self.W) + self.b
        )  # (B, N, gcn_hidden_dim)

        return output


class STMetaGCGRUCell(nn.Module):
    def __init__(self, cheb_k=3, gru_hidden_dim=64):
        super(STMetaGCGRUCell, self).__init__()

        self.gru_hidden_dim = gru_hidden_dim
        self.conv_gate = STMetaGCN(cheb_k=cheb_k)
        self.update = STMetaGCN(cheb_k=cheb_k)

    def forward(self, G, xt, h):
        # xt    (B, N, input_dim)
        # h     (B, N, gru_hidden_dim)
        # Wgcn  (B, N, k*(input_dim+gru_hidden_dim), gru_hidden_dim)
        # bgcn  (B, N, gru_hidden_dim)

        combined = torch.cat([xt, h], dim=-1)
        combined_conv = torch.sigmoid(self.conv_gate(G, combined))
        z, r = torch.split(combined_conv, self.gru_hidden_dim, dim=-1)
        candidate = torch.cat((xt, r * h), dim=-1)
        n = torch.tanh(self.update(G, candidate))

        h = z * n + (1 - z) * h  # (B, N, gru_hidden_dim)

        return h

    def set_weights(self, W_conv_gate, b_conv_gate, W_update, b_update):
        self.conv_gate.set_weights(W_conv_gate, b_conv_gate)
        self.update.set_weights(W_update, b_update)


class STMetaGCGRUEncoder(nn.Module):
    def __init__(
        self,
        cheb_k=3,
        input_dim=1,
        gru_hidden_dim=64,
        st_embedding_dim=95,
        learner_hidden_dim=128,
        z_dim=32,
    ):
        super(STMetaGCGRUEncoder, self).__init__()
        self.cheb_k = cheb_k
        self.input_dim = input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.st_embedding_dim = st_embedding_dim
        self.learner_hidden_dim = learner_hidden_dim
        self.z_dim = z_dim

        self.cell = STMetaGCGRUCell(cheb_k=cheb_k, gru_hidden_dim=gru_hidden_dim)

        self.learner_w_conv_gate = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                learner_hidden_dim,
                cheb_k * (input_dim + gru_hidden_dim) * 2 * gru_hidden_dim,
            ),
        )

        self.learner_b_conv_gate = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 2 * gru_hidden_dim),
        )

        self.learner_w_update = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                learner_hidden_dim,
                cheb_k * (input_dim + gru_hidden_dim) * gru_hidden_dim,
            ),
        )

        self.learner_b_update = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, gru_hidden_dim),
        )

    def forward(self, G, x, meta_input):
        # G            (K, N, N)
        # x            (B, T_in, N, input_dim)
        # meta_input   (B, N, st_embedding_dim+z_dim)

        batch_size = x.shape[0]
        in_steps = x.shape[1]
        num_nodes = x.shape[2]

        W_conv_gate = self.learner_w_conv_gate(meta_input).view(
            batch_size,
            num_nodes,
            self.cheb_k * (self.input_dim + self.gru_hidden_dim),
            2 * self.gru_hidden_dim,
        )
        b_conv_gate = self.learner_b_conv_gate(meta_input).view(
            batch_size, num_nodes, 2 * self.gru_hidden_dim
        )
        W_update = self.learner_w_update(meta_input).view(
            batch_size,
            num_nodes,
            self.cheb_k * (self.input_dim + self.gru_hidden_dim),
            self.gru_hidden_dim,
        )
        b_update = self.learner_b_update(meta_input).view(
            batch_size, num_nodes, self.gru_hidden_dim
        )

        self.cell.set_weights(W_conv_gate, b_conv_gate, W_update, b_update)

        h = torch.zeros(batch_size, num_nodes, self.gru_hidden_dim, device=x.device)

        h_each_step = []
        for t in range(in_steps):
            h = self.cell(G, x[:, t, ...], h)  # (B, N, gru_hidden_dim)
            h_each_step.append(h)  # T_in*(B, N, gru_hidden_dim)

        h_each_step = torch.stack(
            h_each_step, dim=1
        )  # (B, T_in, N, gru_hidden_dim) input for next layer

        return h_each_step, h


class STMetaGCGRUDecoder(nn.Module):
    def __init__(
        self,
        cheb_k=3,
        input_dim=1,
        gru_hidden_dim=64,
        st_embedding_dim=95,
        learner_hidden_dim=128,
        z_dim=32,
    ):
        super(STMetaGCGRUDecoder, self).__init__()
        self.cheb_k = cheb_k
        self.input_dim = input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.st_embedding_dim = st_embedding_dim
        self.learner_hidden_dim = learner_hidden_dim
        self.z_dim = z_dim

        self.cell = STMetaGCGRUCell(cheb_k=cheb_k, gru_hidden_dim=gru_hidden_dim)

        self.learner_w_conv_gate = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                learner_hidden_dim,
                cheb_k * (input_dim + gru_hidden_dim) * 2 * gru_hidden_dim,
            ),
        )

        self.learner_b_conv_gate = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 2 * gru_hidden_dim),
        )

        self.learner_w_update = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                learner_hidden_dim,
                cheb_k * (input_dim + gru_hidden_dim) * gru_hidden_dim,
            ),
        )

        self.learner_b_update = nn.Sequential(
            nn.Linear(st_embedding_dim + z_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, gru_hidden_dim),
        )

    def set_weights(self, meta_input, batch_size, num_nodes):
        # meta_input   (B, N, st_embedding_dim+z_dim)

        W_conv_gate = self.learner_w_conv_gate(meta_input).view(
            batch_size,
            num_nodes,
            self.cheb_k * (self.input_dim + self.gru_hidden_dim),
            2 * self.gru_hidden_dim,
        )
        b_conv_gate = self.learner_b_conv_gate(meta_input).view(
            batch_size, num_nodes, 2 * self.gru_hidden_dim
        )
        W_update = self.learner_w_update(meta_input).view(
            batch_size,
            num_nodes,
            self.cheb_k * (self.input_dim + self.gru_hidden_dim),
            self.gru_hidden_dim,
        )
        b_update = self.learner_b_update(meta_input).view(
            batch_size, num_nodes, self.gru_hidden_dim
        )

        self.cell.set_weights(W_conv_gate, b_conv_gate, W_update, b_update)

    def forward(self, G, xt, h):
        # G            (K, N, N)
        # xt           (B, N, input_dim)

        h = self.cell(G, xt, h)  # (B, N, gru_hidden_dim)

        return h


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


class STMetaGCGRU(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        node_emb_file,
        adj_path,
        adj_type,
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
        cheb_k=3,
    ):
        super(STMetaGCGRU, self).__init__()

        self.USE_META_DECODER = False

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

        self.cheb_k = cheb_k
        adj = load_adj(adj_path, "pkl", adj_type)
        self.P = self.compute_cheby_poly(adj).to(device)

        self.node_embedding = torch.FloatTensor(np.load(node_emb_file)["data"]).to(
            device
        )

        self.tod_onehots = torch.eye(24, device=device)
        self.dow_onehots = torch.eye(7, device=device)

        self.encoders = nn.ModuleList(
            [
                STMetaGCGRUEncoder(
                    self.P.shape[0],
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
                STMetaGCGRUEncoder(
                    self.P.shape[0],
                    gru_hidden_dim,
                    gru_hidden_dim,
                    self.st_embedding_dim,
                    learner_hidden_dim,
                    z_dim,
                )
            )

        if seq2seq:
            if self.USE_META_DECODER:
                self.decoders = nn.ModuleList(
                    [
                        STMetaGCGRUDecoder(
                            self.P.shape[0],
                            input_dim,
                            gru_hidden_dim,
                            self.st_embedding_dim,
                            learner_hidden_dim,
                            z_dim,
                        )
                    ]
                )
                for _ in range(num_layers - 1):
                    self.decoders.append(
                        STMetaGCGRUDecoder(
                            self.P.shape[0],
                            gru_hidden_dim,
                            gru_hidden_dim,
                            self.st_embedding_dim,
                            learner_hidden_dim,
                            z_dim,
                        )
                    )
            else:
                self.decoder = Decoder(
                    num_nodes=self.num_nodes,
                    dim_out=self.output_dim,
                    dim_hidden=self.gru_hidden_dim,
                    cheb_k=self.P.shape[0],
                    num_layers=self.num_layers,
                )
            self.proj = nn.Linear(gru_hidden_dim, output_dim)
        else:
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

        gru_input = x  # (B, T_in, N, 1)
        h_each_layer = []  # last step's h of each layer
        for encoder in self.encoders:
            gru_input, last_h = encoder(self.P, gru_input, meta_input)

            h_each_layer.append(last_h)  # num_layers*(B, N, gru_hidden_dim)

        if self.seq2seq:
            decoder_input = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(x.device)
            out = []  # y_pred for each step
            if self.USE_META_DECODER:
                for decoder in self.decoders:
                    decoder.set_weights(meta_input, batch_size, self.num_nodes)
                for _ in range(self.out_steps):
                    for i in range(self.num_layers):
                        decoder_input = self.decoders[i](
                            self.P, decoder_input, h_each_layer[i]
                        )  # (B, N, gru_hidden_dim)
                        h_each_layer[i] = decoder_input
                    decoder_input = self.proj(
                        decoder_input
                    )  # map last layer's h to y_pred
                    out.append(decoder_input)  # T_out*(B, N, output_dim)
            else:
                for t in range(self.out_steps):
                    output, h_each_layer = self.decoder(
                        self.P, decoder_input, h_each_layer
                    )
                    decoder_input = self.proj(output)
                    out.append(decoder_input)  # T_out*(B, N, output_dim)
            out = torch.stack(out, dim=1)  # (B, T_out, N, output_dim)
        else:
            out = h_each_layer[-1]  # (B, N, gru_hidden_dim) last layer last step's h
            out = self.decoder(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )  # (B, N, T_out, output_dim=1)
            out = out.transpose(1, 2)  # (B, T_out, N, 1)

        return out  # (B, T_out, N, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


if __name__ == "__main__":
    model = STMetaGCGRU(
        torch.device("cpu"),
        207,
        "../data/METRLA/spatial_embeddings.npz",
        "../data/METRLA/adj_mx.pkl",
        "doubletransition",
        learner_hidden_dim=128,
        gru_hidden_dim=32,
        z_dim=32,
        num_layers=1,
        seq2seq=True,
        cheb_k=1,
    )
    summary(model, [64, 12, 207, 3], device="cpu")
