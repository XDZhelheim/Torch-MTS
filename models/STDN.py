import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.sparse as sp
import pickle

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # print(f"Number of isolated points: {isolated_point_num}")
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def cal_lape(adj_mx):
    lape_dim = 32
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    return laplacian_pe

def load_adj(pkl_filename):
    try:
        # METRLA and PEMSBAY
        _, _, adj_mx = load_pickle(pkl_filename)
    except ValueError:
        # PEMS3478
        adj_mx = load_pickle(pkl_filename)
        
    return adj_mx

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None ):
        super(conv2d_, self).__init__()
        self.activation = activation
        # self.dropout = dropout
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]

        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        # (batch_size, num_step, num_vertex, D)
        x = x.permute(0, 3, 2, 1) # (batch_size , D, num_step, num_vertex)

        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))

        x = self.conv(x)

        x = self.batch_norm(x)

        if self.activation is not None :
            x = self.activation(x)

        return x.permute(0, 3, 2, 1)

class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list 
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        # print("before x:", x.shape)
        for conv in self.convs:
            x = conv(x)
        # print("after x: ", x.shape)
        return x
    

class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x
    
class TEmbedding(nn.Module):
    '''
    X: (batch_size, num_his+num_pred, 2) (dayofweek, timeofday)
    T: num of time steps in one day
    return: (batch_size, num_his+num_pred, num_vertex, D)
    '''
    def __init__(self, input_dim, D, num_nodes, bn_decay,) -> None:
        super(TEmbedding, self).__init__()
        self.FC = FC(input_dims=[input_dim, D, D], units=[D, D, D], activations=[torch.relu, torch.relu, torch.sigmoid],
            bn_decay=bn_decay)
        
    def forward(self, X, SE, T, num_vertex, num_his):
        dayofweek = torch.empty(X.shape[0], X.shape[1], 7).to(X.device)
        timeofday = torch.empty(X.shape[0], X.shape[1], T).to(X.device)
        for i in range(X.shape[0]): # shape[0] = batch_size
            dayofweek[i] = F.one_hot(X[..., 0][i].to(torch.int64) % 7, 7) 
        for j in range(X.shape[0]):
            timeofday[j] = F.one_hot(X[..., 1][j].to(torch.int64) % T, T)
        X = torch.cat((timeofday, dayofweek), dim=-1) # (batch_size, num_his+num_pred, 7+T)
        X = X.unsqueeze(dim = 2)
        add_vertex = torch.zeros(1,1,num_vertex,1).to(X.device)
        X = X + add_vertex
        X = self.FC(X)
        X = torch.sin(X)
        His = X[:, :num_his]
        Pred = X[:, num_his:]
        del dayofweek, timeofday, add_vertex, X
        return His + F.relu(SE), Pred

class SEmbedding(nn.Module):
    def __init__(self, D):
        super(SEmbedding, self).__init__()

        self.LaplacianPE1 = nn.Linear(32, 32)
        self.Norm1 = nn.LayerNorm(32, elementwise_affine=False)
        self.act = nn.LeakyReLU()
        self.LaplacianPE2 = nn.Linear(32, D)
        self.Norm2 = nn.LayerNorm(D, elementwise_affine=False)

    def forward(self, lpls, batch_size, pred_steps):
        lap_pos_enc = self.Norm2(self.LaplacianPE2(self.act(self.Norm1(self.LaplacianPE1(lpls)))))
        tensor_neb = lap_pos_enc.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).repeat(1, pred_steps, 1, 1)
        return torch.sigmoid(tensor_neb)

class Trend(nn.Module):
    '''
    X: (batch_size, num_step, num_vertex, D)
    TEmbedding： (batch_size, num_step, num_vertex, D)
    return: (batch_size, num_step, num_vertex, D)
    '''
    def __init__(self):
        super(Trend, self).__init__()
        
    def forward(self, X, STEmbedding): 
        return torch.mul(X, STEmbedding)

class Seasonal(nn.Module):
    '''
    X: (batch_size, num_step, num_vertex, D)
    return: (batch_size, num_step, num_vertex, D)
    '''
    def __init__(self):
        super(Seasonal, self).__init__()

    def forward(self, X, Trend):
        return X-Trend

        
class MAB_new(nn.Module):
    def __init__(self, K,d,input_dim,output_dim,bn_decay):
        super(MAB_new, self).__init__()
        D=K*d
        self.K = K
        self.d=d
        self.FC_q = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=output_dim, activations=F.relu,
                     bn_decay=bn_decay)
    def forward(self, Q, K,batch_size):
        query = self.FC_q(Q)
        key = self.FC_k(K)
        value = self.FC_v(K)
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        result = torch.matmul(attention, value)
        result = result.permute(0, 2, 1, 3)
        result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        result = self.FC(result)
        return result

class AttentionDecoder(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    Y:      [batch_size, num_step, num_vertex, D]
    TE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''
    def __init__(self, K, d,num_of_vertices,set_dim, bn_decay):
        super(AttentionDecoder, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.num_of_vertices=num_of_vertices
        self.set_dim = set_dim
        self.I = nn.Parameter(torch.Tensor(1,set_dim,self.num_of_vertices, 3*D))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB_new(K, d, 3*D, 3*D, bn_decay)
        self.mab1 = MAB_new(K, d, 3*D, D, bn_decay)

    def forward(self, X, TE, SE, mask):
    # def forward(self, X, SE, mask):
        batch_size = X.shape[0]
        mid = X 
        X = torch.cat((X, TE, SE), dim=-1)
        # X = torch.add(X, TE)
        # [batch_size, num_step, num_vertex, K * d]
        I = self.I.repeat(X.size(0), 1, 1, 1)
        H = self.mab0(I, X, batch_size)
        result = self.mab1(X, H, batch_size)
        return torch.add(mid, result)
    
    
class GRU(nn.Module):
    def __init__(self, outfea):
        super(GRU, self).__init__()
        self.ff = nn.Linear(2*outfea, 2*outfea)
        self.zff = nn.Linear(2*outfea, outfea)
        self.outfea = outfea

    def forward(self, x, xh):
        r, u = torch.split(torch.sigmoid(self.ff(torch.cat([x, xh], -1))), self.outfea, -1)
        z = torch.tanh(self.zff(torch.cat([x, r*xh], -1)))
        x = u * z + (1-u) * xh
        return x

class GRUEncoder(nn.Module):
    def __init__(self, outfea, num_step):
        super(GRUEncoder, self).__init__()
        self.gru = nn.ModuleList([GRU(outfea) for i in range(num_step)])
        
    def forward(self, x):
        B,T,N,F = x.shape
        hidden_state = torch.zeros([B,N,F]).to(x.device)
        output = []
        for i in range(T):
            gx = x[:,i,:,:]
            gh = hidden_state
            hidden_state = self.gru[i](gx, gh)
            output.append(hidden_state)
        output = torch.stack(output, 1)
        return output

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()

# GCN
class gcn(nn.Module):
    """
    x:          [batch_size, num_step, num_vertex, D]
    support:    [num_vertex, D, D]
    """
    def __init__(self, c_in, c_out, dropout = 0.3, support_len = 1, order = 2, bn_decay = 0.1):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len+1) * c_in
        self.mlp = FC(c_in, c_out, activations=F.relu, bn_decay=bn_decay)
        self.dropout = dropout
        self.order = order
    def forward(self, x, support):
        x = x.transpose(1, 3)
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order+1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2            
        h = torch.cat(out, dim = 1)
        h = h.transpose(1,3)
        h = self.mlp(h)
        h = h.transpose(1,3)
        h = F.dropout(h, self.dropout, training=self.training)
        h = h.transpose(1,3)
        return h


class STDN(nn.Module):
    def __init__(
        self, 
        device,
        num_of_vertices=207,
        adj_path="../data/METRLA/adj_mx.pkl",
        L=2,
        K=16,
        d=8,
        node_miss_rate=0.1,
        T_miss_len=12,
        order=3,
        reference=3,
        time_slice_size=5,
        num_his=12,
        num_pred=12,
        in_channels=1,
        out_channels=1,
        bn_decay=0.1
        ):
        super(STDN, self).__init__()
        
        adj_mx = load_adj(adj_path)
        lpls = cal_lape(adj_mx)
        self.lpls = torch.from_numpy(np.array(lpls, dtype='float32')).type(torch.FloatTensor).to(device)
        
        self.L=L
        self.K=K
        self.d=d
        D = K * d
        
        self.node_miss_rate=node_miss_rate
        self.T_miss_len=T_miss_len
        self.order = order
        
        set_dim = reference
        self.num_his = num_his
        self.input_dim = int(1440/time_slice_size) + 7
        
        self.num_pred = num_pred
        self.num_of_vertices = num_of_vertices
        
        self.TEmbedding = TEmbedding(self.input_dim, D, self.num_of_vertices, bn_decay)
        self.SEmbedding = SEmbedding(D)
        self.Trend = Trend()
        self.Seasonal = Seasonal()
        self.FeedForward_for_t = FeedForward([D,D], res_ln=True)
        self.FeedForward_for_s = FeedForward([D,D], res_ln=True)
        self.GRU_Trend = GRUEncoder(D, self.num_his)
        self.GRU_Seasonal = GRUEncoder(D, self.num_his)
        self.Decoder = nn.ModuleList([AttentionDecoder(K, d, self.num_of_vertices, set_dim, bn_decay) for _ in range(L)])

        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)  # in_channels=3
        self.FC_2 = FC(input_dims=[D, D], units=[D,out_channels], activations=[F.relu, None],
                       bn_decay=bn_decay)

        # dynamic GCN
        self.nodevec_p1 = nn.Parameter(torch.randn(int(1440/time_slice_size), D).to(device), requires_grad=True).to(device)
        self.nodevec_p2 = nn.Parameter(torch.randn(num_of_vertices, D).to(device), requires_grad=True).to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_of_vertices, D).to(device), requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(D, D, D).to(device), requires_grad=True).to(device)
        self.GCN = gcn(D, D, order = self.order)
        
    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        # print(adp.shape)
        return adp   
        
    def forward(self, X, y_tod_dow):
        # input
        x_tod = (X[..., 0, 1]*288).long() # (B, T)
        x_dow = X[..., 0, 2].long()
        y_tod = (y_tod_dow[..., 0, 0]*288).long()
        y_dow = y_tod_dow[..., 0, 1].long()
        TE = torch.concat([torch.stack([x_dow, x_tod], dim=-1), torch.stack([y_dow, y_tod], dim=-1)], dim=1) # (B, T1+T2, 2)
        
        X = X[..., [0]]
        X = self.FC_1(X)
        ind = TE[:,0,1]
        # dynamic graph construction
        adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        new_supports = [adp]
        X  = self.GCN(X, new_supports)

        SE = self.SEmbedding(self.lpls, X.shape[0] , self.num_pred)
        his, pred = self.TEmbedding(TE, SE, self.input_dim - 7, self.num_of_vertices, self.num_his)
        trend = self.Trend(X,his)
        seasonal = self.Seasonal(X,trend)
        trend = self.FeedForward_for_t(trend)
        seasonal = self.FeedForward_for_s(seasonal)
        # encoder
        trend = self.GRU_Trend(trend)
        seasonal = self.GRU_Seasonal(seasonal)

        result = trend + seasonal
        # decoder
        for net in self.Decoder:
            result = net(result, pred, SE, None)
        result = self.FC_2(result)
        del TE, his, trend, seasonal
        return result

if __name__ == "__main__":
    from torchinfo import summary

    model = STDN("cpu", num_of_vertices=207, adj_path="../data/METRLA/adj_mx.pkl").cpu()
    summary(model, [[64, 12, 207, 3], [64, 12, 207, 2]], device="cpu")
