import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class TemporalAttention(nn.Module):
    def __init__(self, in_dim, num_nodes=None, cut_size=0):
        super(TemporalAttention, self).__init__()
        self.K = 8

        if in_dim % self.K != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(in_dim // self.K)
        self.key_proj = LinearCustom()
        self.value_proj = LinearCustom()

        self.projection1 = nn.Linear(in_dim,in_dim)
        self.projection2 = nn.Linear(in_dim,in_dim)

    def forward(self, query, key, value, parameters):
        batch_size = query.shape[0]

        # [batch_size, num_step, N, K * head_size]
        key = self.key_proj(key, parameters[0])
        value = self.value_proj(value, parameters[1])

        # [K * batch_size, num_step, N, head_size]
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        # query: [K * batch_size, N, 1, head_size]
        # key:   [K * batch_size, N, head_size, num_step]
        # value: [K * batch_size, N, num_step, head_size]
        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))

        attention = torch.matmul(query, key)  # [K * batch_size, N, num_step, num_step]
        attention /= (self.head_size ** 0.5)

        # normalize the attention scores
        # attention = self.mask * attention
        attention = F.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)  # [batch_size * head_size, num_step, N, K]
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        # projection
        x = self.projection1(x)
        x = F.tanh(x)
        x = self.projection2(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_dim, support=None, num_nodes=None):
        super(SpatialAttention, self).__init__()
        self.support = support
        self.K = 8

        if in_dim % self.K != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(in_dim // self.K)
        self.linear = LinearCustom()
        self.projection1 = nn.Linear(in_dim, in_dim)
        self.projection2 = nn.Linear(in_dim, in_dim)

    def forward(self, x, parameters):
        batch_size = x.shape[0]
        # [batch_size, 1, N, K * head_size]
        # query = self.linear(x, parameters[2])
        key = self.linear(x, parameters[0])
        value = self.linear(x, parameters[1])

        # [K * batch_size, num_step, N, head_size]
        query = torch.cat(torch.split(x, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))  # [K * batch_size, N, num_step, num_step]
        attention /= (self.head_size ** 0.5)

        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, value)  # [batch_size * head_size, num_step, N, K]
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        # projection
        x = self.projection1(x)
        x = F.relu(x)
        x = self.projection2(x)
        return x


class LinearCustom(nn.Module):

    def __init__(self):
        super(LinearCustom, self).__init__()

    def forward(self, inputs, parameters):
        weights, biases = parameters[0], parameters[1]
        if len(weights.shape) > 3:
            return torch.matmul(inputs.unsqueeze(-2), weights.unsqueeze(1).repeat(1, inputs.shape[1], 1, 1, 1)).squeeze(
                -2) + biases.unsqueeze(1).repeat(1, inputs.shape[1], 1, 1)
        return torch.matmul(inputs, weights) + biases

class STWA(nn.Module):
    def __init__(self, 
                 device, 
                 num_nodes, 
                 input_dim=1, 
                 output_dim=1, 
                 channels=16, 
                 dynamic=True, 
                 lag=12, 
                 horizon=12, 
                 memory_size=16):
        super(STWA, self).__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.channels = channels
        self.dynamic = dynamic
        self.start_fc = nn.Linear(in_features=input_dim, out_features=self.channels)
        self.memory_size = memory_size

        self.layers = nn.ModuleList(
            [
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes, cuts=12,
                      cut_size=6, no_proxies=2, memory_size=memory_size),
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes, cuts=3,
                      cut_size=4, no_proxies=2, memory_size=memory_size),
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes, cuts=1,
                      cut_size=3, no_proxies=2, memory_size=memory_size),
            ])

        self.skip_layers = nn.ModuleList([
            nn.Linear(in_features=12 * channels, out_features=256),
            nn.Linear(in_features=3 * channels, out_features=256),
            nn.Linear(in_features=1 *channels, out_features=256),
        ])

        self.projections = nn.Sequential(*[
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, horizon)])

        if self.dynamic:
            self.mu_estimator = nn.Sequential(*[
                nn.Linear(input_dim * lag, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, memory_size)
            ])

            self.logvar_estimator = nn.Sequential(*[
                nn.Linear(input_dim * lag, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, memory_size)
            ])

    def forward(self, x):
        if self.dynamic:
            mu = self.mu_estimator(x.transpose(3, 1).squeeze())
            logvar = self.logvar_estimator(x.transpose(3, 1).squeeze())
            z_data = reparameterize(mu, logvar)
        else:
            z_data = 0


        x = self.start_fc(x)
        batch_size = x.size(0)

        skip = 0
        for layer, skip_layer in zip(self.layers, self.skip_layers):
            x = layer(x, z_data)
            skip_inp = x.transpose(2, 1).reshape(batch_size, self.num_nodes, -1)
            skip = skip + skip_layer(skip_inp)

        x = torch.relu(skip)

        return self.projections(x).transpose(2, 1).unsqueeze(-1)


class Layer(nn.Module):
    def __init__(self, device, input_dim, num_nodes, cuts, cut_size, dynamic, memory_size, no_proxies):
        super(Layer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.cuts = cuts
        self.cut_size = cut_size
        self.no_proxies = no_proxies
        self.proxies = nn.Parameter(torch.randn(1, cuts * no_proxies, self.num_nodes, input_dim).to(device),
                                    requires_grad=True).to(device)

        self.temporal_att = TemporalAttention(input_dim, num_nodes=num_nodes, cut_size=cut_size)
        self.spatial_att = SpatialAttention(input_dim, num_nodes=num_nodes)

        if self.dynamic:
            self.mu = nn.Parameter(torch.randn(num_nodes, memory_size).to(device), requires_grad=True).to(device)
            self.logvar = nn.Parameter(torch.randn(num_nodes, memory_size).to(device), requires_grad=True).to(device)

        self.temporal_parameter_generators = nn.ModuleList([
            ParameterGenerator(memory_size=memory_size, input_dim=input_dim, output_dim=input_dim,
                               num_nodes=num_nodes, dynamic=dynamic) for _ in range(2)
        ])

        self.spatial_parameter_generators = nn.ModuleList([
            ParameterGenerator(memory_size=memory_size, input_dim=input_dim, output_dim=input_dim,
                               num_nodes=num_nodes, dynamic=dynamic) for _ in range(2)
        ])

        self.aggregator = nn.Sequential(*[
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        ])

    def forward(self, x, z_data):
        # x shape: B T N C
        batch_size = x.size(0)

        if self.dynamic:
            z_sample = reparameterize(self.mu, self.logvar)
            z_data = z_data + z_sample

        temporal_parameters = [layer(x, z_data) for layer in self.temporal_parameter_generators]
        spatial_parameters = [layer(x, z_data) for layer in self.spatial_parameter_generators]

        data_concat = []
        out = 0
        for i in range(self.cuts):
            # shape is (B, cut_size, N, C)
            t = x[:, i * self.cut_size:(i + 1) * self.cut_size, :, :]

            proxies = self.proxies[:, i * self.no_proxies: (i + 1) * self.no_proxies]
            proxies = proxies.repeat(batch_size, 1, 1, 1) + out
            t = torch.cat([proxies, t], dim=1)

            out = self.temporal_att(t[:, :self.no_proxies, :, :], t, t, temporal_parameters)
            out = self.spatial_att(out, spatial_parameters)
            out = (self.aggregator(out) * out).sum(1, keepdim=True)
            data_concat.append(out)

        return torch.cat(data_concat, dim=1)

class ParameterGenerator(nn.Module):
    def __init__(self, memory_size, input_dim, output_dim, num_nodes, dynamic):
        super(ParameterGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic

        if self.dynamic:
            # print('Using DYNAMIC')
            self.weight_generator = nn.Sequential(*[
                nn.Linear(memory_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, input_dim * output_dim)
            ])
            self.bias_generator = nn.Sequential(*[
                nn.Linear(memory_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, output_dim)
            ])
        else:
            # print('Using FC')
            self.weights = nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)
            self.biases = nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def forward(self, x, memory=None):
        if self.dynamic:
            weights = self.weight_generator(memory).view(x.shape[0], self.num_nodes, self.input_dim, self.output_dim)
            biases = self.bias_generator(memory).view(x.shape[0], self.num_nodes, self.output_dim)
        else:
            weights = self.weights
            biases = self.biases
        return weights, biases
