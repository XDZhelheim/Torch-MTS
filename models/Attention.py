import torch.nn as nn
import torch
from torchinfo import summary


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).
    
    Make sure the tensor is permuted to correct shape before attention.
    
    E.g. 
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.
    
    Also, it supports different src and tgt length.
    
    But must `src length == K length == V length`.
    
    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, mask=False):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.ln2(residual + out)

        return out


class Attention(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps,
        out_steps,
        input_dim=1,
        output_dim=1,
        model_dim=64,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.positional_proj = nn.Linear(1, model_dim)
        self.temporal_proj = nn.Linear(in_steps, out_steps)
        self.output_proj = nn.Linear(model_dim, self.output_dim)

        self.attn_layers = nn.Sequential(
            *[
                SelfAttentionLayer(model_dim, feed_forward_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+timeinday=2)
        x = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, input_dim+timeinday)

        position = x[..., 1:]
        x = x[..., :1]

        x = self.input_proj(x)
        pe = self.positional_proj(position)
        x += pe  # (batch_size, num_nodes, in_steps, model_dim)

        out = self.attn_layers(x)  # (batch_size, num_nodes, in_steps, model_dim)

        out = out.transpose(-1, -2)  # (batch_size, num_nodes, model_dim, in_steps)
        out = self.temporal_proj(out)  # (batch_size, num_nodes, model_dim, out_steps)
        out = self.output_proj(
            out.transpose(-1, -2)
        )  # (batch_size, num_nodes, out_steps, output_dim)

        return out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)


class STAttention(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps,
        out_steps,
        steps_per_day=288,
        input_dim=1,
        output_dim=1,
        input_embedding_dim=12,
        tod_embedding_dim=12,
        dow_embedding_dim=12,
        adaptive_embedding_dim=72,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(in_steps * num_nodes, adaptive_embedding_dim))
        )
        self.temporal_proj = nn.Linear(in_steps, out_steps)
        self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers = nn.Sequential(
            *[
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]
        x = x.view(
            x.shape[0], x.shape[1] * x.shape[2], x.shape[3]
        )  # (batch_size, in_steps * num_nodes, input_dim+tod+dow)

        tod = x[..., 1]
        dow = x[..., 2]
        x = x[..., :1]

        x = self.input_proj(
            x
        )  # (batch_size, in_steps * num_nodes, input_embedding_dim)
        tod_emb = self.tod_embedding(
            (tod * self.steps_per_day).long()
        )  # (batch_size, in_steps * num_nodes, tod_embedding_dim)
        dow_emb = self.dow_embedding(
            dow.long()
        )  # (batch_size, in_steps * num_nodes, dow_embedding_dim)
        adp_emb = self.adaptive_embedding.expand(
            size=(
                batch_size,
                self.num_nodes * self.in_steps,
                self.adaptive_embedding_dim,
            )
        )
        x = torch.cat(
            [x, tod_emb, dow_emb, adp_emb], dim=-1
        )  # (batch_size, in_steps * num_nodes, model_dim)

        out = self.attn_layers(x)  # (batch_size, in_steps * num_nodes, model_dim)

        out = out.view(
            batch_size, self.in_steps, self.num_nodes, self.model_dim
        ).transpose(
            1, 3
        )  # (batch_size, model_dim, num_nodes, in_steps)
        out = self.temporal_proj(out)  # (batch_size, model_dim, num_nodes, out_steps)
        out = self.output_proj(
            out.transpose(1, 3)
        )  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    model = STAttention(207, 12, 12)
    summary(model, [1, 12, 207, 3])
