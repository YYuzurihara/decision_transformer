import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class StateEmbedding(nn.Module):
    def __init__(self, embed_dim, input_shape):
        super(StateEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, embed_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, max_timestep, context_length, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.embed = nn.Embedding(max_timestep * 3, embed_dim)
        
    def forward(self, timesteps, L):
        batch_size = timesteps.size(0)
        position_ids = torch.arange(3 * L, dtype=torch.long, device=timesteps.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        return self.embed(position_ids)


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, embed_dim, context_length):
        super(DecoderBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, training=False):
        attn_output, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_output)
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        return x


class DecisionTransformer(nn.Module):

    def __init__(self, action_space, max_timestep, context_length=30,
                 n_blocks=6, n_heads=8, embed_dim=128):

        super(DecisionTransformer, self).__init__()

        self.state_shape = (84, 84, 4)
        self.action_space = action_space
        self.context_length = context_length

        self.embed_dim = embed_dim

        self.rtgs_embedding = nn.Linear(1, self.embed_dim)
        nn.init.normal_(self.rtgs_embedding.weight, mean=0.0, std=0.02)

        self.state_embedding = StateEmbedding(self.embed_dim, input_shape=self.state_shape)

        self.action_embedding = nn.Embedding(self.action_space, self.embed_dim)
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)

        self.pos_embedding = PositionalEmbedding(
            max_timestep=max_timestep,
            context_length=context_length,
            embed_dim=embed_dim)

        self.dropout = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([DecoderBlock(n_heads, embed_dim, context_length) for _ in range(n_blocks)])

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(self.embed_dim, self.action_space, bias=False)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, rtgs, states, actions, timesteps, training=False):
        """
        Args:
            rtgs: dtype=torch.float32, shape=(B, L, 1)
            states: dtype=torch.float32, shape=(B, L, 84, 84, 4)
            actions dtype=torch.uint8, shape=(B, L, 1)
            timesteps dtype=torch.int32, shape=(B, 1, 1)
        """
        B, L = rtgs.shape[0], rtgs.shape[1]

        rtgs_embed = torch.tanh(self.rtgs_embedding(rtgs))  #  (B, L, embed_dim)

        states_embed = torch.tanh(self.state_embedding(states))  # (B, L, embed_dim)

        action_embed = torch.tanh(self.action_embedding(actions.squeeze(-1)))  # (B, L, embed_dim)

        pos_embed = self.pos_embedding(timesteps, L)  # (B, 3L, embed_dim)

        tokens = torch.stack([rtgs_embed, states_embed, action_embed], dim=1)  # (B, 3, L, embed_dim)
        tokens = tokens.permute(0, 2, 1, 3).reshape(B, 3*L, self.embed_dim)  # (B, 3L, embed_dim)

        x = self.dropout(tokens + pos_embed)
        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)
        logits = self.head(x)  # (B, 3L, action_space)

        # use only predictions from state
        logits = logits[:, 1::3, :]  # (B, L, action_space)

        return logits

    def sample_action(self, rtgs, states, actions, timestep):
        assert len(rtgs) == len(states) == len(actions) + 1

        L = min(len(rtgs), self.context_length)

        rtgs = torch.tensor(rtgs[-L:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        states = torch.tensor(states[-L:], dtype=torch.float32).unsqueeze(0)
        
        # Add dummy action for the last step
        actions = torch.tensor(actions[-L:] + [0], dtype=torch.uint8).unsqueeze(0).unsqueeze(-1)
        timestep = torch.tensor([timestep], dtype=torch.int32).unsqueeze(0).unsqueeze(-1)

        logits_all = self(rtgs, states, actions, timestep)  # (1, L, A)
        logits = logits_all[0, -1, :]

        probs = F.softmax(logits, dim=-1)
        dist = D.Categorical(probs=probs)
        sampled_action = dist.sample().item()

        return sampled_action, probs
