import torch
from torch import nn
from torch.nn import functional as F

# This 3v3 attention model starts from fresh weights; no trained model is included.
# It uses 168 observation features per player and 90 actions.

def activation(name):
    return {'gelu': nn.GELU, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'silu': nn.SiLU}[name]()

class AttentionBlock(nn.Module):

    def __init__(self, width, act, ffn_ratio=2):
        super().__init__()
        self.width, self.heads = (width, 4)
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * ffn_ratio), activation(act), nn.Linear(width * ffn_ratio, width))

    def forward(self, tokens, key_present=None, readout_only=False):
        batch, length, _ = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens)).reshape(batch, length, 3, self.heads, self.width // self.heads)
        if readout_only:
            tokens = tokens[:, ::7]
        if torch.compiler.is_compiling() and not torch.compiler.is_exporting():
            # Compiled: attention over these few tokens as broadcast reductions, which inductor fuses.
            q, k, v = qkv.unbind(2)
            q = q[:, ::7] if readout_only else q
            scores = (q[:, :, None] * k[:, None]).sum(-1) * (self.width // self.heads) ** -0.5
            if key_present is not None:
                scores = scores.masked_fill(~key_present[:, None, :, None], float('-inf'))
            attended = (scores.softmax(2)[..., None] * v[:, None]).sum(2).to(tokens.dtype)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            q = q[:, :, ::7] if readout_only else q
            mask = None if key_present is None else key_present[:, None, None, :]
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0).transpose(1, 2)
        tokens = tokens + self.projection(attended.reshape(batch, tokens.shape[1], self.width))
        return tokens + self.mlp(self.mlp_norm(tokens))

class EntityAttentionActorCritic(nn.Module):

    def __init__(self, width=64, depth=1, act='gelu', ffn_ratio=2):
        super().__init__()
        self.car_encoder = nn.Linear(19, width)
        self.ball_encoder = nn.Linear(9, width)
        self.context_encoder = nn.Linear(45, width)
        self.types = nn.Embedding(5, width)
        self.register_buffer('type_ids', torch.tensor([0, 1, 1, 2, 2, 2, 3, 4]))
        self.blocks = nn.Sequential(*(AttentionBlock(width, act, ffn_ratio) for _ in range(depth)))
        self.norm = nn.LayerNorm(width)
        self.readout = nn.Sequential(nn.Linear(width * 2, width), activation(act))
        self.actor = nn.Linear(width, 90)
        self.critic = nn.Linear(width, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.types.weight, std=0.02)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def read_tokens(self, tokens, key_present=None):
        for index, block in enumerate(self.blocks):
            tokens = block(tokens, key_present, readout_only=index == len(self.blocks) - 1)
        tokens = self.norm(tokens)
        return self.readout(torch.cat((tokens[:, 0], tokens[:, -1]), -1))

    def encode(self, obs):
        leading = obs.shape[:-1]
        obs = obs.reshape(-1, 168)
        cars = self.car_encoder(obs[:, 12:126].reshape(-1, 6, 19))
        ball = self.ball_encoder(obs[:, 3:12]).unsqueeze(1)
        context = self.context_encoder(torch.cat((obs[:, :3], obs[:, 126:]), -1)).unsqueeze(1)
        tokens = torch.cat((cars, ball, context), 1) + self.types(self.type_ids)
        return (leading, self.read_tokens(tokens))

    def forward(self, obs):
        leading, features = self.encode(obs)
        return (self.actor(features).reshape(*leading, self.actor.out_features), self.critic(features).reshape(leading))

from actions import action_table

PAD_X = (0, 2, 1, 4, 3, 5, 7, 6, 9, 8, 12, 11, 10, 14, 13, 17, 16, 15, 19, 18, 21, 20, 22, 24, 23, 26, 25, 27, 29, 28, 31, 30, 33, 32)
CONTROL_X = (1, -1, 1, -1, -1, 1, 1, 1)

class XReflection(nn.Module):

    def __init__(self):
        super().__init__()
        index = torch.arange(168)
        index[134:] = torch.tensor(PAD_X) + 134
        sign = torch.ones(168)
        sign[[0, 3, 6, 10, 11]] = -1
        for car in range(6):
            base = 12 + car * 19
            sign[[base, base + 3, base + 6, base + 9, base + 13, base + 14]] = -1
        sign[126:134] = torch.tensor(CONTROL_X)
        table = action_table('cpu')
        mirrored = table * torch.tensor(CONTROL_X)
        matches = (mirrored[:, None, :] == table[None, :, :]).all(-1)
        self.register_buffer('observation_index', index)
        self.register_buffer('observation_sign', sign)
        self.register_buffer('action_index', matches.long().argmax(-1))

    def forward(self, obs):
        return obs[..., self.observation_index] * self.observation_sign

class SymmetricAttentionActorCritic(nn.Module):
    obs_size = 168

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.reflection = XReflection()

    def forward(self, observation):
        leading = observation.shape[:-1]
        obs = observation.reshape(-1, 168)
        mirrored = self.reflection(obs)
        differs = obs != mirrored
        first = differs.to(torch.int32).argmax(-1, keepdim=True)
        flip = obs.gather(-1, first) > mirrored.gather(-1, first)
        canonical = torch.where(flip, mirrored, obs)
        logits, values = self.base(canonical)
        reflected_logits = logits[:, self.reflection.action_index]
        fixed = ~differs.any(-1, keepdim=True)
        logits = torch.where(fixed, (logits + reflected_logits) * 0.5, torch.where(flip, reflected_logits, logits))
        return (logits.reshape(*leading, logits.shape[-1]), values.reshape(leading))


def make_model():
    model = SymmetricAttentionActorCritic(EntityAttentionActorCritic(width=256, depth=2, act='gelu'))
    # To start a new run from saved weights with a fresh optimizer:
    # model.load_state_dict(torch.load('checkpoint.pt', map_location='cpu', weights_only=True)['model'])
    return model


def muon_parameters(model):
    excluded = {'car_encoder', 'ball_encoder', 'context_encoder', 'actor', 'critic'}
    return [layer.weight for name, layer in model.base.named_modules()
            if isinstance(layer, nn.Linear) and name.split('.')[0] not in excluded]


def critic_parameters(model):
    return list(model.base.critic.parameters())
