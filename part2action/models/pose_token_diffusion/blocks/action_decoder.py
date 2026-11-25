"""DDPM action chunks with separate visual/scene/pose/robot memory tokens."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CrossAttentionDenoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim
        self.input = nn.Linear(config.action_dim, width)
        self.position = nn.Parameter(torch.randn(1, config.action_horizon, width)*.02)
        layer = nn.TransformerDecoderLayer(width, config.attention_heads, width*2,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.layers = nn.TransformerDecoder(layer, config.decoder_layers)
        for block in self.layers.layers:
            for attention in (block.self_attn, block.multihead_attn):
                nn.init.xavier_uniform_(attention.in_proj_weight)
                nn.init.zeros_(attention.in_proj_bias)
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, config.action_dim))

    def forward(self, noisy_action, time_embedding, memory, valid_mask):
        queries = self.input(noisy_action.to(memory.dtype))
        queries = queries+self.position.to(queries.dtype)+time_embedding[:, None]
        return self.output(self.layers(queries, memory, memory_key_padding_mask=~valid_mask))


class PoseTokenDiffusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.denoiser = CrossAttentionDenoiser(config)
        self.type_embedding = nn.Parameter(torch.randn(6, config.hidden_dim)*.02)
        self.time = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(),
                                  nn.Linear(config.hidden_dim, config.hidden_dim))
        self.register_buffer('action_scale', torch.tensor(config.action_scale, dtype=torch.float32))
        self.register_buffer('action_offset', torch.tensor(config.action_offset, dtype=torch.float32))
        time = torch.linspace(0, 1, config.diffusion_steps+1, dtype=torch.float64)
        cumulative = torch.cos((time+.008)/1.008*math.pi/2).square()
        cumulative = cumulative/cumulative[0]
        betas = (1-cumulative[1:]/cumulative[:-1]).clamp(max=.999).float()
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1-betas)
        self.register_buffer('alpha_bars', (1-betas).cumprod(0))

    def context(self, encoded, pose_memory, region):
        scene_count = min(encoded.scene_tokens.shape[1], self.config.scene_token_budget)
        indices = torch.linspace(0, encoded.scene_tokens.shape[1]-1, scene_count,
                                 device=encoded.scene_tokens.device).long()
        parts = [encoded.visual_tokens, encoded.scene_tokens[:, indices], region.tokens,
                 encoded.language_tokens, encoded.robot_tokens, pose_memory.tokens]
        masks = [None, encoded.scene_valid_mask[:, indices] if encoded.scene_valid_mask is not None else None,
                 None, encoded.language_valid_mask, None, pose_memory.valid_mask]
        tokens, validity = [], []
        dtype = encoded.visual_tokens.dtype
        for i, (part, valid) in enumerate(zip(parts, masks)):
            tokens.append(part.to(dtype)+self.type_embedding[i].to(dtype))
            validity.append(torch.ones(part.shape[:2], device=part.device, dtype=torch.bool) if valid is None else valid)
        return torch.cat(tokens, 1), torch.cat(validity, 1)

    def time_embedding(self, steps, dtype):
        width = self.config.hidden_dim
        half = width//2
        frequencies = torch.exp(-math.log(10000.)*torch.arange(half, device=steps.device)/max(half-1, 1))
        values = steps.float()[:, None]*frequencies[None]
        values = torch.cat([values.sin(), values.cos()], -1)
        return self.time(F.pad(values, (0, width-values.shape[-1])).to(dtype))

    def denoise(self, x, steps, context, mask):
        return self.denoiser(x, self.time_embedding(steps, context.dtype), context, mask)

    def forward(self, encoded, pose_memory, *, region, target_actions=None):
        context, mask = self.context(encoded, pose_memory, region)
        if target_actions is None:
            return {'action_chunk': self.sample(context, mask)}
        normalized = (target_actions.float()-self.action_offset)/self.action_scale
        steps = torch.randint(self.config.diffusion_steps, (normalized.shape[0],), device=normalized.device)
        noise = torch.randn_like(normalized)
        alpha_bar = self.alpha_bars[steps, None, None]
        noisy = alpha_bar.sqrt()*normalized+(1-alpha_bar).sqrt()*noise
        return dict(action_noise_pred=self.denoise(noisy, steps, context, mask),
                    action_noise=noise, diffusion_t=steps)

    @torch.no_grad()
    def sample(self, context, mask):
        # The encoder and memory assembly have already run once for this observation.
        x = torch.randn(context.shape[0], self.config.action_horizon, self.config.action_dim,
                        device=context.device, dtype=torch.float32)
        for step in reversed(range(self.config.diffusion_steps)):
            steps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            noise = self.denoise(x, steps, context, mask).float()
            beta, alpha, alpha_bar = self.betas[step], self.alphas[step], self.alpha_bars[step]
            previous = self.alpha_bars[step-1] if step else x.new_tensor(1.)
            clean = ((x-(1-alpha_bar).sqrt()*noise)/alpha_bar.sqrt()).clamp(
                -self.config.diffusion_clip_sample, self.config.diffusion_clip_sample)
            mean = previous.sqrt()*beta/(1-alpha_bar)*clean+alpha.sqrt()*(1-previous)/(1-alpha_bar)*x
            if step:
                x = mean+(beta*(1-previous)/(1-alpha_bar)).sqrt()*torch.randn_like(x)
            else:
                x = mean
        return x*self.action_scale+self.action_offset
