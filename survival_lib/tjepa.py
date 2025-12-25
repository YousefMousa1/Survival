import copy
import math

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Missing dependency: torch. Install it to use T-JEPA embeddings."
    ) from exc


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _get_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


def apply_masks_from_idx(x, masks, offset=0):
    all_x = []
    for m in masks:
        m = m + offset
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


def _sample_num_mask(rng, min_mask, max_mask, num_features):
    num_mask = math.inf
    while num_mask > num_features:
        rand = rng.random()
        num_mask = min_mask + round((max_mask - min_mask) * rand)
    return int(num_mask)


def sample_masks(
    batch_size,
    num_features,
    allow_overlap,
    min_context_share,
    max_context_share,
    min_target_share,
    max_target_share,
    num_preds,
    num_encs,
):
    rng = np.random.default_rng()
    min_context = max(1, round(num_features * min_context_share))
    max_context = max(1, round(num_features * max_context_share))
    min_target = max(1, round(num_features * min_target_share))
    max_target = max(1, round(num_features * max_target_share))

    n_mskd_ctx = _sample_num_mask(rng, min_context, max_context, num_features)
    n_mskd_trg = _sample_num_mask(rng, min_target, max_target, num_features)

    masks_ctx = []
    masks_trg = []
    for _ in range(batch_size):
        all_indices = np.arange(num_features)
        rng.shuffle(all_indices)
        ctx_masks = []
        for _ in range(num_encs):
            ctx = all_indices[:n_mskd_ctx]
            ctx_masks.append(torch.tensor(ctx, dtype=torch.long))
            all_indices = np.setdiff1d(all_indices, ctx)

        trg_masks = []
        for _ in range(num_preds):
            if allow_overlap:
                rng.shuffle(all_indices)
                trg = all_indices[:n_mskd_trg]
            else:
                remaining = all_indices
                rng.shuffle(remaining)
                trg = remaining[:n_mskd_trg]
            trg_masks.append(torch.tensor(trg, dtype=torch.long))

        masks_ctx.append(ctx_masks)
        masks_trg.append(trg_masks)

    masks_ctx = [torch.stack([m[i] for m in masks_ctx]) for i in range(num_encs)]
    masks_trg = [torch.stack([m[i] for m in masks_trg]) for i in range(num_preds)]
    return masks_ctx, masks_trg


class NumericTokenizer(nn.Module):
    def __init__(self, n_features, embed_dim, n_cls_tokens):
        super().__init__()
        self.n_features = n_features
        self.n_cls_tokens = n_cls_tokens
        self.weight = nn.Parameter(torch.empty(n_features, embed_dim))
        self.bias = nn.Parameter(torch.empty(n_features, embed_dim))
        self.cls_tokens = nn.Parameter(torch.empty(n_cls_tokens, embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.cls_tokens, a=math.sqrt(5))

    def forward(self, x):
        batch_size = x.size(0)
        feat_tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(
            0
        )
        if self.n_cls_tokens > 0:
            cls = self.cls_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.cat([cls, feat_tokens], dim=1)
        return feat_tokens


class TabularEncoder(nn.Module):
    def __init__(
        self,
        n_features,
        embed_dim,
        num_layers,
        num_heads,
        mlp_dim,
        dropout,
        activation,
        n_cls_tokens,
    ):
        super().__init__()
        self.n_cls_tokens = n_cls_tokens
        self.tokenizer = NumericTokenizer(n_features, embed_dim, n_cls_tokens)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation=activation,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, masks=None):
        tokens = self.tokenizer(x)
        if masks is not None:
            tokens = apply_masks_from_idx(tokens, masks, offset=self.n_cls_tokens)
        return self.encoder(tokens)


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        num_features,
        model_hidden_dim,
        pred_embed_dim,
        num_layers,
        num_heads,
        p_dropout,
        activation,
        dim_feedforward,
    ):
        super().__init__()
        self.num_features = num_features
        self.pred_embed_dim = pred_embed_dim
        self.predictor_emb = nn.Linear(model_hidden_dim, pred_embed_dim, bias=True)
        self.layer_norm = nn.LayerNorm(pred_embed_dim)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_features, pred_embed_dim), requires_grad=False
        )
        pos = self._sincos_pos_embed(pred_embed_dim, np.arange(num_features))
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(pos).float().unsqueeze(0)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=pred_embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=p_dropout,
            batch_first=True,
            activation=activation,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.predictor_norm = nn.LayerNorm(pred_embed_dim)
        self.predictor_proj = nn.Linear(pred_embed_dim, model_hidden_dim, bias=True)
        nn.init.normal_(self.mask_token, std=0.02)

    def _sincos_pos_embed(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega
        out = np.einsum("m,d->md", pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        return np.concatenate([emb_sin, emb_cos], axis=1)

    def forward(self, x, masks_enc, masks_pred):
        batch_size = x.shape[0] // len(masks_enc)
        x = self.predictor_emb(x)
        x = self.layer_norm(x)

        pos_ctx = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        pos_ctx = apply_masks_from_idx(pos_ctx, masks_enc)
        x = x + pos_ctx

        pos_pred = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        pos_pred = apply_masks_from_idx(pos_pred, masks_pred)
        pred_tokens = self.mask_token.repeat(pos_pred.size(0), pos_pred.size(1), 1)
        pred_tokens = pred_tokens + pos_pred

        x = x.repeat(len(masks_pred), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        x = self.transformer(x)
        x = self.predictor_norm(x)
        x = x[:, -pred_tokens.size(1) :]
        x = self.predictor_proj(x)
        return x


@torch.no_grad()
def _ema_update(target, online, momentum):
    for tgt, src in zip(target.parameters(), online.parameters()):
        tgt.data.mul_(momentum).add_(src.data, alpha=1.0 - momentum)


def train_tjepa(
    x,
    embed_dim=64,
    num_layers=4,
    num_heads=8,
    mlp_dim=256,
    dropout=0.1,
    activation="relu",
    pred_embed_dim=64,
    pred_num_layers=2,
    pred_num_heads=4,
    pred_dropout=0.1,
    pred_dim_feedforward=256,
    mask_allow_overlap=False,
    mask_min_ctx_share=0.2,
    mask_max_ctx_share=0.4,
    mask_min_trgt_share=0.2,
    mask_max_trgt_share=0.4,
    mask_num_preds=4,
    mask_num_encs=1,
    n_cls_tokens=1,
    epochs=100,
    batch_size=64,
    lr=1e-4,
    weight_decay=1e-5,
    momentum=0.996,
    seed=42,
    device=None,
):
    _set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x_tensor = torch.tensor(x.values, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_features = x.shape[1]
    context_encoder = TabularEncoder(
        n_features,
        embed_dim,
        num_layers,
        num_heads,
        mlp_dim,
        dropout,
        activation,
        n_cls_tokens,
    ).to(device)
    target_encoder = copy.deepcopy(context_encoder).to(device)
    for param in target_encoder.parameters():
        param.requires_grad = False

    predictor = TransformerPredictor(
        num_features=n_features,
        model_hidden_dim=embed_dim,
        pred_embed_dim=pred_embed_dim,
        num_layers=pred_num_layers,
        num_heads=pred_num_heads,
        p_dropout=pred_dropout,
        activation=activation,
        dim_feedforward=pred_dim_feedforward,
    ).to(device)

    optim = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    context_encoder.train()
    predictor.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            masks_enc, masks_pred = sample_masks(
                batch.size(0),
                n_features,
                mask_allow_overlap,
                mask_min_ctx_share,
                mask_max_ctx_share,
                mask_min_trgt_share,
                mask_max_trgt_share,
                mask_num_preds,
                mask_num_encs,
            )
            masks_enc = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]

            with torch.no_grad():
                h = target_encoder(batch)
                h = apply_masks_from_idx(h, masks_pred, offset=n_cls_tokens)

            z = context_encoder(batch, masks_enc)
            pred = predictor(z, masks_enc, masks_pred)
            loss = loss_fn(pred, h)

            optim.zero_grad()
            loss.backward()
            optim.step()
            _ema_update(target_encoder, context_encoder, momentum)

    context_encoder.eval()
    target_encoder.eval()
    with torch.no_grad():
        tokens = target_encoder(x_tensor.to(device))
        if n_cls_tokens > 0:
            cls_tokens = tokens[:, :n_cls_tokens, :].mean(dim=1)
        else:
            cls_tokens = tokens.mean(dim=1)
    return cls_tokens.cpu().numpy()


def tjepa_forward_only(
    x,
    embed_dim=64,
    num_layers=4,
    num_heads=8,
    mlp_dim=256,
    dropout=0.1,
    activation="relu",
    n_cls_tokens=1,
    seed=42,
    device=None,
):
    _set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x_tensor = torch.tensor(x.values, dtype=torch.float32)
    n_features = x.shape[1]
    encoder = TabularEncoder(
        n_features,
        embed_dim,
        num_layers,
        num_heads,
        mlp_dim,
        dropout,
        activation,
        n_cls_tokens,
    ).to(device)
    encoder.eval()
    with torch.no_grad():
        tokens = encoder(x_tensor.to(device))
        if n_cls_tokens > 0:
            cls_tokens = tokens[:, :n_cls_tokens, :].mean(dim=1)
        else:
            cls_tokens = tokens.mean(dim=1)
    return cls_tokens.cpu().numpy()
