from typing import Tuple
import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, E)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTFusion(nn.Module):
    """
    ViT-based multimodal fusion for tabular feature vectors.
    Each modality vector -> projected embedding token.
    Tokens + [CLS] -> Transformer Encoder -> classifier.

    Inputs:
      radiomics: (B, R)
      feat2d:    (B, D2)
      feat3d:    (B, D3)
      clinical:  (B, C)

    Output:
      logits: (B, num_classes)
    """

    def __init__(
        self,
        input_dims: Tuple[int, int, int, int],
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 2,
        use_modality_type_embed: bool = True,
    ):
        super().__init__()
        r_dim, f2d_dim, f3d_dim, c_dim = input_dims

        # modality projections -> token embeddings
        self.proj_r = nn.Sequential(nn.Linear(r_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.proj_2d = nn.Sequential(nn.Linear(f2d_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.proj_3d = nn.Sequential(nn.Linear(f3d_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.proj_c = nn.Sequential(nn.Linear(c_dim, embed_dim), nn.LayerNorm(embed_dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # token positions (fixed number of tokens: 1 CLS + 4 modality tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 4, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # optional modality-type embeddings (helps the transformer distinguish tokens)
        self.use_modality_type_embed = use_modality_type_embed
        if use_modality_type_embed:
            # indices: 0=CLS, 1=radiomics, 2=2d, 3=3d, 4=clinical
            self.type_embed = nn.Embedding(5, embed_dim)

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, radiomics, feat2d, feat3d, clinical):
        # project to tokens
        t_r = self.proj_r(radiomics).unsqueeze(1)   # (B,1,E)
        t_2d = self.proj_2d(feat2d).unsqueeze(1)
        t_3d = self.proj_3d(feat3d).unsqueeze(1)
        t_c = self.proj_c(clinical).unsqueeze(1)

        x = torch.cat([t_r, t_2d, t_3d, t_c], dim=1)  # (B,4,E)

        # prepend CLS
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)        # (B,1,E)
        x = torch.cat([cls, x], dim=1)                # (B,5,E)

        # add embeddings
        x = x + self.pos_embed
        if self.use_modality_type_embed:
            type_ids = torch.tensor([0, 1, 2, 3, 4], device=x.device).unsqueeze(0).expand(B, -1)
            x = x + self.type_embed(type_ids)

        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token
        logits = self.head(cls_out)
        return logits
