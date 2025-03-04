import torch
import torch.nn as nn
import torch_redstone as rst
from einops import rearrange
from .pointnet_util import PointNetSetAbstraction


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *extra_args, **kwargs):
        return self.fn(self.norm(x), *extra_args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., rel_pe = False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.rel_pe = rel_pe
        if rel_pe:
            self.pe = nn.Sequential(nn.Conv2d(3, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x, centroid_delta):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        pe = self.pe(centroid_delta) if self.rel_pe else 0
        dots = (torch.matmul(q, k.transpose(-1, -2)) + pe) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., rel_pe = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, rel_pe = rel_pe)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, centroid_delta):
        for attn, ff in self.layers:
            x = attn(x, centroid_delta) + x
            x = ff(x) + x
        return x


class PointPatchTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, sa_dim, patches, prad, nsamp, in_dim=3, dim_head=64, rel_pe=False, patch_dropout=0) -> None:
        super().__init__()
        self.patches = patches
        self.patch_dropout = patch_dropout
        self.sa = PointNetSetAbstraction(npoint=patches, radius=prad, nsample=nsamp, in_channel=in_dim + 3, mlp=[64, 64, sa_dim], group_all=False)
        self.lift = nn.Sequential(nn.Conv1d(sa_dim + 3, dim, 1), rst.Lambda(lambda x: torch.permute(x, [0, 2, 1])), nn.LayerNorm([dim]))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, 0.0, rel_pe)

    def forward(self, features):
        self.sa.npoint = self.patches
        if self.training:
            self.sa.npoint -= self.patch_dropout
        print("input", features.shape)
        centroids, feature = self.sa(features[:, :3], features)
        # print("f", feature.shape, 'c', centroids.shape)
        x = self.lift(torch.cat([centroids, feature], dim=1))

        x = rst.supercat([self.cls_token, x], dim=-2)
        centroids = rst.supercat([centroids.new_zeros(1), centroids], dim=-1)

        centroid_delta = centroids.unsqueeze(-1) - centroids.unsqueeze(-2)
        x = self.transformer(x, centroid_delta)

        return x[:, 0]


class PatchTokenExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 获取内部PointPatchTransformer模型
        if hasattr(model, 'ppat'):
            self.ppat = model.ppat
        else:
            self.ppat = model
            
    def forward(self, features):
        # 访问模型的内部变量
        self.sa = self.ppat.sa
        centroids, feature = self.sa(features[:, :3], features)
        x = self.ppat.lift(torch.cat([centroids, feature], dim=1))
        
        # 添加CLS token
        x = torch.cat([self.ppat.cls_token.unsqueeze(0).repeat(x.size(0), 1, 1), x], dim=1)
        centroids = torch.cat([centroids.new_zeros(centroids.size(0), 3, 1), centroids], dim=-1)
        
        centroid_delta = centroids.unsqueeze(-1) - centroids.unsqueeze(-2)
        x = self.ppat.transformer(x, centroid_delta)
        
        # 返回所有tokens，而不仅仅是CLS token
        return {
            'cls_token': x[:, 0],       # CLS token [B, 512]
            'patch_tokens': x[:, 1:],   # Patch tokens [B, N, 512] 
            'centroids': centroids      # Patch中心点坐标 [B, 3, N+1]
        }
    
class Projected(nn.Module):
    def __init__(self, ppat, proj) -> None:
        super().__init__()
        self.ppat = ppat
        self.proj = proj

    def forward(self, features: torch.Tensor):
        return self.proj(self.ppat(features))
