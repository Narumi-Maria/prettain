# coding=utf-8
'''
在原始VIT的基础上，加了一个backbone
'''

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    # 对输入的x做一个layerNormalization层归一化，然后再放到Attention模块中做自注意力
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 分头
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Attention计算公式
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 把每个头沿着d的维度合并
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 做一个残差结构
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, backbone, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=4, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # patch大小要能被图片大小整除
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # patch的个数
        num_patches = (image_size // patch_size) ** 2
        # 把每个patch展成一维后的大小
        patch_dim = channels * patch_size ** 2
        # 最后如何代表图片，cls-class token，mean-所有token做平均
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # VIT前面是否要过一个resnet
        self.backbone = backbone
        # Rearrange切分图片，imgsize*imgsize->(h*patch_size)*(w*patch_size)->(h*w)*(patch_size*patch_size)
        # 共有h*w个patch，nn.Linear把每个patch转为token，转换后为b*num_patches*token_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        # 编码位置信息,class_token也有
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 可训练的class_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # resnet layer4 output shape: b,c,8,8, if resnet18, c=512
        # n = (image_size // patch_size) ** 2 and n must be greater than 16. if use resnet_layer4_backbone, patch_size = 1 or 2
        if self.backbone:
            img = self.backbone(img)

        # 经过patch_embedding后，x为batchsize*num_patches*token_dim
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # repeat在batchsize的维度，将class_token给每个图片都复制一份
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # 将class_token拼接到num_patches的维度
        x = torch.cat((cls_tokens, x), dim=1)
        # 定位
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # 输入到transformer中进行自注意力的特征提取
        x = self.transformer(x)

        # 此时如果pool是cls就只取class_token，如果是mean就取所有token的平均
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    vit_model = ViT(backbone = None,
                image_size = 256,
                patch_size = 32,
                num_classes = 2,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            )
    input=torch.rand([1,4,256,256])
    x=vit_model(input)