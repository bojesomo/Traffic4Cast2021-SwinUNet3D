import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from backbones import CSWinTransformer, SwinTransformer, StarTransformer
from timm.models.layers import trunc_normal_
from backbones.heads import UPerHead


default_cfg = {
# ## Standard
# 'tiny_224':
#     {'patch_size': 4, 'embed_dim': 64, 'depth': [1, 2, 21, 1], 'split_size': [1, 2, 7, 7], 'num_heads': [2, 4, 8, 16],
#      'mlp_ratio': 4, 'img_size': 224, 'drop_path_rate': 0.1},
# 'small_224':
#     {'patch_size': 4, 'embed_dim': 64, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 7, 7], 'num_heads': [2, 4, 8, 16],
#      'mlp_ratio': 4, 'img_size': 224, 'drop_path_rate': 0.3},
# 'base_224':
#     {'patch_size': 4, 'embed_dim': 96, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 7, 7], 'num_heads': [4, 8, 16, 32],
#      'mlp_ratio': 4, 'img_size': 224, 'drop_path_rate': 0.5},
# 'large_224':
#     {'patch_size': 4, 'embed_dim': 144, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 7, 7],
#      'num_heads': [6, 12, 24, 24], 'mlp_ratio': 4, 'img_size': 224, 'drop_path_rate': 0.7},
# 'base_384':
#     {'patch_size': 4, 'embed_dim': 96, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 12, 12],
#      'num_heads': [4, 8, 16, 32], 'mlp_ratio': 4, 'img_size': 384, 'drop_path_rate': 0.5},
# 'large_384':
#     {'patch_size': 4, 'embed_dim': 144, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 12, 12],
#      'num_heads': [6, 12, 24, 24], 'mlp_ratio': 4, 'img_size': 384, 'drop_path_rate': 0.7},
# ## W4C
# 'tiny_W4C':
#     {'patch_size': 4, 'embed_dim': 64, 'depth': [1, 2, 21, 1], 'split_size': [1, 2, 8, 8], 'num_heads': [2, 4, 8, 16],
#      'mlp_ratio': 4, 'img_size': 256, 'drop_path_rate': 0.1},
# 'small_W4C':
#     {'patch_size': 4, 'embed_dim': 64, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 8, 8], 'num_heads': [2, 4, 8, 16],
#      'mlp_ratio': 4, 'img_size': 256, 'drop_path_rate': 0.3},
# 'base_W4C':
#     {'patch_size': 4, 'embed_dim': 96, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 8, 8], 'num_heads': [4, 8, 16, 32],
#      'mlp_ratio': 4, 'img_size': 256, 'drop_path_rate': 0.5},
# 'large_W4C':
#     {'patch_size': 4, 'embed_dim': 144, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 8, 8],
#      'num_heads': [6, 12, 24, 24], 'mlp_ratio': 4, 'img_size': 256, 'drop_path_rate': 0.7},
## T4C
'tiny':
    {'patch_size': 4, 'embed_dim': 64, 'depth': [1, 2, 21, 1], 'split_size': [1, 2, 16, 16], 'num_heads': [2, 4, 8, 16],
     'mlp_ratio': 4, 'img_size': [495, 436], 'drop_path_rate': 0.1},
'small':
    {'patch_size': 4, 'embed_dim': 64, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 16, 16], 'num_heads': [2, 4, 8, 16],
     'mlp_ratio': 4, 'img_size': [495, 436], 'drop_path_rate': 0.3},
'base':
    {'patch_size': 4, 'embed_dim': 96, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 16, 16],
     'num_heads': [4, 8, 16, 32], 'mlp_ratio': 4, 'img_size': [495, 436], 'drop_path_rate': 0.5},
'large':
    {'patch_size': 4, 'embed_dim': 144, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 16, 16],
     'num_heads': [6, 12, 24, 24], 'mlp_ratio': 4, 'img_size': [495, 436], 'drop_path_rate': 0.7},
}


class CSWin(CSWinTransformer):
    def __init__(self, model_name='base_224', **kwargs):
        super().__init__(**default_cfg[model_name], **kwargs)


class Model(nn.Module):
    def __init__(self, model_name: str = 'base', n_classes: int = 48, **kwargs) -> None:
        super().__init__()
        self.backbone = CSWin(model_name, extract_features=True, **kwargs)
        # self.backbone.get_features()

        # print(self.backbone.embed_dims)
        self.decode_head = UPerHead(self.backbone.embed_dims, 256, (1, 2, 3, 6), n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # def init_pretrained(self, pretrained: str = None) -> None:
    #     if pretrained:
    #         self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        # print([t.shape for t in y])
        y = self.decode_head(y)  # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)  # to original image shape
        return y
