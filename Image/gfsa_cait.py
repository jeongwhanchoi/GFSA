# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial

# from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.vision_transformer import _cfg
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'gfsa_cait_M48', 'gfsa_cait_M36', 'gfsa_cait_M4',
    'gfsa_cait_S36', 'gfsa_cait_S24','gfsa_cait_S24_224',
    'gfsa_cait_XS24','gfsa_cait_XXS24','gfsa_cait_XXS24_224',
    'gfsa_cait_XXS36','gfsa_cait_XXS36_224'
]


class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attnscale=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attnscale = attnscale
        if attnscale:
            self.lamb = nn.Parameter(torch.zeros(num_heads), requires_grad=True)

    def forward(self, x ):
        
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        # del q; del k
        attn = attn.softmax(dim=-1)

        if self.attnscale:
            h=4
            identity = torch.eye(attn.shape[-1],attn.shape[-1]).to(attn.device)
            identity = identity[None, None, ...]
            reaction = (h-1) * (attn - identity)@attn
            reaction += attn
            beta = self.lamb[None, :, None, None]
            attn = attn + beta * reaction 

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x   

        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls     

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 2 * self.dim + self.dim * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += self.dim * self.dim
        return flops
        
class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4):
        super().__init__()

        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    
    def forward(self, x, x_cls):
        
        u = torch.cat((x_cls,x),dim=1)
        
        
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        
        return x_cls 

    def flops(self, N):
        flops = 0
        # norm1
        flops += self.dim
        # attn
        flops += self.attn.flops(N)
        # gamma_1 * x
        flops += self.dim
        # mlp
        flops += 2 * self.dim * self.dim * self.mlp_ratio
        # gamma_2 * x
        flops += self.dim
        # norm2
        flops += self.dim
        return flops
        
class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attnscale=True):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        
        self.proj_drop = nn.Dropout(proj_drop)

        self.attnscale = attnscale
        if attnscale:
            self.lamb = nn.Parameter(torch.zeros(num_heads), requires_grad=True)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale , qkv[1], qkv[2] 
        del qkv
    
        attn = (q @ k.transpose(-2, -1))

        # pre-softmax communication
        attn = self.proj_l(attn.permute(0,2,3,1)).permute(0,3,1,2)
                
        attn = attn.softmax(dim=-1)
  
        # post-softmax communication
        attn = self.proj_w(attn.permute(0,2,3,1)).permute(0,3,1,2)

        if self.attnscale:
            h=4
            identity = torch.eye(attn.shape[-1],attn.shape[-1]).to(attn.device)
            identity = identity[None, None, ...]
            reaction = (h-1) * (attn - identity)@attn
            reaction += attn
            beta = self.lamb[None, :, None, None]
            attn = attn + beta * reaction 
           

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # attn = self.proj_l(attn)
        flops += N * N * self.num_heads * self.num_heads
        # attn = self.proj_w(attn)
        flops += N * N * self.num_heads * self.num_heads
        # attnscale
        flops += self.num_heads * N * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class LayerScale_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention_talking_head,
                 Mlp_block=Mlp,init_values=1e-4):
        super().__init__()
        
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):        
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x 
    
    def flops(self, N):
        flops = 0
        # norm1
        flops += self.dim * N
        # attn
        flops += self.attn.flops(N)
        # gamma_1 * x
        flops += self.dim * N
        # mlp
        flops += 2 * N * self.dim * self.dim * self.mlp_ratio
        # gamma_2 * x
        flops += self.dim * N
        # norm2
        flops += self.dim * N
        return flops
    
    
class cait_models(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = LayerScale_Block,
                 block_layers_token = LayerScale_Block_CA,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = partial(Attention_talking_head, attnscale=True),Mlp_block=Mlp,
                 init_scale=1e-4,
                 Attention_block_token_only=partial(Class_Attention, attnscale=False),
                 Mlp_block_token_only= Mlp, 
                 depth_token_only=2,
                 mlp_ratio_clstk = 4.0,
                 **kwargs):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)] 
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only,init_values=init_scale)
            for i in range(depth_token_only)])
            
        self.norm = norm_layer(embed_dim)


        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x,cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        
        x = self.head(x)

        return x

    def flops(self):
        # patch embed
        Ho = Wo = self.img_size // self.patch_size
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size * self.patch_size)
        # if self.norm is not None:
        #     flops += Ho * Wo * self.embed_dim

        # attn blocks
        for i, layer in enumerate(self.blocks):
            flops += layer.flops(Ho * Wo)
        # attn CA blocks
        for i, layer in enumerate(self.blocks_token_only):
            flops += layer.flops(Ho * Wo)
        flops += Ho * Wo * self.embed_dim

        # mlp readout
        flops += self.num_features * self.num_classes
        return flops

@register_model
def gfsa_cait_XXS24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def gfsa_cait_XXS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 
@register_model
def gfsa_cait_XXS36_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def gfsa_cait_XXS36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def gfsa_cait_XS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=288, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 




@register_model
def gfsa_cait_S24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224, patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S24_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def gfsa_cait_S24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384, patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def gfsa_cait_S36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=384, depth=36, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)

    return model 





@register_model
def gfsa_cait_M36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384, patch_size=16, embed_dim=768, depth=36, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/M36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)

    return model 


@register_model
def gfsa_cait_M48(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 448 , patch_size=16, embed_dim=768, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/M48_448.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model         
