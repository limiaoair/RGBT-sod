## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math

from einops import rearrange


class Transformerstart(nn.Module):
    """
    表示的是交叉开始阶段 (b, 12, 12, 768)
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Transformerstart, self).__init__()

        self.transformer = TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.cross = cross_Attention(dim, num_heads, bias)

    def forward(self, x, y):
        # B,H,W,C-->B,C,H,W
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        x = self.norm1(x)
        y = self.norm1(y)
        f = self.norm1(x+y)

        f = f + self.cross(x, y)
        f = f + self.ffn(self.norm2(f))

        x = x + self.attn(x)
        x = x + self.ffn(self.norm2(x))
        
        y = y + self.attn(y)
        y = y + self.ffn(self.norm2(y))

        return x, y, f


class Transformer_triple_cross(nn.Module):
    """
    表示的是交叉中间阶段的
    input (b, 12, 12, 768),是BHWC,但是要求BLC
    x提供q
    """
    def __init__(self, dim, num_heads, input_res):
        super(Transformer_triple_cross, self).__init__()
        self.transformer = SwinCrossTransformerBlock(dim=dim, input_resolution=input_res,num_heads=num_heads)

    def forward(self, x, y, z):
        # x->RGB  y->main  z->T
        # print(x.shape,y.shape,z.shape) 
        # torch.Size([8, 144, 768]) torch.Size([8, 144, 768]) torch.Size([8, 144, 768])
        xy = self.transformer(x, y)
        xyz = self.transformer(z, xy)
        return xyz
    
class Transformer_triple_cross_stage(nn.Module):
    def __init__(
        self, 
        dim,
        num_heads, 
        depth, 
        input_res,
        norm_layer=nn.LayerNorm, 
        upsample=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            Transformer_triple_cross(
                dim=dim,
                num_heads=num_heads,
                input_res=input_res,
            )
            for i in range(depth)])
        
        if upsample is not None:
            self.upsample = upsample(dim)
        else:
            self.upsample = None
        
    def forward(self, x, y, z):
        # print("1")

        x = to_3d_my(x)
        # y = to_3d_my(y)
        z = to_3d_my(z)
        for blk in self.blocks:
            # print("*")
            # print(x.shape, y.shape, z.shape)
            y = blk(x, y, z)

        # print(y.shape)  #torch.Size([8, 144, 768])
        if self.upsample is not None:
            # print("2")
            y = self.upsample(y)
            # print(y.shape)
        # print(y.shape)
        return y




class cross_Attention(nn.Module):
    """
    y is T
    """
    def __init__(self, dim, num_heads, bias):
        super(cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x, y):
        assert x.shape == y.shape
        # x = x.permute(0, 3, 1, 2)
        # y = y.permute(0, 3, 1, 2)  #
        b,c,h,w = x.shape
        
        qkvx = self.qkv_dwconv(self.qkv(x))
        _,kx,vx = qkvx.chunk(3, dim=1)   
        qkvy = self.qkv_dwconv(self.qkv(y))
        qy,_,_ = qkvy.chunk(3, dim=1) 

        qy = self.q_dwconv(qy)

        # qx = rearrange(qx, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        kx = rearrange(kx, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vx = rearrange(vx, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        qy = rearrange(qy, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # ky = rearrange(ky, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # vy = rearrange(vy, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        qy = torch.nn.functional.normalize(qy, dim=-1)
        kx = torch.nn.functional.normalize(kx, dim=-1)

        attn = (qy @ kx.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ vx)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


#####################################################
## Layer Norm###
def to_3d_my(x):
    return rearrange(x, 'b h w c-> b (h w) c')

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d_my(x,h,w):
    return rearrange(x, 'b (h w) c -> b h w c',h=h,w=w)

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        # 添加了正则化
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
        self.lnorm = nn.LayerNorm(n_feat//2)

    def forward(self, x):
        # print(x.shape)  # x_list[inx], y, z_list[inx]
        B, L, C = x.shape
        aa = math.sqrt(L)
        x = to_4d(x, h=aa, w=aa)
        x.permute(0, 3, 1, 2)
        x = self.body(x)
        print(x.shape)
        x = self.lnorm(x)
        x = to_3d(x)
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        # self.body = nn.Conv2d(n_feat, n_feat*2, kernel_size=1, stride=1, padding=1, bias=False)
        self.body = nn.Conv2d(n_feat, n_feat*2, kernel_size=1, stride=1, bias=False)
        self.shuff = nn.PixelShuffle(2)
        self.lnorm = nn.LayerNorm(n_feat//2)
    def forward(self, x):
        B, L, C = x.shape
        # print("111",x.shape)
        # aa = math.sqrt(L)
        # print(aa)
        x = to_4d(x, h=int(math.sqrt(L)), w=int(math.sqrt(L)))
        # print("22222",x.shape)  # BCHW
        # x.permute(0, 3, 1, 2)
        x = self.body(x)
        x = self.shuff(x)
        # print(x.shape)
        x = to_3d(x)
        # print(x.shape)
        x = self.lnorm(x)
        # print(x.shape)
        return x



##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1





# def Edge_pre(mask, a, b):
#     # 假设seg_map是分割映射（numpy数组）  
#     # 转换为灰度图像（如果已经是二值图像，则可能不需要这一步）  
#     # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask  
    
#     # 使用Canny边缘检测  
#     edges = cv2.Canny(mask, a, b)
#     return edges



# def bdcn_loss2(inputs, targets, l_weight=1.1):
#     # bdcn loss modified in DexiNed

#     targets = targets.long()
#     mask = targets.float()
#     num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
#     num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

#     mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
#     mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
#     inputs= torch.sigmoid(inputs)
#     cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
#     cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
#     return l_weight*cost

# # ------------ cats losses ----------
# def bdrloss(prediction, label, radius,device='cpu'):
#     '''
#     The boundary tracing loss that handles the confusing pixels.
#     '''

#     filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
#     filt.requires_grad = False
#     filt = filt.to(device)

#     bdr_pred = prediction * label
#     pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

#     texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
#     mask = (texture_mask != 0).float()
#     mask[label == 1] = 0
#     pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

#     softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
#     cost = -label * torch.log(softmax_map)
#     cost[label == 0] = 0

#     return torch.sum(cost.float().mean((1, 2, 3)))

# def textureloss(prediction, label, mask_radius, device='cpu'):
#     '''
#     The texture suppression loss that smooths the texture regions.
#     '''
#     filt1 = torch.ones(1, 1, 3, 3)
#     filt1.requires_grad = False
#     filt1 = filt1.to(device)
#     filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
#     filt2.requires_grad = False
#     filt2 = filt2.to(device)

#     pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
#     label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

#     mask = 1 - torch.gt(label_sums, 0).float()

#     loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
#     loss[mask == 0] = 0

#     return torch.sum(loss.float().mean((1, 2, 3)))

# def cats_loss(prediction, label, l_weight=[0.,0.], device='cpu'):
#     # tracingLoss

#     tex_factor,bdr_factor = l_weight
#     balanced_w = 1.1
#     label = label.float()
#     prediction = prediction.float()
#     with torch.no_grad():
#         mask = label.clone()

#         num_positive = torch.sum((mask == 1).float()).float()
#         num_negative = torch.sum((mask == 0).float()).float()
#         beta = num_negative / (num_positive + num_negative)
#         mask[mask == 1] = beta
#         mask[mask == 0] = balanced_w * (1 - beta)
#         mask[mask == 2] = 0

#     prediction = torch.sigmoid(prediction)

#     cost = torch.nn.functional.binary_cross_entropy(
#         prediction.float(), label.float(), weight=mask, reduction='none')
#     cost = torch.sum(cost.float().mean((1, 2, 3)))  # by me
#     label_w = (label != 0).float()
#     textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, device=device)
#     bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, device=device)

#     return cost + bdr_factor * bdrcost + tex_factor * textcost


import torch
from torch import nn

# from .transformer import Mlp
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    # two mlp, fc-relu-drop-fc-relu-drop
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LinearProject(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super(LinearProject, self).__init__()
        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads,  attn_drop=0., proj_drop=0.):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim*2, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, complement):

        # x [B, HW, C]
        B_x, N_x, C_x = x.shape

        x_copy = x

        complement = torch.cat([x, complement], 1)

        B_c, N_c, C_c = complement.shape

        # q [B, heads, HW, C//num_heads]
        q = self.to_q(x).reshape(B_x, N_x, self.num_heads, C_x//self.num_heads).permute(0, 2, 1, 3)
        kv = self.to_kv(complement).reshape(B_c, N_c, 2, self.num_heads, C_c//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_x, N_x, C_x)

        x = x + x_copy

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 1., attn_drop=0., proj_drop=0.,drop_path = 0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(CrossTransformerEncoderLayer, self).__init__()
        self.x_norm1 = norm_layer(dim)
        self.c_norm1 = norm_layer(dim)

        self.attn = MultiHeadCrossAttention(dim, num_heads, attn_drop, proj_drop)

        self.x_norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

        self.drop1 = nn.Dropout(drop_path)
        self.drop2 = nn.Dropout(drop_path)

    def forward(self, x, complement):
        x = self.x_norm1(x)
        complement = self.c_norm1(complement)

        x = x + self.drop1(self.attn(x, complement))
        x = x + self.drop2(self.mlp(self.x_norm2(x)))
        return x
    
class CrossTransformer(nn.Module):
    def __init__(self, x_dim, c_dim, depth, num_heads, mlp_ratio =1., attn_drop=0., proj_drop=0., drop_path =0.):
        super(CrossTransformer, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearProject(x_dim, c_dim, CrossTransformerEncoderLayer(c_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop_path)),
                LinearProject(c_dim, x_dim, CrossTransformerEncoderLayer(x_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop_path))
            ]))

    def forward(self, x, complement):
        # x_meta = torch.cat([x, meta_h], dim=-1).unsqueeze(1)
        # complement_meta = torch.cat([complement, meta_h], dim=-1).unsqueeze(1)
        # x = x.unsqueeze(1)
        # complement = complement.unsqueeze(1)
        for x_attn_complemnt, complement_attn_x in self.layers:
            x = x_attn_complemnt(x, complement=complement) + x
            complement = complement_attn_x(complement, complement=x) + complement
        return x.squeeze(1), complement.squeeze(1)

class CrossTransformer_meta(nn.Module):
    def __init__(self, x_dim, c_dim, depth, num_heads, mlp_ratio =1., attn_drop=0., proj_drop=0., drop_path =0.):
        super(CrossTransformer_meta, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                LinearProject(x_dim, c_dim, CrossTransformerEncoderLayer(c_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop_path))
                )

    def forward(self, x, complement):
        # x_meta = torch.cat([x, meta_h], dim=-1).unsqueeze(1)
        # complement_meta = torch.cat([complement, meta_h], dim=-1).unsqueeze(1)
        x = x.unsqueeze(1)
        complement = complement.unsqueeze(1)
        for x_attn_complemnt in self.layers:
            x = x_attn_complemnt(x, complement=complement) + x

        return x.squeeze(1)


class WindowCrossAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y,mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        y = torch.cat([x,y],1)

        B_c,N_c,C_c = y.shape

        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = self.to_q(x).reshape(B_,N,self.num_heads,C//self.num_heads).permute(0, 2, 1, 3)
        kv = self.to_kv(y).reshape(B_c,N_c,2,self.num_heads,C_c//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = torch.cat([relative_position_bias,relative_position_bias],2)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = torch.cat([mask,mask],2)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N_c) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N_c)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # torch.Size([8, 12, 12, 768]) torch.Size([8, 12, 12, 768])
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # print(x.shape)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # print(windows.shape)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class SwinCrossTransformerBlock(nn.Module):
    r""" Swin Cross Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=6, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        if int(self.input_resolution[0]) < 14:
            self.window_size = int(self.input_resolution[0]) // 2
        else:
            self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = WindowCrossAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, y):
        H, W = self.input_resolution
        # print(H,W) # 12, 12
        B, L, C = x.shape
        # print(x.shape)  # [8, 144, 768]
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        # print(x.shape)  # [8, 144, 768]
        x = x.view(B, H, W, C)
        # print(x.shape)  # torch.Size([8, 12, 12, 768])
        # shortcut2 = y
        y = self.norm1(y)
        y = y.view(B,H,W,C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y,shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        # print(shifted_x.shape, shifted_y.shape)
        # #torch.Size([8, 12, 12, 768]) torch.Size([8, 12, 12, 768])

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows,y_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class SwinCrossTransformer(nn.Module):
    def __init__(self,x_dim, c_dim, depth,input_resolution, num_heads, mlp_ratio =1., attn_drop=0., proj_drop=0., drop_path =0.):
        super(SwinCrossTransformer, self).__init__()

        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                SwinCrossTransformerBlock(x_dim,input_resolution,num_heads,shift_size=0 if (i % 2 == 0) else 7 // 2),#shift_size=0 if (i % 2 == 0) else 7 // 2
                SwinCrossTransformerBlock(c_dim,input_resolution,num_heads,shift_size=0 if (i % 2 == 0) else 7 // 2)
            ]))

    def forward(self,x,y):
        for x_attn_y, y_attn_x in self.layers:
            x = x_attn_y(x,y)+x
            y = y_attn_x(y,x)+y
        return x,y

