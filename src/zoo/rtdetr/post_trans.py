import copy

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

def visual_attention(data):
    data = data.cpu()
    data = data.detach().numpy()

    plt.xlabel('x')
    plt.ylabel('score')
    plt.imshow(data)
    plt.show()

class Attention_msa(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, cls_score=None, fg_score=None, return_attention=False, ave=True, sim_thresh=0.75, use_mask=False):
        B, N, C = x_cls.shape
        # 1, b*500, 256
        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, 1, 4, b*30, 64)

        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # (1, 4, b*500, 64)  # make torchscript happy (cannot use tensor as tuple)

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)   # (1, 4, b*500, 64)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)   # (1, 4, b*500, 64)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)   # (1, 4, b*500, 64)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)    #(1,4,b*500,b*500)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)    #(1,4,b*500,b*500)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)    #(1,4,b*500,b*500)
        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask #(1, 4, b*500, b*500)
        attn_cls = attn_cls.softmax(dim=-1) #(1, 4, b*500, b*500)
        attn = self.attn_drop(attn_cls) #(1, 4, b*500, b*500)

        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C) #(1, b*500, 256)

        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)  #(1, b*500, 256)
        x_cls = torch.cat([x, x_ori], dim=-1)  #(1, b*500, 512)

        if ave:
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')  #(b*500, b*500)
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')  #(b*500, b*500)

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads  #(b*500, b*500)
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)  #(b*500, b*500)
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads  #(b*500, b*500)

            sim_round2 = torch.softmax(sim_attn, dim=-1)  #(b*500, b*500)
            sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))  #(b*500, b*500)
            return x_cls, None, sim_round2
        else:
            return x_cls, None, None

class MSA_yolov(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super().__init__()
        self.msa = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)

    def find_similar_round2(self, features, sort_results):
        key_feature = features[0]  #(b*500, 512)
        support_feature = features[0]  #(b*500, 512)
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (sort_results @ support_feature)  #(b*500, 512)  # .transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)  #(b*500, 1025)
        return cls_feature

    def forward(self, out_feat, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True, use_mask=False):
        trans_cls, trans_reg, sim_round2 = self.msa(out_feat.reshape(1, -1, 256), cls_score, fg_score, sim_thresh=sim_thresh, ave=ave, use_mask=use_mask)#(1, b*500, 512), None, (b*500, b*500)
        msa = self.linear1(trans_cls)  #(1,b*500,512)
        msa = self.find_similar_round2(msa, sim_round2)   #(b*500, 1024)
        # print(msa.shape, out_feat.shape)
        out = self.linear2(msa).reshape(out_feat.shape[0], out_feat.shape[1], 256)   #(b, 500, 256)
        return out


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=512, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, input):
        """Forward pass for the AIFI transformer layer."""
        # pos_embed = self.build_2d_sincos_position_embedding(input.shape[0], input.shape[1], c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(input, pos=None)
        #return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        x = x.reshape(input.shape[0], input.shape[1], 256)
        return x.contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]



# import copy
#
# import torch
# import torch.nn as nn
# from matplotlib import pyplot as plt
#
# def visual_attention(data):
#     data = data.cpu()
#     data = data.detach().numpy()
#
#     plt.xlabel('x')
#     plt.ylabel('score')
#     plt.imshow(data)
#     plt.show()
#
# class Attention_msa(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
#         # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
#         # qkv_bias : Is it matter?
#         # qk_scale, attn_drop,proj_drop will not be used
#         # object = Attention(dim,num head)
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = scale  # qk_scale or head_dim ** -0.5
#
#         self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#
#     def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False, ave=True, sim_thresh=0.75, use_mask=False):
#         B, N, C = x_cls.shape
#         # 1, b*30, 320
#         qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B, num_head, N, c
#         qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, 1, 4, b*30, 80)
#         q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # (1, 4, b*30, 80)  # make torchscript happy (cannot use tensor as tuple)
#         q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]  # (1, 4, b*30, 80)
#
#         q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)   # (1, 4, b*30, 80)
#         k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)   # (1, 4, b*30, 80)
#         q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)   # (1, 4, b*30, 80)
#         k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)   # (1, 4, b*30, 80)
#         v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)   # (1, 4, b*30, 80)
#
#         if cls_score == None:
#             cls_score = 1
#         else:
#             cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)    #(1,4,b*30,b*30)
#
#         if fg_score == None:
#             fg_score = 1
#         else:
#             fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)    #(1,4,b*30,b*30)
#
#         attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)    #(1,4,b*30,b*30)
#         if use_mask:
#             # only reference object with higher confidence..
#             cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
#             fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
#         else:
#             cls_score_mask = fg_score_mask = 1
#
#         # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
#         # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
#         # visual_attention(cls_score[0, 0, :, :])
#         # visual_attention(cls_score_mask[0,0,:,:])
#
#         attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask #(1, 4, b*30, b*30)
#         attn_cls = attn_cls.softmax(dim=-1) #(1, 4, b*30, b*30)
#         attn_cls = self.attn_drop(attn_cls) #(1, 4, b*30, b*30)
#
#         attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score * fg_score_mask #(1, 4, b*30, b*30)
#         attn_reg = attn_reg.softmax(dim=-1) #(1, 4, b*30, b*30)
#         attn_reg = self.attn_drop(attn_reg) #(1, 4, b*30, b*30)
#
#         attn = (attn_reg + attn_cls) / 2 #(1, 4, b*30, b*30)
#         x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C) #(1, b*30, 320)
#
#         x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)  #(1, b*30, 320)
#         x_cls = torch.cat([x, x_ori], dim=-1)  #(1, b*30, 640)
#         #
#
#         if ave:
#             ones_matrix = torch.ones(attn.shape[2:]).to('cuda')  #(b*30, b*30)
#             zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')  #(b*30, b*30)
#
#             attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads  #(b*30, b*30)
#             sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)  #(b*30, b*30)
#             sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads  #(b*30, b*30)
#
#             sim_round2 = torch.softmax(sim_attn, dim=-1)  #(b*30, b*30)
#             sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))  #(b*30, b*30)
#             return x_cls, None, sim_round2
#         else:
#             return x_cls, None, None
#
# class MSA_yolov(nn.Module):
#
#     def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
#         super().__init__()
#         self.msa = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
#         self.linear1 = nn.Linear(2 * dim, 2 * dim)
#         self.linear2 = nn.Linear(4 * dim, out_dim)
#
#     def find_similar_round2(self, features, sort_results):
#         key_feature = features[0]  #(b*30, 640)
#         support_feature = features[0]  #(b*30, 640)
#         if not self.training:
#             sort_results = sort_results.to(features.dtype)
#         soft_sim_feature = (
#                     sort_results @ support_feature)  #(b*30, 640)  # .transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
#         cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)  #(b*30, 1280)
#         return cls_feature
#
#     def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True, use_mask=False):
#         trans_cls, trans_reg, sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score, sim_thresh=sim_thresh, ave=ave,
#                                                     use_mask=use_mask)#(1, b*30, 640), None, (b*30, b*30)
#         msa = self.linear1(trans_cls)  #(1,b*30,640)
#         msa = self.find_similar_round2(msa, sim_round2)   #(b*30, 1280)
#
#         out = self.linear2(msa)    #(b*30, 1280)
#         return out
















