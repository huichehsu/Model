# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d  # 导入 DeformConv2d
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from torch.utils.checkpoint import checkpoint
import thop
from torch import Tensor
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_
import torch_dct as dct
import cv2
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np
import warnings
from einops import rearrange
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from mmcv.ops.carafe import normal_init, xavier_init, carafe
except ImportError:

    def xavier_init(module: nn.Module,
                    gain: float = 1,
                    bias: float = 0,
                    distribution: str = 'normal') -> None:
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def carafe(x, normed_mask, kernel_size, group=1, up=1):
            b, c, h, w = x.shape
            _, m_c, m_h, m_w = normed_mask.shape
            # print('x', x.shape)
            # print('normed_mask', normed_mask.shape)
            # assert m_c == kernel_size ** 2 * up ** 2
            assert m_h == up * h
            assert m_w == up * w
            pad = kernel_size // 2
            # print(pad)
            pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
            # print("x", x.shape)
            # print("pad_x", pad_x.shape)
            unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
            # print("unfold_x", unfold_x.shape)
            # unfold_x = unfold_x.reshape(b, c, 1, kernel_size, kernel_size, h, w).repeat(1, 1, up ** 2, 1, 1, 1, 1)
            unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
            # print("reshape unfold_x", unfold_x.shape)
            unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
            # print("interpolate unfold_x", unfold_x.shape)
            # normed_mask = normed_mask.reshape(b, 1, up ** 2, kernel_size, kernel_size, h, w)
            unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
            # print("reshape unfold_x", unfold_x.shape)
            normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
            # print(normed_mask.shape)
            res = unfold_x * normed_mask
            # test
            # res[:, :, 0] = 1
            # res[:, :, 1] = 2
            # res[:, :, 2] = 3
            # res[:, :, 3] = 4
            res = res.sum(dim=2).reshape(b, c, m_h, m_w)
            # res = F.pixel_shuffle(res, up)
            # print(res.shape)
            # print(res)
            return res

    def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "Fusion",
    "Cross_attention",
    "RA",
    "FDConv"
    "SDP",
    "FEM",
    "AFS",
    "WFFM_Concat",
    "FreqFusion",
    "Ehance_suppress",
    "Mask",
    "SFS_Conv",
    "ScConv",
    "LDConv"
)

class SDP(nn.Module):
    def __init__(self, c1, c2, block_size):
        """
        Args:
            in_channels: 輸入特徵圖的通道數
            block_size: 用於劃分特徵塊的空間尺寸（假設輸入 H, W 均能被 block_size 整除）
        """
        super(SDP, self).__init__()
        self.in_channels = c1
        self.block_size = 4

        self.sigmoid = nn.Sigmoid()
        self.DWconv = nn.Conv2d(c2, c1, 1, 1, 0)
        
        # 定義 1x1 卷積，用於生成 Q、K、V
        self.q_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

    def show_feature(self, x, name="feature"):
        feature_map = x[0].detach().cpu().numpy()  
        # print("feature_map", feature_map.shape)
        feature_map = np.sum(feature_map, axis=0, keepdims=True)[0]
        # print("feature_map", feature_map.shape)
        norm_img = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
        norm_img = norm_img.astype(np.uint8)
        # cv2.imshow(name, norm_img)
        cv2.imwrite(f"feature_map/{name}.png", norm_img)
        # cv2.waitKey(0)
    # def show_feature(self, x, name="feature_heatmap"):
    #     # 1. 取出 feature map，假設 x shape = [1, C, H, W]，這裡選第 0 個 channel
    #     feature_map = x[0][0].detach().cpu().numpy()  # shape (H, W)

    #     # 2. 正規化到 0–255 並轉成 uint8
    #     norm = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
    #     norm = norm.astype(np.uint8)

    #     # 3. 套用 OpenCV 的 JET colormap
    #     heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    #     #    heatmap 現在是 BGR 三通道影像

    #     # （可選）若想疊加到灰階原圖上，可以這樣做：
    #     # gray_bgr = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    #     # overlay = cv2.addWeighted(gray_bgr, 0.5, heatmap, 0.5, 0)

    #     # 4. 顯示熱力圖
    #     cv2.imshow(name, heatmap)
    #     cv2.waitKey(0)
        # cv2.destroyWindow(name)        

    def forward(self, x):
        """
        Args:
            c: 下層特徵圖 (B, C, H, W)，來自 Ci
            p_upper: 上層特徵圖 (B, C, H_u, W_u)，來自 Pi+1，需要上採樣到與 c 相同尺寸
        Returns:
            加入了空間依賴關係後的特徵圖，形狀與 c 相同
        """
        c, p_upper = x[0], x[1]
        B, C, H, W = c.shape
        
        # 將上層特徵圖上採樣至與 c 相同的空間尺寸
        p_upper_upsampled = F.interpolate(p_upper, size=(H, W), mode='bilinear', align_corners=False)
        # self.show_feature(p_upper_upsampled, "Fd")
       
        
        # 分別通過 1x1 卷積生成 Q、K、V
        Q = self.q_conv(c)             # (B, C, H, W)
        K = self.k_conv(p_upper_upsampled)  # (B, C, H, W)
        V = self.v_conv(p_upper_upsampled)  # (B, C, H, W)

        
        # 保證 H 和 W 都能被 block_size 整除
        assert H % self.block_size == 0 and W % self.block_size == 0, "H 和 W 必須能被 block_size 整除"
        new_H = H // self.block_size
        new_W = W // self.block_size
        block_area = self.block_size * self.block_size
        
        # 將 Q、K、V 重塑成多個局部特徵塊，每個特徵塊大小為 (block_size x block_size)
        # 變換步驟：
        # 1. 先 reshape 為 (B, C, new_H, block_size, new_W, block_size)
        # 2. permute 使得 channel 移到最後，形狀變為 (B, new_H, new_W, block_size, block_size, C)
        # 3. 合併 new_H 和 new_W 成為 n 個特徵塊，每個塊包含 block_area 個像素點
        Q = Q.view(B, C, new_H, self.block_size, new_W, self.block_size).permute(0, 2, 4, 3, 5, 1).contiguous()
        Q = Q.view(B, new_H * new_W, block_area, C)  # (B, n, block_area, C)
        
        K = K.view(B, C, new_H, self.block_size, new_W, self.block_size).permute(0, 2, 4, 3, 5, 1).contiguous()
        K = K.view(B, new_H * new_W, block_area, C)  # (B, n, block_area, C)
        
        V = V.view(B, C, new_H, self.block_size, new_W, self.block_size).permute(0, 2, 4, 3, 5, 1).contiguous()
        V = V.view(B, new_H * new_W, block_area, C)  # (B, n, block_area, C)
        
        # 在每個特徵塊內計算像素級別的相似性
        scale = math.sqrt(C)
        # 相似性矩陣 shape: (B, n, block_area, block_area)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        
        # 對 V 進行加權，得到每個局部塊的輸出
        out = torch.matmul(attn, V)  # (B, n, block_area, C)
        
        # 將局部塊重新合併回原始的空間尺寸
        out = out.view(B, new_H, new_W, self.block_size, self.block_size, C)
        # 調整維度，恢復為 (B, C, H, W)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, C, H, W)
        
        # 將學習到的空間依賴特徵與原始下層特徵相加（殘差連接）
        out = p_upper_upsampled + out
        # self.show_feature(out, "align_Fd")
        # out[out < out.mean()*1.3] = 0
        
        

        gs = self.sigmoid(self.DWconv(out))
        ram = 1 - gs
        pss = c * ram 
        # self.show_feature(gs, "align_gs")
        # self.show_feature(ram, name="align_raw")
        # self.show_feature(pss, name="align_tdsa")

        p_upper_upsampled_conv = self.sigmoid(self.DWconv(p_upper_upsampled))
        new = 1 - p_upper_upsampled_conv
        # self.show_feature(new, name="ori_raw")
        # mean_value = new.mean()  # 計算每個通道的均值
        # new[new > new.mean() ] = 1
        result = c * new

        # self.show_feature(p_upper_upsampled_conv, "gs")
        # self.show_feature(new, name="raw")
        # self.show_feature(c, name="Fs")
        # self.show_feature(result, name='tdsa')



        return pss  #两个特征图相减



class AFS(nn.Module):
    def __init__(self, c1, c2, hidden_channels=64):
        super(AFS, self).__init__()
        # 預處理合併後的特徵
        add_in_channels = c1 + c2
        self.initial_conv = nn.Conv2d(add_in_channels, hidden_channels, kernel_size=3, padding=1)
        
        # 偏移預測網路 f_d
        self.deviation_net = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//2, 2, kernel_size=3, padding=1),  # 輸出偏移 (dx, dy)
            nn.Tanh()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()
        self.DWconv = nn.Conv2d(c2, c1, 1, 1, 0)


    def forward(self, x):
        # 合併特徵
        x_v, x_t = x[0], x[1]  # 假設 x 是一個包含兩個特徵圖的元組
        if x_v.shape[2:] != x_t.shape[2:]:
            x_t = F.interpolate(x_t, size=x_v.shape[2:], mode='nearest')

        x_cat = torch.cat([x_v, x_t], dim=1)
        x_d = self.initial_conv(x_cat)
        
        # 計算偏移量 Δp
        delta_p = self.deviation_net(x_d)

        # 網格生成，用於應用偏移（這裡用簡化的方式，實際應視應用設計）
        B, _, H, W = delta_p.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        grid = torch.stack((grid_x, grid_y), 2).to(x_v.device)  # shape: (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)

        # 加上偏移量，並使用 grid_sample 進行採樣
        grid = grid + delta_p.permute(0, 2, 3, 1)
        x_v_aligned = F.grid_sample(x_v, grid, align_corners=True)
        x_t_aligned = F.grid_sample(x_t, grid, align_corners=True)

        gs = self.sigmoid(self.DWconv(x_t_aligned))


        ram = 1 - gs
        pss = x_v_aligned * ram

        return pss


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
    
class KernelSpatialModulation_Global(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16, 
                 temp=1.0, kernel_temp=None, kernel_att_init='dyconv_as_extra', att_multi=2.0, ksm_only_kernel_att=False, att_grid=1, stride=1, spatial_freq_decompose=False,
                 act_type='sigmoid'):
        super(KernelSpatialModulation_Global, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        self.temperature = temp
        self.kernel_temp = kernel_temp
        
        self.ksm_only_kernel_att = ksm_only_kernel_att

        # self.temperature = nn.Parameter(torch.FloatTensor([temp]), requires_grad=True)
        self.kernel_att_init = kernel_att_init
        self.att_multi = att_multi
        # self.kn = nn.Parameter(torch.FloatTensor([kernel_num]), requires_grad=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.att_grid = att_grid
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        # self.bn = nn.Identity()
        # self.bn = nn.BatchNorm2d(attention_channel)
        self.bn = nn.GroupNorm(1, attention_channel)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = StarReLU()
        # self.dropout = nn.Dropout2d(p=0.1)
        # self.sp_att = SpatialGate(stride=stride, out_channels=1)

        # self.attup = AttUpsampler(inplane=in_planes, flow_make_k=1)

        self.spatial_freq_decompose = spatial_freq_decompose
        # self.channel_compress = ChannelPool()
        # self.channel_spatial = BasicConv(
        #     # 2, 1, 7, stride=1, padding=(7 - 1) // 2, relu=False
        #     2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        # )
        # self.filter_spatial = BasicConv(
        #     # 2, 1, 7, stride=stride, padding=(7 - 1) // 2, relu=False
        #     2, 1, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, relu=False
        # )
        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            if spatial_freq_decompose:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes * 2 if self.kernel_size > 1 else in_planes, 1, bias=True)
            else:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
            # self.channel_fc_bias = nn.Parameter(torch.zeros(1, in_planes, 1, 1), requires_grad=True)
            self.func_channel = self.get_channel_attention

        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            if spatial_freq_decompose:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes * 2, 1, stride=stride, bias=True)
            else:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
            # self.filter_fc_bias = nn.Parameter(torch.zeros(1, in_planes, 1, 1), requires_grad=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1 or self.ksm_only_kernel_att:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            # self.kernel_fc = nn.Conv2d(attention_channel, kernel_num * kernel_size * kernel_size, 1, bias=True)
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if hasattr(self, 'channel_spatial'):
            nn.init.normal_(self.channel_spatial.conv.weight, std=1e-6)
        if hasattr(self, 'filter_spatial'):
            nn.init.normal_(self.filter_spatial.conv.weight, std=1e-6)
            
        if hasattr(self, 'spatial_fc') and isinstance(self.spatial_fc, nn.Conv2d):
            # nn.init.constant_(self.spatial_fc.weight, 0)
            nn.init.normal_(self.spatial_fc.weight, std=1e-6)
            # self.spatial_fc.weight *= 1e-6
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            else:
                # nn.init.constant_(self.spatial_fc.weight, 0)
                # nn.init.constant_(self.spatial_fc.bias, 0)
                pass

        if hasattr(self, 'func_filter') and isinstance(self.func_filter, nn.Conv2d):
            # nn.init.constant_(self.func_filter.weight, 0)
            nn.init.normal_(self.func_filter.weight, std=1e-6)
            # self.func_filter.weight *= 1e-6
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            else:
                # nn.init.constant_(self.func_filter.weight, 0)
                # nn.init.constant_(self.func_filter.bias, 0)
                pass

        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            # nn.init.constant_(self.kernel_fc.weight, 0)
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
                # nn.init.constant_(self.kernel_fc.weight, 0)
                # nn.init.constant_(self.kernel_fc.bias, -10)
                # nn.init.constant_(self.kernel_fc.weight[0], 6)
                # nn.init.constant_(self.kernel_fc.weight[1:], -6)
            else:
                # nn.init.constant_(self.kernel_fc.weight, 0)
                # nn.init.constant_(self.kernel_fc.bias, 0)
                # nn.init.constant_(self.kernel_fc.bias, -10)
                # nn.init.constant_(self.kernel_fc.bias[0], 10)
                pass
            
        if hasattr(self, 'channel_fc') and isinstance(self.channel_fc, nn.Conv2d):
            # nn.init.constant_(self.channel_fc.weight, 0)
            nn.init.normal_(self.channel_fc.weight, std=1e-6)
            # nn.init.constant_(self.channel_fc.bias[1], 6)
            # nn.init.constant_(self.channel_fc.bias, 0)
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            else:
                # nn.init.constant_(self.channel_fc.weight, 0)
                # nn.init.constant_(self.channel_fc.bias, 0)
                pass
            

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        if self.act_type =='sigmoid':
            channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi # b, kn, cout, cin, k, k
        elif self.act_type =='tanh':
            channel_attention = 1 + torch.tanh_(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature) # b, kn, cout, cin, k, k
        else:
            raise NotImplementedError
        # channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi # b, kn, cout, cin, k, k
        # channel_attention = torch.sigmoid(self.channel_fc(x) / self.temperature) * self.att_multi # b, kn, cout, cin, k, k
        # channel_attention = self.channel_fc(x) # b, kn, cout, cin, k, k
        # channel_attention = torch.tanh_(self.channel_fc(x) / self.temperature) + 1 # b, kn, cout, cin, k, k
        return channel_attention

    def get_filter_attention(self, x):
        if self.act_type =='sigmoid':
            filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi # b, kn, cout, cin, k, k
        elif self.act_type =='tanh':
            filter_attention = 1 + torch.tanh_(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature) # b, kn, cout, cin, k, k
        else:
            raise NotImplementedError
        # filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi # b, kn, cout, cin, k, k
        # filter_attention = self.filter_fc(x) # b, kn, cout, cin, k, k
        # filter_attention = torch.tanh_(self.filter_fc(x) / self.temperature) + 1 # b, kn, cout, cin, k, k
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size) 
        if self.act_type =='sigmoid':
            spatial_attention = torch.sigmoid(spatial_attention / self.temperature) * self.att_multi
        elif self.act_type =='tanh':
            spatial_attention = 1 + torch.tanh_(spatial_attention / self.temperature)
        else:
            raise NotImplementedError
        return spatial_attention

    def get_kernel_attention(self, x):
        # kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, self.kernel_size, self.kernel_size)
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type =='softmax':
            kernel_attention = F.softmax(kernel_attention / self.kernel_temp, dim=1)
        elif self.act_type =='sigmoid':
            kernel_attention = torch.sigmoid(kernel_attention / self.kernel_temp) * 2 / kernel_attention.size(1)
        elif self.act_type =='tanh':
            kernel_attention = (1 + torch.tanh(kernel_attention / self.kernel_temp)) / kernel_attention.size(1)
        else:
            raise NotImplementedError
            
        # kernel_attention = kernel_attention / self.temperature
        # kernel_attention = kernel_attention / kernel_attention.abs().sum(dim=1, keepdims=True)
        return kernel_attention
    
    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
        
    def _forward(self, x):
        # comp_x = self.channel_compress(x)
        # csg = self.channel_spatial(comp_x).sigmoid_() * self.att_multi
        # csg = 1
        # fsg = self.filter_spatial(comp_x).sigmoid_() * self.att_multi
        # fsg = 1
        # x_h = x.mean(dim=-1, keepdims=True)
        # x_w = x.mean(dim=-2, keepdims=True)
        # x_h = self.relu(self.bn(self.fc(x_h)))
        # x_w = self.relu(self.bn(self.fc(x_w)))
        # avg_x = (self.avgpool(x_h) + self.avgpool(x_w)) * 0.5
        # avg_x = self.avgpool(self.relu(self.bn(self.fc(x))))
        avg_x = self.relu(self.bn(self.fc(x)))
        return self.func_channel(avg_x), self.func_filter(avg_x), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return self.attup.flow_warp(self.func_channel(x), grid), self.attup.flow_warp(self.func_filter(x), grid), self.func_spatial(avg_x), self.func_kernel(avg_x), sp_gate
        # return (self.func_channel(x_h) * self.func_channel(x_w)).sqrt(), (self.func_filter(x_h) * self.func_filter(x_w)).sqrt(), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return (self.func_channel(x_h) * self.func_channel(x_w)), (self.func_filter(x_h) * self.func_filter(x_w)), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return ((self.func_channel(x_h) + self.func_channel(x_w)) * csg).sigmoid_() * self.att_multi, ((self.func_filter(x_h) + self.func_filter(x_w)) * fsg).sigmoid_() * self.att_multi, self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return (self.func_channel(x_h) * self.func_channel(x_w) * csg), (self.func_filter(x_h) * self.func_filter(x_w) * fsg), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return (self.dropout(self.func_channel(x_h) * self.func_channel(x_w))), (self.dropout(self.func_filter(x_h) * self.func_filter(x_w))), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # k_att = F.relu(self.func_kernel(x) - 0.8 * self.func_kernel(x_inverse))
        # k_att = k_att / (k_att.sum(dim=1, keepdim=True) + 1e-8)
        # return self.func_channel(x), self.func_filter(x), self.func_spatial(x), k_att


class KernelSpatialModulation_Local(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super(KernelSpatialModulation_Local, self).__init__()
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        if channel is not None: k_size =  round((math.log2(channel) / 2) + 0.5) // 2 * 2 + 1
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        nn.init.constant_(self.conv.weight, 1e-6)
        self.use_global = use_global
        if self.use_global:
            self.complex_weight = nn.Parameter(torch.randn(1, self.channel // 2 + 1 , 2, dtype=torch.float32) * 1e-6)
            # self.norm = nn.GroupNorm(num_groups=32, num_channels=channel)
        self.norm = nn.LayerNorm(self.channel)
            # self.norm_std = nn.LayerNorm(self.channel)
            # trunc_normal_(self.complex_weight, std=.02)
            # self.sigmoid = nn.Sigmoid()
            # nn.init.constant(self.conv.weight.data) # nn.init.normal_(self.conv.weight, std=1e-6)
            # nn.init.zeros_(self.conv.weight)

    def forward(self, x, x_std=None):
        # feature descriptor on the global spatial information
        # y = self.avg_pool(x)
        # b,c,1, -> b,1,c, -> b, kn * out_n, c
        # x = torch.cat([x, x_std], dim=-2)
        x = x.squeeze(-1).transpose(-1, -2) # b,1,c,
        b, _, c = x.shape
        if self.use_global:
            x_rfft = torch.fft.rfft(x.float(), dim=-1) # b, 1 or 2, c // 2 +1
            # print(x_rfft.shape)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1) # b, 1, c // 2 +1
        x = self.norm(x)
            # x = torch.stack([self.norm(x[:, 0]), self.norm_std(x[:, 1])], dim=1)
        # b,1,c, -> b, kn * out_n, c
        att_logit = self.conv(x)
        # print(att_logit.shape)
        # print(att.shape)
        # Multi-scale information fusion
        # att = self.sigmoid(att) * 2
        att_logit = att_logit.reshape(x.size(0), self.kn, self.out_n, c) # b, kn, k1*k2, cin
        att_logit = att_logit.permute(0, 1, 3, 2) # b, kn, cin, k1*k2
        # print(att_logit.shape)
        return att_logit


class FrequencyBandModulation(nn.Module):
    def __init__(self, 
                in_channels,
                k_list=[2,4,8],
                lowfreq_att=False,
                fs_feat='feat',
                act='sigmoid',
                spatial='conv',
                spatial_group=1,
                spatial_kernel=3,
                init='zero',
                **kwargs,
                ):
        super().__init__()
        # k_list.sort()
        # print()
        self.k_list = k_list
        # self.freq_list = freq_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.in_channels = in_channels
        # self.residual = residual
        if spatial_group > 64: spatial_group=in_channels
        self.spatial_group = spatial_group
        self.lowfreq_att = lowfreq_att
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:  _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=self.spatial_group, 
                                            stride=1,
                                            kernel_size=spatial_kernel, 
                                            groups=self.spatial_group,
                                            padding=spatial_kernel//2, 
                                            bias=True)
                if init == 'zero':
                    nn.init.normal_(freq_weight_conv.weight, std=1e-6)
                    freq_weight_conv.bias.data.zero_()   
                else:
                    # raise NotImplementedError
                    pass
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError
        self.act = act

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'tanh':
            freq_weight = 1 + freq_weight.tanh()
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError
        return freq_weight

    def forward(self, x, att_feat=None):
        """
        att_feat:feat for gen att
        """
        if att_feat is None: att_feat = x
        x_list = []
        x = x.to(torch.float32)
        pre_x = x.clone()
        b, _, h, w = x.shape
        h, w = int(h), int(w)
        # x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))
        x_fft = torch.fft.rfft2(x, norm='ortho')

        for idx, freq in enumerate(self.k_list):
            mask = torch.zeros_like(x_fft[:, 0:1, :, :], device=x.device)
            _, freq_indices = get_fft2freq(d1=x.size(-2), d2=x.size(-1), use_rfft=True)
            # mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 1.0
            # print(freq_indices.shape)
            freq_indices = freq_indices.max(dim=-1, keepdims=False)[0]
            # print(freq_indices)
            mask[:,:, freq_indices < 0.5 / freq] = 1.0
            # print(mask.sum())
            # low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask), norm='ortho').real
            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), dim=(-2, -1), norm='ortho')
            try: 
                low_part = low_part.real
            except:
                pass
            high_part = pre_x - low_part
            pre_x = low_part
            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self.sp_act(freq_weight)
            # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        if self.lowfreq_att:
            freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
            freq_weight = self.sp_act(freq_weight)
            # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)
        x = sum(x_list)
        return x

def get_fft2freq(d1, d2, use_rfft=False):
    # Frequency components for rows and columns
    freq_h = torch.fft.fftfreq(d1)  # Frequency for the rows (d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)  # Frequency for the columns (d2)
    else:
        freq_w = torch.fft.fftfreq(d2)
    
    # Meshgrid to create a 2D grid of frequency coordinates
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w), dim=-1)
    # print(freq_hw)
    # print(freq_hw.shape)
    # Calculate the distance from the origin (0, 0) in the frequency space
    dist = torch.norm(freq_hw, dim=-1)
    # print(dist.shape)
    # Sort the distances and get the indices
    sorted_dist, indices = torch.sort(dist.view(-1))  # Flatten the distance tensor for sorting
    # print(sorted_dist.shape)
    
    # Get the corresponding coordinates for the sorted distances
    if use_rfft:
        d2 = d2 // 2 + 1
        # print(d2)
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)  # Convert flat indices to 2D coords
    # print(sorted_coords.shape)
    # # Print sorted distances and corresponding coordinates
    # for i in range(sorted_dist.shape[0]):
    #     print(f"Distance: {sorted_dist[i]:.4f}, Coordinates: ({sorted_coords[i, 0]}, {sorted_coords[i, 1]})")
    
    if False:
        # Plot the distance matrix as a grayscale image
        plt.imshow(dist.cpu().numpy(), cmap='gray', origin='lower')
        plt.colorbar()
        plt.title('Frequency Domain Distance')
        plt.show()
    return sorted_coords.permute(1, 0), freq_hw


class FDConv(nn.Conv2d):
    def __init__(self, 
                 *args, 
                 reduction=0.0625, 
                 kernel_num=4,
                 use_fdconv_if_c_gt=16, #if channel greater or equal to 16, e.g., 64, 128, 256, 512
                 use_fdconv_if_k_in=[1, 3], #if kernel_size in the list
                 use_fbm_if_k_in=[3], #if kernel_size in the list
                 kernel_temp=1.0,
                 temp=None,
                 att_multi=2.0,
                 param_ratio=1,
                 param_reduction=1.0,
                 ksm_only_kernel_att=False,
                 att_grid=1,
                 use_ksm_local=True,
                 ksm_local_act='sigmoid',
                 ksm_global_act='sigmoid',
                 spatial_freq_decompose=False,
                 convert_param=True,
                 linear_mode=False,
                 fbm_cfg={
                    'k_list':[2, 4, 8],
                    'lowfreq_att':False,
                    'fs_feat':'feat',
                    'act':'sigmoid',
                    'spatial':'conv',
                    'spatial_group':1,
                    'spatial_kernel':3,
                    'init':'zero',
                    'global_selection':False,
                 },
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.spatial_freq_decompose = spatial_freq_decompose
        self.use_fbm_if_k_in = use_fbm_if_k_in

        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act
        assert self.ksm_local_act in ['sigmoid', 'tanh']
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']

        ### Kernel num & Kernel temp setting
        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)
        if temp is None:
            temp = kernel_temp

        # print('*** kernel_num:', self.kernel_num)
        self.alpha = min(self.out_channels, self.in_channels) // 2 * self.kernel_num * self.param_ratio / param_reduction
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return
        self.KSM_Global = KernelSpatialModulation_Global(self.in_channels, self.out_channels, self.kernel_size[0], groups=self.groups, 
                                                        temp=temp,
                                                        kernel_temp=kernel_temp,
                                                        reduction=reduction, kernel_num=self.kernel_num * self.param_ratio, 
                                                        kernel_att_init=None, att_multi=att_multi, ksm_only_kernel_att=ksm_only_kernel_att, 
                                                        act_type=self.ksm_global_act,
                                                        att_grid=att_grid, stride=self.stride, spatial_freq_decompose=spatial_freq_decompose)
        
        if self.kernel_size[0] in use_fbm_if_k_in:
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)
            # self.FBM = OctaveFrequencyAttention(2 * self.in_channels // 16, **fbm_cfg)
            # self.channel_comp = ChannelPool(reduction=16)
            
        if self.use_ksm_local:
            self.KSM_Local = KernelSpatialModulation_Local(channel=self.in_channels, kernel_num=1, out_n=int(self.out_channels * self.kernel_size[0] * self.kernel_size[1]) )
        
        self.linear_mode = linear_mode
        self.convert2dftweight(convert_param)
            

    def convert2dftweight(self, convert_param):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True) # 2, d1 * k1 * (d2 * k2 // 2 + 1)
        # freq_indices = freq_indices.reshape(2, self.kernel_num, -1)
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)) # d1 * k1, d2 * k2 // 2 + 1
        if self.param_reduction < 1:
            freq_indices = freq_indices[:, torch.randperm(freq_indices.size(1), generator=torch.Generator().manual_seed(freq_indices.size(1)))] # 2, indices
            freq_indices = freq_indices[:, :int(freq_indices.size(1) * self.param_reduction)] # 2, indices
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            weight_rfft = weight_rfft.reshape(-1, 2)[None, ].repeat(self.param_ratio, 1, 1) / (min(self.out_channels, self.in_channels) // 2)
        else:
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(self.param_ratio, 1, 1, 1) / (min(self.out_channels, self.in_channels) // 2) #param_ratio, d1, d2, k*k, 2
        
        if convert_param:
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)
            del self.weight
        else:
            if self.linear_mode:
                self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=True)
        self.indices = []
        for i in range(self.param_ratio):
            self.indices.append(freq_indices.reshape(2, self.kernel_num, -1)) # paramratio, 2, kernel_num, d1 * k1 * (d2 * k2 // 2 + 1) // kernel_num

    def get_FDW(self, ):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)) # d1 * k1, d2 * k2 // 2 + 1
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(self.param_ratio, 1, 1, 1) / (min(self.out_channels, self.in_channels) // 2) #param_ratio, d1, d2, k*k, 2
        return weight_rfft
        
    def forward(self, x):
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return super().forward(x)
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)
        if self.use_ksm_local:
            # global_x_std = torch.std(x, dim=(-1, -2), keepdim=True)
            hr_att_logit = self.KSM_Local(global_x) # b, kn, cin, cout * ratio, k1*k2, 
            hr_att_logit = hr_att_logit.reshape(x.size(0), 1, self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])
            # hr_att_logit = hr_att_logit + self.hr_cin_bias[None, None, :, None, None, None] + self.hr_cout_bias[None, None, None, :, None, None] + self.hr_spatial_bias[None, None, None, None, :, :]
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2, 4, 5)
            if self.ksm_local_act == 'sigmoid':
                hr_att = hr_att_logit.sigmoid() * self.att_multi
            elif self.ksm_local_act == 'tanh':
                hr_att = 1 + hr_att_logit.tanh()
            else:
                raise NotImplementedError
        else:
            hr_att = 1
        b = x.size(0)
        batch_size, in_planes, height, width = x.size()
        DFT_map = torch.zeros((b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1] // 2 + 1, 2), device=x.device)
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)
        if hasattr(self, 'dft_weight'):
            dft_weight = self.dft_weight
        else:
            dft_weight = self.get_FDW()

        for i in range(self.param_ratio):
            indices = self.indices[i]
            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None]
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack([w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
            else:
                w = dft_weight[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha # 1, kernel_num, -1, 2
                # print(w.shape)
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack([w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)

        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(DFT_map), dim=(1, 2)).reshape(batch_size, 1, self.out_channels, self.kernel_size[0], self.in_channels, self.kernel_size[1])
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5)
        # print(spatial_attention, channel_attention, filter_attention)
        if hasattr(self, 'FBM'):
            x = self.FBM(x)
            # x = self.FBM(x, self.channel_comp(x))

        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] < (in_planes + self.out_channels) * height * width:
            # print(channel_attention.shape, filter_attention.shape, hr_att.shape)
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            # aggregate_weight = spatial_attention * channel_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            # print(aggregate_weight.abs().max())
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_attention, float): 
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1)) # * filter_attention.reshape(b, -1, 1, 1)
        else:
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            if not isinstance(channel_attention, float): 
                x = x * channel_attention.view(b, -1, 1, 1)
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups * batch_size)
            # if isinstance(filter_attention, torch.FloatTensor): 
            if isinstance(filter_attention, float): 
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1)) * filter_attention.view(b, -1, 1, 1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output

    def profile_module(
                self, input: Tensor, *args, **kwargs
            ):
            # TODO: to edit it
            b_sz, c, h, w = input.shape
            seq_len = h * w

            # FFT iFFT
            p_ff, m_ff = 0, 5 * b_sz * seq_len * int(math.log(seq_len)) * c
            # others
            # params = macs = sum([p.numel() for p in self.parameters()])
            params = macs = self.hidden_size * self.hidden_size_factor * self.hidden_size * 2 * 2 // self.num_blocks
            # // 2 min n become half after fft
            macs = macs * b_sz * seq_len

            # return input, params, macs
            return input, params, macs + m_ff



class Cross_attention(nn.Module):
    def __init__(self, c1, c2, block_size):
        """
        Args:
            in_channels: 輸入特徵圖的通道數
            block_size: 用於劃分特徵塊的空間尺寸（假設輸入 H, W 均能被 block_size 整除）
        """
        super(Cross_attention, self).__init__()
        self.in_channels = c1
        self.block_size = block_size

        self.sigmoid = nn.Sigmoid()
        self.DWconv = nn.Conv2d(c2, c1, 1, 1, 0)
        
        # 定義 1x1 卷積，用於生成 Q、K、V
        self.q_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        
    def forward(self, x):
        
        c, d = x[0], x[1]
        B, C, H, W = c.shape
               
        # 分別通過 1x1 卷積生成 Q、K、V
        Q = self.q_conv(d)             # (B, C, H, W)
        K = self.k_conv(c)  # (B, C, H, W)
        V = self.v_conv(c)  # (B, C, H, W)

        
        # 保證 H 和 W 都能被 block_size 整除
        assert H % self.block_size == 0 and W % self.block_size == 0, "H 和 W 必須能被 block_size 整除"
        new_H = H // self.block_size
        new_W = W // self.block_size
        block_area = self.block_size * self.block_size
        
        # 將 Q、K、V 重塑成多個局部特徵塊，每個特徵塊大小為 (block_size x block_size)
        # 變換步驟：
        # 1. 先 reshape 為 (B, C, new_H, block_size, new_W, block_size)
        # 2. permute 使得 channel 移到最後，形狀變為 (B, new_H, new_W, block_size, block_size, C)
        # 3. 合併 new_H 和 new_W 成為 n 個特徵塊，每個塊包含 block_area 個像素點
        Q = Q.view(B, C, new_H, self.block_size, new_W, self.block_size).permute(0, 2, 4, 3, 5, 1).contiguous()
        Q = Q.view(B, new_H * new_W, block_area, C)  # (B, n, block_area, C)
        
        K = K.view(B, C, new_H, self.block_size, new_W, self.block_size).permute(0, 2, 4, 3, 5, 1).contiguous()
        K = K.view(B, new_H * new_W, block_area, C)  # (B, n, block_area, C)
        
        V = V.view(B, C, new_H, self.block_size, new_W, self.block_size).permute(0, 2, 4, 3, 5, 1).contiguous()
        V = V.view(B, new_H * new_W, block_area, C)  # (B, n, block_area, C)
        
        # 在每個特徵塊內計算像素級別的相似性
        scale = math.sqrt(C)
        # 相似性矩陣 shape: (B, n, block_area, block_area)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        
        # 對 V 進行加權，得到每個局部塊的輸出
        out = torch.matmul(attn, V)  # (B, n, block_area, C)
        
        # 將局部塊重新合併回原始的空間尺寸
        out = out.view(B, new_H, new_W, self.block_size, self.block_size, C)
        # 調整維度，恢復為 (B, C, H, W)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, C, H, W)
        
        # 將學習到的空間依賴特徵與原始下層特徵相加（殘差連接）
        out = c + out

        return out  #两个特征图相减

# class RA(nn.Module):  #相邻消除模块
#     def __init__(self, c1, c2, upsize=2): #c1小 c2大
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=upsize, mode='nearest')
#         self.sigmoid = nn.Sigmoid()
#         self.DWconv = nn.Conv2d(c2, c1, 1, 1, 0)
#         # self.Dsample = nn.Conv2d(c1, c1, upsize, upsize)
#     def forward(self, x):
#         ps, ps1 = x[0], x[1]

#         # gs = self.sigmoid(self.DWconv(self.upsample(ps1)))
#         if ps.shape[2:] != ps1.shape[2:]:
#             ps1 = F.interpolate(ps1, size=ps.shape[2:], mode='nearest')     
#         gs = self.sigmoid(self.DWconv(ps1))


#         ram = 1 - gs
#         pss = ps * ram

#         return pss  #两个特征图相减


class Xc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Xc, self).__init__()

        self.Xcenter = Conv(in_channels, in_channels, k=1, g=in_channels)
        self.Xcenter_relu = nn.ReLU(inplace=True)

        self.Xsurround = Conv(in_channels, in_channels, k=3, p=1, g=in_channels)
        self.Xsurround_relu = nn.ReLU(inplace=True)



    def forward(self, input):
        xcenter = self.Xcenter_relu(self.Xcenter(input))
        xsurround = self.Xsurround_relu(self.Xsurround(input))
        

        x = xcenter - xsurround


        return x


class Yc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Yc, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dilated_conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)   
        self.relu = nn.ReLU(inplace=True)



    def forward(self, input):
        conv3x3 = self.relu(self.conv3x3(input))
        dilated_conv3x3 = self.relu(self.dilated_conv3x3(input))
        

        y =  dilated_conv3x3 + conv3x3

        return y

class W(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(W, self).__init__()
        
        self.h = Conv(inchannel, inchannel, k=(1, 3), p=(0, 1), g=inchannel)
        self.v = Conv(inchannel, inchannel, k=(3, 1), p=(1, 0), g=inchannel)
        
        
        self.relu = nn.ReLU()

    def forward(self, x):

        h = self.relu(self.h(x))
        

        v = self.relu(self.v(h))
        

        return v

class XYWA_f(nn.Module):
    def __init__(self, c1, ratio=16): # ratio 可以調整，例如 8 或 16
        super(XYWA_f, self).__init__()
        self.dilc21 = Xc(c1, c1)
        self.dilc22 = Conv(c1, 1, k=1)

        # self.conv1 = nn.Conv2d(c1, c1, 1, 1, 0) # 根據BRSTD論文圖5，XYWA的輸入直接來自backbone或前一層

        self.relu = nn.ReLU(inplace=True) # inplace=True 節省內存

        # Channel Attention Parts
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 共享的FC層，或者為avg和max分別設計FC再融合
        # 這裡使用共享FC的簡化版本，先拼接
        self.fc1 = nn.Conv2d(c1, c1 // ratio, 1, bias=False) # 輸入通道數變為 c1*2
        self.fc2 = nn.Conv2d(c1 // ratio, c1, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # 注意這裡的輸入 x，而不是 x1
        # Spatial Attention (Ws)
        # 根據BRSTD Fig.5，X,Y,W通路處理輸入x，然後各自通過1x1卷積得到單通道圖，再相加
        # 您的 Xc, Yc, W 類別的 out_channels 參數似乎沒有在內部使用，它們都輸出與輸入相同的通道數
        # 因此 dilc12, dilc22, dilc32 應該是 Conv(c1, 1, k=1)

        s2_feat = self.dilc21(x)
        spatial_attention_map = self.dilc22(s2_feat)


        # Channel Attention (Wc)
        avg_p = self.avg_pool(x)
        channel_attention_vector = self.fc2(self.relu(self.fc1(avg_p))) # Wc, [B, c1, 1, 1]

        combined_attention = self.sigmoid(spatial_attention_map * channel_attention_vector)

        # Apply attention: F_out = F_in * W_A + F_in
        out = combined_attention * x + x
        return out
# class GlobalContextChannelAttention(nn.Module):
#     def __init__(self, c1, ratio=256):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(c1, c1 // ratio, 1, bias=True)
#         self.fc2 = nn.Conv2d(c1 // ratio, c1, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#         self.large_conv_11 = Conv(c1, c1, k=5, p=2)
#         self.large_conv_12 = Conv(c1, 1, k=5, p=2)


#         self.h = Conv(c1, c1, k=(1, 3), p=(0, 1))
#         self.v = Conv(c1, 1, k=(3, 1), p=(1, 0))

#         self.small_conv = Conv(c1, 1, k=1)

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         """
#         Args:
#             x_feat_to_adjust (Tensor): 需要被調整的特徵圖 (例如您計算的 feat)
#             x_global_context_source (Tensor): 用於提取全局上下文的源特徵圖 (例如 ps1)
#         """
#         large_feat = self.relu(self.large_conv_12(self.relu(self.large_conv_11(x))))
#         small_feat_1 = self.relu(self.small_conv(x))
#         small_feat_2 = self.relu(self.v(self.relu(self.h(x))))

#         mix_feat = small_feat_1 + small_feat_2 -large_feat

#         # c = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
#         # sc = self.sigmoid(mix_feat * c)

#         # out = x * sc + x
#         return mix_feat


class RA(nn.Module):  #相邻消除模块
    def __init__(self, c1, c2, upsize=2): #c1小 c2大
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=upsize, mode='nearest')
        self.sigmoid = nn.Sigmoid()
        self.DWconv = nn.Conv2d(c2, c1, 1, 1, 0)
        # self.Dsample = nn.Conv2d(c1, c1, upsize, upsize)
    
    def show_feature(self, x, name='feature_map'):
        # print("x", x.shape)
        feature_map = x[0].detach().cpu().numpy()  
        feature_map = np.sum(feature_map, axis=0, keepdims=True)[0]
        norm_img = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
        norm_img = norm_img.astype(np.uint8)
        # cv2.imshow(name, norm_img)
        cv2.imwrite(f"feature_map/{name}.png", norm_img)


    def forward(self, x):
        ps, ps1 = x[0], x[1]
        B, C, H, W = ps.shape
        
        gs = self.sigmoid(self.DWconv(F.interpolate(ps1, size=(H, W), mode='nearest')))
        ram = 1 - gs

        pss = ps * ram
        self.show_feature(ps, name='shollow')
        self.show_feature(ps1, name='deep')
        self.show_feature(gs, name='upsample')
        self.show_feature(ram, name='inverse')
        self.show_feature(pss, name='tdsa')

        return pss  #两个特征图相减


# class SEBlock(nn.Module):
#     def __init__(self, c1, ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1   = nn.Linear(c1, c1 // ratio)
#         self.relu  = nn.ReLU(inplace=True)
#         self.fc2   = nn.Linear(c1 // ratio, c1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         out = self.avg_pool(x).view(x.size(0), -1)
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)
#         return x * out

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class GlobalContextChannelAttention(nn.Module): # 简化后的 GCCA
#     def __init__(self, c1, ratio=16):
#         super().__init__()
#         self.se = SEBlock(c1, ratio)
#         self.conv1 = Conv(c1, c1 // 2, k=3, p=1) # 减少输出通道
#         self.conv2 = Conv(c1 // 2, 1, k=1, act=False) # 1x1 卷积，简化操作
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         se_out = self.se(x)
#         feat = self.conv1(x)
#         gate = self.sigmoid(self.conv2(feat))
#         out = x * gate + x # 简化门控
#         return out

# class RA(nn.Module):  # 相邻消除模块
#     def __init__(self, c1, c2, upsize=2):  # c1 小 c2 大
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=upsize, mode='bilinear', align_corners=False)
#         self.conv_ps1 = Conv(c2, c1, k=1)  # 减少 ps1 的通道数
#         self.deform_conv = DeformConv2d(c1, c1, kernel_size=3, padding=1) # 可变形卷积
#         self.spatial_attention = SpatialAttention(kernel_size=7) # 空间注意力
#         self.channel_attention = SEBlock(c1, ratio=16) # 通道注意力
#         self.gate_conv = Conv(c1, 1, k=3, p=1, act=False) # 生成门控信号
#         self.sigmoid = nn.Sigmoid()
#         self.weight_ps = nn.Parameter(torch.ones(1)) # 可学习的权重
#         self.weight_pss = nn.Parameter(torch.ones(1))

#     def forward(self, x):
#         ps, ps1 = x[0], x[1]  # ps: 浅层, ps1: 深层

#         # ps1_reduced = self.conv_ps1(ps1) # 首先减少通道数
#         # ps1_deform = self.deform_conv(ps1_reduced, ) # 可变形卷积

#         # 空间注意力：提取大型物体特征
#         ps1_spatial = self.spatial_attention(ps1)
#         ps1_attended = ps1 * ps1_spatial

#         # 通道注意力：选择重要的大型物体特征通道
#         ps1_channel = self.channel_attention(ps1_attended)

#         if ps.shape[2:] != ps1_channel.shape[2:]:
#             ps1_resized = F.interpolate(ps1_channel, size=ps.shape[2:], mode='bilinear', align_corners=False)
#         else:
#             ps1_resized = ps1_channel

#         gate = self.sigmoid(self.gate_conv(ps1_resized))
#         ram = 1 - gate
#         pss = ps * ram
#         out = (self.weight_pss * pss) + (self.weight_ps * ps)
#         return out




class Fusion(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        x1, x2 = x[0], x[1]
        x = x1 + x2
        return x

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# class FEM(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
#         super(FEM, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce
#         self.branch0 = nn.Sequential(
#             BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )

#         self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)

#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.relu(out)

#         return out


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/ghostnet
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (tuple): Tuple containing:
                - Equivalent kernel (torch.Tensor)
                - Equivalent bias (torch.Tensor)
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            (tuple): Tuple containing:
                - Fused kernel (torch.Tensor)
                - Fused bias (torch.Tensor)
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Select and return a particular index from input.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        # 定义查询、键和值的线性变换
        query_channels = max(in_channels // 8, 1)
        self.q = nn.Conv2d(in_channels, query_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, query_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.q(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, query_channels, HW]
        proj_key = self.k(x).view(B, -1, H * W)  # [B, query_channels, HW]
        energy = torch.bmm(proj_query, proj_key)         # [B, HW, HW]
        attention = F.softmax(energy, dim=-1)
        proj_value = self.v(x).view(B, -1, H * W) # [B, C, HW]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)
        out = out + x

        return out

# class WFFM(nn.Module):
#     def __init__(self, c1, c2): #c1小 c2大
#         super().__init__()
#         self.d = 1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.relu = nn.ReLU(inplace=True)
#         # 1x1 conv
#         self.conv1 = nn.Conv1d(c2 * 2, c2, kernel_size=1, stride=1)
#         self.sigmoid = nn.Sigmoid()
#         # self.self_attention = Attention(in_channels=c2) 
#     def forward(self, x):
#         concat = torch.cat(x, self.d)

#         avg_feat = self.relu(self.avg_pool(concat))
#         max_feat = self.relu(self.max_pool(concat))

#         avg_sum = avg_feat.sum(dim=[2, 3]) 
#         max_sum = max_feat.sum(dim=[2, 3])

#         cat_feat = torch.cat([avg_sum, max_sum], dim=1)
#         cat_feat = cat_feat.unsqueeze(-1)  # (B, C//4, 1)


#         chn_score = self.sigmoid(self.conv1(cat_feat)).unsqueeze(-1)
#         out = torch.mul(concat, chn_score) + concat

#         # out = self.self_attention(out)+ out

#         return out

class FEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.h = Conv(in_channels, in_channels, k=(1, 3), p=(0, 1), g=in_channels)
        self.v = Conv(in_channels, in_channels, k=(3, 1), p=(1, 0), g=in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.dilated_conv = Conv(in_channels, in_channels, k=5, p=4, d=2, g=in_channels)

        self.point_wise = Conv(in_channels, out_channels, k=1)

    def forward(self, x):
        h = self.relu(self.h(x))
        v = self.relu(self.v(h))

        x_dilated = self.relu(self.dilated_conv(x))

        x_point_wise = self.relu(self.point_wise(x))

        result = v + x_dilated + x_point_wise

        return result


class WFFM_Concat(nn.Module):
    def __init__(self, Channel_all, dimension=1):
        super(WFFM_Concat, self).__init__()
        self.d = dimension
        self.Channel_all = Channel_all
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型 parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()

        w = self.w[:(C1 + C2)] # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x = [x1, x2]
        return torch.cat(x, self.d)




# class FEM(nn.Module):
#     def __init__(self, c1, c2, stride=1, scale=0.1, map_reduce=8): #c1小 c2大
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(0.1))  # 可學習的高通濾波參數   

#         self.scale = scale
#         out_planes = c2
#         in_planes = c1
#         self.sigmoid = nn.Sigmoid()
#         inter_planes = in_planes // map_reduce
#         self.branch0 = nn.Sequential(
#             BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )

#         self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)


#     def dct_2d(self, x):
#         # 先对最后第二个维度做 DCT-II，再对最后一个维度做 DCT-II
#         x_dct = dct.dct_2d(x.float(), norm='ortho')
#         return x_dct

#     def idct_2d(self, x):
#         # 逆过程：先对最后第二个维度做 IDCT，再对最后一个维度做 IDCT
#         x_idct = dct.idct_2d(x, norm='ortho')
#         return x_idct


#     def high_pass_filter(self, dct_coeffs, h, w):
#         """ 構建高通濾波遮罩 """
#         alpha = self.alpha
#         mask = torch.ones((h, w), device=dct_coeffs.device)
#         mask[:int(alpha*h), :int(alpha*w)] = 0
#         return mask
#     def forward(self, x):
#         b, c, h, w = x.shape

#         x_dct = self.dct_2d(x)  # 對輸入進行 DCT-II 變換
#         mask = self.high_pass_filter(x_dct, h, w)
#         x_dct = x_dct * mask
#         x_hpf = self.idct_2d(x_dct)
#         x_hpf = torch.abs(x_hpf)
#         x_enhance = torch.mul(x, x_hpf)

#         x0 = self.branch0(x_enhance)
#         x1 = self.branch1(x_enhance)
#         x2 = self.branch2(x_enhance)

#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         short = self.shortcut(x_enhance)
#         out = out * self.scale + short
#         out = self.relu(out)


#         out = self.sigmoid(out) * x + x


#         return out


        
# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU(inplace=True) if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M：窗口的行数
    - N：窗口的列数

    返回：
    - 二维Hamming窗
    """
    # 生成水平和垂直方向上的Hamming窗
    # hamming_x = np.blackman(M)
    # hamming_x = np.kaiser(M)
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # 通过外积生成二维Hamming窗
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d

class FreqFusion(nn.Module):
    def __init__(self,
                hr_channels,
                lr_channels,
                scale_factor=1,
                lowpass_kernel=5,
                highpass_kernel=3,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64,        
                align_corners=False,
                upsample_mode='nearest',
                feature_resample=False, # use offset generator or not
                feature_resample_group=4,
                comp_feat_upsample=True, # use ALPF & AHPF for init upsampling
                use_high_pass=True,
                use_low_pass=True,
                hr_residual=True,
                semi_conv=True,
                hamming_window=True, # for regularization, do not matter really
                feature_resample_norm=True,
                **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels,1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels,1)
        self.content_encoder = nn.Conv2d( # ALPF generator
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample
        if self.feature_resample:
            self.dysampler = LocalSimGuidedSampler(in_channels=compressed_channels, scale=2, style='lp', groups=feature_resample_group, use_direct_scale=True, kernel_size=encoder_kernel, norm=feature_resample_norm)
        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d( # AHPF generator
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        self.hamming_window = hamming_window
        lowpass_pad=0
        highpass_pad=0
        if self.hamming_window:
            self.register_buffer('hamming_lowpass', torch.FloatTensor(hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
            self.register_buffer('hamming_highpass', torch.FloatTensor(hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel**2)) # group
        # print("mask_channel", mask_channel)
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        # print("mask", mask.shape)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # print("mask", mask.shape)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        # print(hamming)
        # print(mask.shape)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask =  mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        # print("mask", mask.shape)
        return mask

    def forward(self, x): # use check_point to save GPU memory
        hr_feat, lr_feat = x
        return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)

        mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat) #从hr_feat得到初始高通滤波特征
        mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel, hamming=self.hamming_highpass) #kernel归一化得到初始高通滤波
        compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat, mask_hr_init, self.highpass_kernel, self.up_group, 1) #利用初始高通滤波对压缩hr_feat的高频增强 （x-x的低通结果=x的高通结果）


        mask_lr_hr_feat = self.content_encoder(compressed_hr_feat) #从hr_feat得到初始低通滤波特征
        mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, hamming=self.hamming_lowpass) #kernel归一化得到初始低通滤波
        
        mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat) #从lr_feat得到另一部分初始低通滤波特征
        mask_lr_lr_feat = F.interpolate( #利用初始低通滤波对另一部分初始低通滤波特征上采样
            carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
        mask_lr = mask_lr_hr_feat + mask_lr_lr_feat #将两部分初始低通滤波特征合在一起  ！！！

        mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass) #得到初步融合的初始低通滤波
        mask_hr_lr_feat = F.interpolate( #使用初始低通滤波对lr_feat处理，分辨率得到提高
            carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
        mask_hr = mask_hr_hr_feat + mask_hr_lr_feat # 将两部分初始高通滤波特征合在一起 ！！！

        # print("mask_hr", mask_hr.shape)
        # print("mask_lr", mask_lr.shape)

        
        # mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
        # lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)

        mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
        hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
        hr_feat = hr_feat_hf + hr_feat

        lr_feat = carafe(lr_feat, mask_hr, self.highpass_kernel, self.up_group, 2)

        concat = torch.cat([lr_feat, hr_feat], 1)
                
        return  concat



class LocalSimGuidedSampler(nn.Module):
    """
    offset generator in FreqFusion
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, use_direct_scale=True, kernel_size=1, local_window=3, sim_type='cos', norm=True, direction_feat='sim_concat'):
        super().__init__()
        assert scale==2
        assert style=='lp'

        self.scale = scale
        self.style = style
        self.groups = groups
        self.local_window = local_window
        self.sim_type = sim_type
        self.direction_feat = direction_feat

        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2
        if self.direction_feat == 'sim':
            self.offset = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        elif self.direction_feat == 'sim_concat':
            self.offset = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        else: raise NotImplementedError
        normal_init(self.offset, std=0.001)
        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            elif self.direction_feat == 'sim_concat':
                self.direct_scale = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            else: raise NotImplementedError
            constant_init(self.direct_scale, val=0.)

        out_channels = 2 * groups
        if self.direction_feat == 'sim':
            self.hr_offset = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        elif self.direction_feat == 'sim_concat':
            self.hr_offset = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        else: raise NotImplementedError
        normal_init(self.hr_offset, std=0.001)
        
        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.hr_direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            elif self.direction_feat == 'sim_concat':
                self.hr_direct_scale = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            else: raise NotImplementedError
            constant_init(self.hr_direct_scale, val=0.)

        self.norm = norm
        if self.norm:
            self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
            self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)
        else:
            self.norm_hr = nn.Identity()
            self.norm_lr = nn.Identity()
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)
    
    def sample(self, x, offset, scale=None):
        if scale is None: scale = self.scale
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(
            B, 2, -1, scale * H, scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, scale * H, scale * W)
    
    def forward(self, hr_x, lr_x, feat2sample):
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)

        if self.direction_feat == 'sim':
            hr_sim = compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')
            lr_sim = compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')
        elif self.direction_feat == 'sim_concat':
            hr_sim = torch.cat([hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            lr_sim = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            hr_x, lr_x = hr_sim, lr_sim
        # offset = self.get_offset(hr_x, lr_x)
        offset = self.get_offset_lp(hr_x, lr_x, hr_sim, lr_sim)
        return self.sample(feat2sample, offset)
    
    # def get_offset_lp(self, hr_x, lr_x):
    def get_offset_lp(self, hr_x, lr_x, hr_sim, lr_sim):
        if hasattr(self, 'direct_scale'):
            # offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * (self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x), self.scale)).sigmoid() + self.init_pos
            offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x), self.scale)).sigmoid() + self.init_pos
            # offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (self.direct_scale(lr_sim) + F.pixel_unshuffle(self.hr_direct_scale(hr_sim), self.scale)).sigmoid() + self.init_pos
        else:
            offset =  (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def get_offset(self, hr_x, lr_x):
        if self.style == 'pl':
            raise NotImplementedError
        return self.get_offset_lp(hr_x, lr_x)
    

def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    """
    计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

    参数：
    - input_tensor: 输入张量，形状为[B, C, H, W]
    - k: 范围大小，表示周围KxK范围内的点

    返回：
    - 输出张量，形状为[B, KxK-1, H, W]
    """
    B, C, H, W = input_tensor.shape
    # 使用零填充来处理边界情况
    # padded_input = F.pad(input_tensor, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

    # 展平输入张量中每个点及其周围KxK范围内的点
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation) # B, CxKxK, HW
    # print(unfold_tensor.shape)
    unfold_tensor = unfold_tensor.reshape(B, C, k**2, H, W)

    # 计算余弦相似度
    if sim == 'cos':
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    # 移除中心点的余弦相似度，得到[KxK-1]的结果
    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

    # 将结果重塑回[B, KxK-1, H, W]的形状
    similarity = similarity.view(B, k * k - 1, H, W)
    return similarity


class Ehance_suppress(nn.Module): 
    def __init__(self, c1, gate):
        super().__init__()
        in_channels = c1
        upsize = 2
        self.gate = gate
        self.relu = nn.ReLU(inplace=True)


        self.conv_7x7 = Conv(in_channels, in_channels, k=7, p=3, g=in_channels)
        # self.conv_3x3 = Conv(in_channels, in_channels, k=3, p=1, g=in_channels)

        ratio = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c1, c1 // ratio, 1, bias=True)
        self.fc2 = nn.Conv2d(c1 // ratio, c1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()


    def show_feature(self, x, name='feature_map'):
        # print("x", x.shape)
        feature_map = x[0].detach().cpu().numpy()  
        feature_map = np.sum(feature_map, axis=0, keepdims=True)[0]
        norm_img = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
        norm_img = norm_img.astype(np.uint8)
        cv2.imshow(name, norm_img)
        cv2.imwrite(f"feature_map/{name}.png", norm_img)
        cv2.waitKey(0)
    
    def forward(self, x):
        x_large = self.relu(self.conv_7x7(x))
        # self.show_feature(x, "ori_feature")
        # x_small = self.relu(self.conv_3x3(x))

        # weight_small = self.sigmoid(x_small)
        weight_large = self.sigmoid(x_large)
        weight_x = self.sigmoid(x)
        if self.gate:
            # self.show_feature(x, "x_tiny")
            mask = weight_x - weight_large
            # self.show_feature(weight_x, "weight_x")
            # self.show_feature(weight_large, "weight_large")

            # self.show_feature(mask, "mask_tiny")

            x_enhance_combine = x*mask
            # self.show_feature(x_enhance_combine, "x_enhance_combine_tiny")

        else:
            # self.show_feature(x, "x_big")
            mask = weight_x + weight_large
            # self.show_feature(mask, "mask_big")

            x_enhance_combine = x*mask
            # self.show_feature(x_enhance_combine, "x_enhance_combine_big")

        c = self.fc2(self.relu(self.fc1(self.avg_pool(x_enhance_combine))))
        # print("c", c.shape)

        combined_output = x_enhance_combine * c
        # self.show_feature(combined_output, "output")

        return combined_output  

class Mask(nn.Module): 
    def __init__(self, c1):
        super().__init__()
        in_channels = c1
        upsize = 2
        # self.gate = gate
        self.relu = nn.ReLU(inplace=True)


        ratio = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c1, c1 // ratio, 1, bias=True)
        self.fc2 = nn.Conv2d(c1 // ratio, c1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.upsample = nn.Upsample(scale_factor=upsize, mode='bilinear', align_corners=True)

    def show_feature(self, x, name='feature_map'):
        # print("x", x.shape)
        feature_map = x[0].detach().cpu().numpy()  
        feature_map = np.sum(feature_map, axis=0, keepdims=True)[0]
        norm_img = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
        norm_img = norm_img.astype(np.uint8)
        # cv2.imshow(name, norm_img)
        cv2.imwrite(f"feature_map/{name}.png", norm_img)
        # cv2.waitKey(0)
    
    def forward(self, x):
        x1, x2 = x # x1是小物體特徵，x2是大物體特徵

        if x1.shape[2] != x2.shape[2]:
            self.show_feature(x1, "x1_tiny")
            self.show_feature(x2, "x2_big")
            x2_upsample = self.upsample(x2)
            self.show_feature(x2_upsample, "x2_upsample")

            feat_inverse_mean = torch.mean(x2_upsample, dim=1, keepdim=True)
            m_F_bar = torch.mean(feat_inverse_mean)
            v_F_bar = torch.std(feat_inverse_mean)

            threshold = m_F_bar + v_F_bar

            mask = (feat_inverse_mean >= threshold).float()
            self.show_feature(mask, "mask")
            x_enhance_combine = x1 - mask*x1    
            self.show_feature(x_enhance_combine, "x_enhance_combine_tiny")
        
        else:
            x_enhance_combine = x2
            # self.show_feature(x_enhance_combine, "x_enhance_combine_big")

        c = self.fc2(self.relu(self.fc1(self.avg_pool(x_enhance_combine))))
        # print("c", c.shape)

        combined_output = x_enhance_combine * c
        # self.show_feature(combined_output, "output")

        return combined_output  



class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        return self.act(self.pool(x))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1,))
        return self.act(x)


class FractionalGaborFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super(FractionalGaborFilter, self).__init__()

        self.real_weights = nn.ParameterList()
        self.imag_weights = nn.ParameterList()

        for angle in angles:
            for scale in scales:
                # real_weight, imag_weight = self.generate_fractional_gabor(in_channels, out_channels, kernel_size, order, angle, scale)
                real_weight = self.generate_fractional_gabor(in_channels, out_channels, kernel_size, order, angle, scale)
                self.real_weights.append(nn.Parameter(real_weight))
                # self.imag_weights.append(nn.Parameter(imag_weight))

    def generate_fractional_gabor(
        self, in_channels, out_channels, size, order, angle, scale):

        x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
        x_theta = x * np.cos(angle) + y * np.sin(angle)
        y_theta = -x * np.sin(angle) + y * np.cos(angle)

        real_part = np.exp(-((x_theta**2 + (y_theta / scale) ** 2) ** order)) * np.cos(2 * np.pi * x_theta / scale)
        # imag_part = np.exp(-((x_theta ** 2 + (y_theta / scale) ** 2) ** order)) * np.sin(2 * np.pi * x_theta / scale)

        # Reshape to match the specified out_channels and size
        real_weight = torch.tensor(real_part, dtype=torch.float32).view(1, 1, size[0], size[1])
        # imag_weight = torch.tensor(imag_part, dtype=torch.float32).view(1, 1, size[0], size[1])

        # Repeat along the out_channels dimension
        real_weight = real_weight.repeat(out_channels, 1, 1, 1)
        # imag_weight = imag_weight.repeat(out_channels, 1, 1, 1)

        return real_weight  # , imag_weight

    def forward(self, x):
        real_weights = [weight for weight in self.real_weights]
        # imag_weights = [weight for weight in self.imag_weights]

        real_result = sum(weight * x for weight in real_weights)
        # imag_result = sum(weight * x for weight in imag_weights)

        return real_result  # - imag_result


class GaborSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super(GaborSingle, self).__init__()
        self.gabor = FractionalGaborFilter(in_channels, out_channels, kernel_size, order, angles, scales)
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.gabor(self.t)
        out = F.conv2d(x, out, stride=1, padding=(out.shape[-2] - 1) // 2)
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)  # Padding on the left and top
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class GaborFPU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        order=0.25,
        angles=[0, 45, 90, 135],
        scales=[1, 2, 3, 4],
    ):
        super(GaborFPU, self).__init__()
        self.gabor = GaborSingle(in_channels // 4, out_channels // 4, (3, 3), order, angles, scales)
        self.fc = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat(
            [self.gabor(x1), self.gabor(x2), self.gabor(x3), self.gabor(x4)], dim=1
        )
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class FrFTFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super(FrFTFilter, self).__init__()

        self.register_buffer(
            "weight",
            self.generate_FrFT_filter(in_channels, out_channels, kernel_size, f, order),
        )

    def generate_FrFT_filter(self, in_channels, out_channels, kernel, f, p):
        N = out_channels
        d_x = kernel[0]
        d_y = kernel[1]
        x = np.linspace(1, d_x, d_x)
        y = np.linspace(1, d_y, d_y)
        [X, Y] = np.meshgrid(x, y)

        real_FrFT_filterX = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filterY = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filter = np.zeros([d_x, d_y, out_channels])
        for i in range(N):
            real_FrFT_filterX[:, :, i] = np.cos(-f * (X) / math.sin(p) + (f * f + X * X) / (2 * math.tan(p)))
            real_FrFT_filterY[:, :, i] = np.cos(-f * (Y) / math.sin(p) + (f * f + Y * Y) / (2 * math.tan(p)))
            real_FrFT_filter[:, :, i] = (real_FrFT_filterY[:, :, i] * real_FrFT_filterX[:, :, i])
        g_f = np.zeros((kernel[0], kernel[1], in_channels, out_channels))
        for i in range(N):
            g_f[:, :, :, i] = np.repeat(real_FrFT_filter[:, :, i : i + 1], in_channels, axis=2)
        g_f = np.array(g_f)
        g_f_real = g_f.reshape((out_channels, in_channels, kernel[0], kernel[1]))

        return torch.tensor(g_f_real).type(torch.FloatTensor)

    def forward(self, x):
        x = x * self.weight
        return x

    def generate_FrFT_list(self, in_channels, out_channels, kernel, f_list, p):
        FrFT_list = []
        for f in f_list:
            FrFT_list.append(self.generate_FrFT_filter(in_channels, out_channels, kernel, f, p))
        return FrFT_list


class FrFTSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super().__init__()
        self.fft = FrFTFilter(in_channels, out_channels, kernel_size, f, order)
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fft(self.t)
        out = F.conv2d(x, out, stride=1, padding=(out.shape[-2] - 1) // 2)
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class FourierFPU(nn.Module):
    def __init__(self, in_channels, out_channels, order=0.25):
        super().__init__()
        self.fft1 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.25, order)
        self.fft2 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.50, order)
        self.fft3 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.75, order)
        self.fft4 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 1.00, order)
        self.fc = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat([self.fft1(x1), self.fft2(x2), self.fft3(x3), self.fft4(x4)], dim=1)
        x_out = self.fc(x_out)
        x_out = x_out + x
        return x_out


class SPU_local(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = Conv(in_channels // 2, in_channels // 2, (1, 3), g= in_channels // 2, p=(0, 1))
        self.c2 = Conv(in_channels // 2, in_channels // 2, (3, 1), g= in_channels // 2, p=(1, 0))
        self.c3 = Conv(in_channels, in_channels, 1)
    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x1 = self.c1(x1)
        x2 = self.c2(x2 + x1)
        x_out = torch.cat([x1, x2], dim=1)
        x_out = self.c3(x_out) + x

        return x_out


class SPU_global(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = Conv(in_channels // 2, in_channels // 2, 3, d=2, g= in_channels // 2, p=2)
        self.c2 = Conv(in_channels // 2, in_channels // 2, 5, g= in_channels // 2)
        self.c3 = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x1 = self.c1(x1)
        x2 = self.c2(x2 + x1)
        x_out = torch.cat([x1, x2], dim=1)
        x_out = self.c3(x_out) + x
        return x_out


class SFS_Conv(nn.Module):
    def __init__(
        self, in_channels, out_channels):
        super().__init__()
        self.PWC0 = Conv(in_channels, in_channels // 2, 1)
        self.PWC1 = Conv(in_channels, in_channels // 2, 1)
        self.SPU_global = SPU_global(in_channels // 2, out_channels)
        self.SPU_local = SPU_local(in_channels // 2, out_channels)


        self.PWC_o = Conv(out_channels // 2, out_channels, 1)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # print("x in SFS_Conv:", x.shape)
        x_spa_1 = self.SPU_global(self.PWC0(x))
        x_spa_2 = self.SPU_local(self.PWC1(x))
        out = torch.cat([x_spa_1, x_spa_2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        output = self.PWC_o(out1 + out2)


        return output

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps
    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = True
                 ):
        super().__init__()
        
        self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.weight/sum(self.gn.weight)
        w_gamma     = w_gamma.view(1,-1,1,1)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        w1          = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
        w2          = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
        x_1         = w1 * x
        x_2         = w2 * x
        y           = self.reconstruct(x_1,x_2)
        return y
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class ScConv(nn.Module):
    def __init__(self,
                op_channel:int,
                group_num:int = 4,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRU = SRU( op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold )
        self.CRU = CRU( op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size )
    
    def forward(self,x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),nn.BatchNorm2d(outc),nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x,p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0,base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number >  0:
            mod_p_n_x,mod_p_n_y = torch.meshgrid(
                torch.arange(row_number,row_number+1),
                torch.arange(0,mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x,p_n_y  = torch.cat((p_n_x,mod_p_n_x)),torch.cat((p_n_y,mod_p_n_y))
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        # p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + self.p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    
    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
        
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset



if __name__ == '__main__':
    # Example usage
    # in_planes = 64    # 輸入通道數
    # out_planes = 128  # 輸出通道數
    # stride = 1
    # scale = 0.1
    # map_reduce = 8

    # # 建立 FEM 模塊實例
    # model = FEM(in_planes=in_planes, out_planes=out_planes, stride=stride, scale=scale, map_reduce=map_reduce)

    # # 生成一個 dummy input：假設 batch size 為 1, 高與寬為 32 (根據需要調整尺寸)
    # dummy_input = torch.randn(1, in_planes, 32, 32)

    # # 執行正向傳播
    # output = model(dummy_input)

    # # 輸出結果形狀
    # print("輸出形狀:", output.shape)

    # in_channels = 64
    # model = GlobalContextChannelAttention(64)

    # # 模擬特徵圖輸入 (batch_size=2, channels=64, height=32, width=32)
    # x_v = torch.randn(2, in_channels, 32, 32)
    # x_t = torch.randn(2, in_channels, 16, 16)

    # output = model((x_v))

    # print("輸出形狀：", output.shape)
    # hr_feat = torch.rand(1, 64, 512, 512)
    # lr_feat = torch.rand(1, 128, 256, 256)
    # model = FreqFusion(hr_channels=64, lr_channels=128)
    # concat = model(hr_feat=hr_feat, lr_feat=lr_feat)

    x = torch.randn(1, 64, 640, 640)
    model = Ehance_suppress(64, True)
    output = model(x)
    print("輸入形狀:", x.shape)
    print("輸出形狀:", output.shape)

