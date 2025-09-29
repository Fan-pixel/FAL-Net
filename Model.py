import torch
from torch import nn
import torch.nn.functional as F
import functools
from fremamba import fremamba
from Model_util import PALayer, ConvGroups, FE_Block, Fusion_Block, ResnetBlock, ConvBlock, CALayer, SKConv
from Parameter_test import atp_cal, Dense
# from mods import HFRM

class fusion_h(nn.Module):
    def __init__(self, dim=3, block_depth=3):
        super(fusion_h, self).__init__()
        self.sig = nn.Sigmoid()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv_fuse = nn.Conv2d(dim, dim, 1)

    def forward(self, x, y):
        x = self.sig(x)
        y = self.conv1(y)
        return self.conv_fuse(x * y)


class Conv1x1(nn.Module):
    def __init__(self, inc=3, outc=3):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inc, outc, 1)

    def forward(self, x):
        return self.conv(x)




# 专家分支模块（之前设计的）
class ExpertBranch(nn.Module):
    def __init__(self, channels, kernel_size, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# 专家系统门控反馈模块
class ExpertGatedFeedback(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()

        self.expert_light = ExpertBranch(channels, kernel_size=3, depth=2)
        self.expert_medium = ExpertBranch(channels, kernel_size=5, depth=3)
        self.expert_heavy = ExpertBranch(channels, kernel_size=7, depth=4)

        self.gate_fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 2, 3, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Softmax(dim=1)
        )

        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, dec_feat, enc_feat, haze_level_map=None):
        if enc_feat.shape[2:] != dec_feat.shape[2:]:
            enc_feat = F.interpolate(enc_feat, size=dec_feat.shape[2:], mode='bilinear', align_corners=False)

        light_feat = self.expert_light(dec_feat)
        medium_feat = self.expert_medium(dec_feat)
        heavy_feat = self.expert_heavy(dec_feat)

        # print('dec_feat.shape:', dec_feat.shape)
        # print('enc_feat.shape:', enc_feat.shape)


        gate_weights = self.gate_fc(torch.cat([dec_feat, enc_feat], dim=1))  # ← in_channels 就是 dec+enc 的通道和

        if haze_level_map is not None:
            haze_level_map = F.adaptive_avg_pool2d(haze_level_map, 1).expand(-1, 3, -1, -1)
            gate_weights = gate_weights * haze_level_map
            gate_weights = gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-6)

        fused_expert = (
            gate_weights[:, 0:1, :, :] * light_feat +
            gate_weights[:, 1:2, :, :] * medium_feat +
            gate_weights[:, 2:3, :, :] * heavy_feat
        )
        if enc_feat.shape[1] != fused_expert.shape[1]:
            enc_feat = F.interpolate(enc_feat, size=fused_expert.shape[2:], mode='bilinear', align_corners=False)
            enc_feat = nn.Conv2d(enc_feat.shape[1], fused_expert.shape[1], kernel_size=1).to(enc_feat.device)(enc_feat)

        gate = self.gate_conv(torch.cat([fused_expert, enc_feat], dim=1))

        fused = gate * fused_expert + (1 - gate) * enc_feat
        return self.fuse(fused)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 1. 轻量轮廓检测模块（输出轮廓掩码）
class ContourDetector(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        mask = self.sigmoid(x)  # [0,1]之间，代表轮廓概率
        return mask


# ---------------------------
# 2. 可微分“频谱”变换模拟（用卷积替代STFT，示范用）
class DifferentiableSTFT(nn.Module):
    def __init__(self, in_channels=3, freq_channels=16):
        super().__init__()
        # 模拟频谱变换为多通道卷积，感受野代表频率带
        self.conv_freq = nn.Conv2d(in_channels, freq_channels, kernel_size=7, padding=3)
        self.conv_inv = nn.ConvTranspose2d(freq_channels, in_channels, kernel_size=7, padding=3)
    
    def forward(self, x):
        # x: [B,3,H,W]
        freq_feat = self.conv_freq(x)
        return freq_feat
    
    def inverse(self, freq_feat):
        # freq_feat: [B,freq_channels,H,W]
        x_recon = self.conv_inv(freq_feat)
        return x_recon


# ---------------------------
# 3. 频谱增强模块，结合轮廓门控
class FrequencyEnhancer(nn.Module):
    def __init__(self, freq_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(freq_channels, freq_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(freq_channels, freq_channels, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, freq_feat, contour_mask):
        # contour_mask: [B,1,H,W]，扩展通道匹配freq_feat
        mask_expanded = contour_mask.expand_as(freq_feat)
        gated = freq_feat * mask_expanded  # 动态门控增强轮廓区域
        out = self.relu(self.conv1(gated))
        out = self.relu(self.conv2(out))
        return out


class Base_Model(nn.Module):
    def __init__(self, ngf=64, bn=False):
        super(Base_Model, self).__init__()
        # 下采样
        self.down1 = ResnetBlock(3, first=True)

        self.down2 = ResnetBlock(ngf, levels=2)

        self.down3 = ResnetBlock(ngf * 2, levels=2, bn=bn)

        self.down1_high = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(3, ngf, kernel_size=7, padding=0),
                                        nn.InstanceNorm2d(ngf),
                                        nn.ReLU(True))

        self.down2_high = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf * 2),
                                        nn.ReLU(True))

        self.down3_high = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf * 4),
                                        nn.ReLU(True))

        self.res = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True),
            CALayer(ngf * 4),
            PALayer(ngf * 4)
        )

        self.res_atp = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True),
            CALayer(ngf * 4),
            PALayer(ngf * 4)
        )

        self.res_tran = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True),
            CALayer(ngf * 4),
            PALayer(ngf * 4)
        )

        self.fusion_layer = nn.ModuleList([fusion_h(dim=2 ** i * ngf) for i in range(0, 3)])
        self.skfusion = SKConv(features=2 ** 3 * ngf)
        self.conv1 = Conv1x1(inc=2 ** 3 * ngf, outc=2 ** (3 - 1) * ngf)
        # 上采样

        self.up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(ngf * 2) if not bn else nn.BatchNorm2d(ngf * 2),
            CALayer(ngf * 2),
            PALayer(ngf * 2))

        self.up2 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            CALayer(ngf),
            PALayer(ngf))

        self.up3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh())

        self.info_up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(ngf * 2) if not bn else nn.BatchNorm2d(ngf * 2, eps=1e-5),
        )

        self.info_up2 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf)  # if not bn else nn.BatchNorm2d(ngf, eps=1e-5),
        )

        self.fam1 = FE_Block(ngf, ngf)
        self.fam2 = FE_Block(ngf, ngf * 2)
        self.fam3 = FE_Block(ngf * 2, ngf * 4)

        self.att1 = Fusion_Block(ngf)
        self.att2 = Fusion_Block(ngf * 2)
        self.att3 = Fusion_Block(ngf * 4, bn=bn)

        self.merge2 = nn.Sequential(
            ConvBlock(ngf * 2, ngf * 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        self.merge3 = nn.Sequential(
            ConvBlock(ngf, ngf, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )

        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.atp = atp_cal()
        self.tran = Dense()

        
        # 设置参数
        dim = 256
        d_state = 32
        input_resolution = (64, 64)
        self.num_tokens = 32
        inner_rank = 32
        mlp_ratio = 2.0

        self.conv1 = nn.Conv2d(256, dim, kernel_size=3, stride=1, padding=1)
        self.fremamba = fremamba(dim=dim, d_state=d_state, input_resolution=input_resolution, num_tokens=self.num_tokens, inner_rank=inner_rank, mlp_ratio=mlp_ratio)
        self.conv2 = nn.Conv2d(dim, 256, kernel_size=3, stride=1, padding=1)


         # 替换fam2、fam3为专家反馈模块
        self.expert_feedback2 = ExpertGatedFeedback(ngf* 2 + 3,channels=ngf* 2)
        self.expert_feedback3 = ExpertGatedFeedback(ngf * 4+ 3,channels=ngf* 4)


                # 新增频谱增强模块
        self.contour_detector = ContourDetector(in_channels=3)
        self.stft = DifferentiableSTFT(in_channels=3, freq_channels=16)
        self.freq_enhancer = FrequencyEnhancer(freq_channels=16)
        self.freq_recon_conv = nn.Conv2d(3, ngf * 4, kernel_size=1)

    def forward(self, hazy, high):


        x_down1_high = self.down1_high(hazy)


        x_down2_high = self.down2_high(x_down1_high)  # [bs, 128, 128, 128]

        x_down3_high = self.down3_high(x_down2_high)  # [bs, 256, 64, 64]
#------------------------------------------------------------------------
    # === [2+3] fremamba: 内容驱动token + soft prompt ===
        x_embed = self.conv1(x_down3_high)        # [B, 256, 64, 64]
        B, C, H, W = x_embed.shape
        x_seq = x_embed.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]

        # 使用 CNN+AvgPool 驱动的 token
        avg_pool = F.adaptive_avg_pool2d(x_embed, (1, 1))    # [B, 256, 1, 1]
        content_token = avg_pool.squeeze(-1).squeeze(-1)     # [B, 256]
        token = content_token.unsqueeze(1).repeat(1, self.fremamba.num_tokens, 1)  # [B, num_tokens, C]

        # 调用 fremamba（已内置 soft prompt）
        x_fused = self.fremamba(x_seq, (H, W), token)

        # 恢复回 feature map
        x_fused = x_fused.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_down3_high = self.conv2(x_fused)  # [B, 256, 64, 64]
        ###############   hazy encode   ###################

        x_down1 = self.down1(hazy)  # [bs, ngf, ngf * 4, ngf * 4]

        att1 = self.att1(x_down1_high, x_down1)

        x_down2 = self.down2(x_down1)  # [bs, ngf*2, ngf*2, ngf*2]
        att2 = self.att2(x_down2_high, x_down2)
        # print("att2",att2.shape)

        x_down3 = self.down3(x_down2)  # [bs, ngf * 4, ngf, ngf]
        att3 = self.att3(x_down3_high, x_down3)

              # ---------------- 新增频谱增强 ----------------
        contour_mask = self.contour_detector(hazy)          # [B,1,H,W]
        freq_feat = self.stft(hazy)                          # [B,16,H,W]
        freq_enhanced = self.freq_enhancer(freq_feat, contour_mask)  # [B,16,H,W]
        freq_recon = self.stft.inverse(freq_enhanced)       # [B,3,H,W]
        if freq_recon.shape[2:] != x_down3.shape[2:]:
            freq_recon = F.interpolate(freq_recon, size=x_down3.shape[2:], mode='bilinear', align_corners=False)

        freq_recon = self.freq_recon_conv(freq_recon)  # [B, ngf*4, H, W]

        # 频谱增强特征融合到x_down3（你也可以试拼接+1x1conv）
        x_down3 = x_down3 + freq_recon

                # 更新 att3
        att3 = self.att3(x_down3_high, x_down3)


        ############### 透射率估计 ###############
        x6_tran = self.res_tran(x_down3)
        tran = self.tran(x6_tran, hazy)             # 透射率t，shape和x_down3一致或可调整

        # 计算雾浓度图
        haze_level_map = 1.0 - tran
        haze_level_map = torch.clamp(haze_level_map, 0, 1).detach()

        # print("haze_level_map",haze_level_map.shape)


        ############### 专家融合门控 ###############
        att2 = self.expert_feedback2(att2, haze_level_map)
        fuse2 = self.fam2(att1, att2)

        att3 = self.expert_feedback3(att3, haze_level_map)
        fuse3 = self.fam3(fuse2, att3)
    
        ###############   dehaze   ###################

        x6 = self.res(x_down3)

        fuse_up2 = self.info_up1(fuse3)
        fuse_up2 = self.merge2(fuse_up2 + x_down2)

        fuse_up3 = self.info_up2(fuse_up2)
        fuse_up3 = self.merge3(fuse_up3 + x_down1)

        x_up2 = self.up1(x6 + fuse3)

        x_up3 = self.up2(x_up2 + fuse_up2)

        x_up4 = self.up3(x_up3 + fuse_up3)

        # model = HFRM(in_channels=3, out_channels=3).cuda()
        # x_up4 = model(x_up4)
        ###############   atp   ###################

        x6_atp = self.res_atp(x_down3)
        atp = self.atp(x6_atp)

        ###############   tran   ###################

        x_up4 = x_up4.to(x6_tran.device)

        x6_tran = self.res_tran(x_down3)
        tran = self.tran(x6_tran, x_up4)

        ##############  Atmospheric scattering model  #################

        zz = torch.abs((tran)) + (10 ** -10)  # t
        shape_out1 = atp.data.size()

        shape_out = shape_out1[2:4]
        if shape_out1[2] >= shape_out1[3]:
            atp = F.avg_pool2d(atp, shape_out1[3])
        else:
            atp = F.avg_pool2d(atp, shape_out1[2])
        atp = self.upsample(self.relu(atp), size=shape_out)

        haze = (x_up4 * zz) + atp * (1 - zz)
        dehaze = (hazy - atp) / zz + atp  # 去雾公式

        return haze, dehaze, x_up4, tran, atp


if __name__ == '__main__':
    G = Base_Model()
    a = torch.randn(1, 3, 512, 768)
    b = torch.randn(1, 3, 512, 768)
    G(a, b)


class Discriminator(nn.Module):
    """
    Discriminator class
    """

    def __init__(self, inp=3, out=1):
        """
        Initializes the PatchGAN model with 3 layers as discriminator

        Args:
        inp: number of input image channels
        out: number of output image channels
        """

        super(Discriminator, self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        model = [
            nn.Conv2d(inp, 64, kernel_size=4, stride=2, padding=1),  # input 3 channels
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, out, kernel_size=4, stride=1, padding=1)  # output only 1 channel (prediction map)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
            Feed forward the image produced by generator through discriminator

            Args:
            input: input image

            Returns:
            outputs prediction map with 1 channel
        """
        result = self.model(input)

        return result

# class UpSample(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UpSample, self).__init__()
#         self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                                 nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
#
#     def forward(self, x):
#         x = self.up(x)
#         return x
