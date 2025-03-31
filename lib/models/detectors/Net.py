
from lib.RCAB import *
from lib.models.backbones.pvtv2 import pvt_v2_b4


# Channel Reduce
class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel, RFB=False):
        super(Reduction, self).__init__()
        # self.dyConv = Dynamic_conv2d(in_channel,out_channel,3,padding = 1)
        if (RFB):
            self.reduce = nn.Sequential(
                RFB_modified(in_channel, out_channel),
            )
        else:
            self.reduce = nn.Sequential(
                BasicConv2d(in_channel, out_channel, 1),
            )

    def forward(self, x):
        return self.reduce(x)


#
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SGR(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SGR, self).__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention()
        self.cb_high = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.cv_low = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.cb_1 = nn.Conv2d(2*channel,channel, 1)


    def forward(self, x_low, x_high):
        x_high = self.cb_high(x_high)
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear')
        x_low = self.cv_low(x_low)
        x = torch.cat((x_high, x_low), dim=1)
        # Spatial Attention
        x_sa = self.sa(x)
        # Channle Attention
        x_ca = self.cb_1(x)
        x_ca = self.ca(x_ca)

        x = x_ca * x_sa * x_high
        x = x_high +x


        return x

# SFC module：Scale-wise Feature Capturing Module
class SFC(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(SFC, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.conv_cat = nn.Sequential(
            BasicConv2d(in_channel * 3, out_channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),

        )

        self.ConvOut = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        out = self.ConvOut(x_cat +x)

        return out


# BFRE Subnetwork：Background-assisted Foreground Region Enhancement
class BFRE(nn.Module):
    def __init__(self, channel=64):
        super(BFRE, self).__init__()

        self.cb_1 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.cb_2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.cbr_1 = ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        self.cbr_2 = ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)

        self.sfc = SFC(in_channel=channel, out_channel=channel)
        self.cbfuse = nn.Sequential(
            BasicConv2d(2*channel, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, fg_high, fg_low, bg_high):#up表示高分辨率特征图，down表示低分辨率特征图
        fg_high_1 = self.cb_1(fg_high)
        fg_low = F.interpolate(fg_low, size=fg_high.size()[2:], mode='bilinear')
        fg_low_1 = self.cb_2(fg_low)
        # bg_high_1 = self.cbr_3(-bg_high)

        fg_high_2 = self.cbr_2(fg_high_1)
        fg_low_2 = self.cbr_1(fg_low_1)

        z1 = F.relu((fg_high_1-bg_high) * fg_low_2, inplace=True)

        z2 = F.relu((fg_low_1-bg_high) * fg_high_2, inplace=True)

        fuse = self.cbfuse(torch.cat((z1,z2),1))# 相当于3fg和一个bg的比例，所以需要
        # fuse and Integrity enhence
        out = self.sfc(fuse)

        return out


class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()
        #  ---- PVTv2_B4 Backbone ----

        self.bkbone = pvt_v2_b4()  # [64, 128, 320, 512]
        # 获取预训练的参数
        save_model = torch.load('./pretrained/pvt_v2_b4.pth')
        # 获取当前模型的参数
        model_dict = self.bkbone.state_dict()
        # 加载部分能用的参数
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # 更新现有的model_dict
        model_dict.update(state_dict)
        # 加载真正需要的state_dict
        self.bkbone.load_state_dict(model_dict)
        enc_channels = [64, 128, 320, 512]

        self.reduce_1 = Reduction(enc_channels[0], channel, RFB=False)
        self.reduce_2 = Reduction(enc_channels[1], channel, RFB=False)
        self.reduce_3 = Reduction(enc_channels[2], channel, RFB=False)
        self.reduce_4 = Reduction(enc_channels[3], channel, RFB=False)

        self.reduce_5 = Reduction(enc_channels[0], channel, RFB=False)
        self.reduce_6 = Reduction(enc_channels[1], channel, RFB=False)
        self.reduce_7 = Reduction(enc_channels[2], channel, RFB=False)
        self.reduce_8 = Reduction(enc_channels[3], channel, RFB=False)

        self.bfre1 = BFRE(channel)
        self.bfre2 = BFRE(channel)
        self.bfre3 = BFRE(channel)
        self.fbre1 = BFRE(channel=channel)
        self.fbre2 = BFRE(channel=channel)
        self.fbre3 = BFRE(channel=channel)

        self.sgr1 = SGR(channel)
        self.sgr2 = SGR(channel)
        self.sgr3 = SGR(channel)
        self.sgr4 = SGR(channel)
        self.sgr5 = SGR(channel)
        self.sgr6 = SGR(channel)


        self.pre_fg1 = nn.Conv2d(channel, 1, 1)
        self.pre_fg2 = nn.Conv2d(channel, 1, 1)
        self.pre_fg3 = nn.Conv2d(channel, 1, 1)
        self.pre_bg1 = nn.Conv2d(channel, 1, 1)
        self.pre_bg2 = nn.Conv2d(channel, 1, 1)
        self.pre_bg3 = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        # Feature Extraction
        shape = x.size()[2:]
        #  ---- PVTv2_B4 Backbone ----
        x1, x2, x3, x4 = self.bkbone(x)

        # Channel Reduce
        x1_fg = self.reduce_1(x1)
        x2_fg = self.reduce_2(x2)
        x3_fg = self.reduce_3(x3)
        x4_fg = self.reduce_4(x4)

        x1_bg = self.reduce_5(-x1.clone())
        x2_bg = self.reduce_6(-x2.clone())
        x3_bg = self.reduce_7(-x3.clone())
        x4_bg = self.reduce_8(-x4.clone())

    # stage 1
        # SGR in fg
        sgr_fg = self.sgr1(x4_fg,x1_fg)

        # SGR in bg
        sgr_bg = self.sgr2(x4_bg,x1_bg)

        # FBRE in bg: fg指导bg
        bg_1 = self.fbre1(x3_bg, x4_bg,F.interpolate(x3_fg,size=x3_bg.size()[2:],mode='bilinear'))  # 24*24
        # BFRE in fg
        fg_1 = self.bfre1(x3_fg, x4_fg, bg_1)  # B*C*24*24
    # stage 2
        # SGR in fg
        sgr_fg = self.sgr3(fg_1,sgr_fg)

        # SGR in bg
        sgr_bg = self.sgr4(bg_1,sgr_bg)

        # FBRE in bg: fg指导bg
        bg_2 = self.fbre2(x2_bg, bg_1,F.interpolate(fg_1,size=x2_bg.size()[2:],mode='bilinear'))  # 48*48
        # BFRE in fg
        fg_2 = self.bfre2(x2_fg, fg_1, bg_2)  # B*C*48*48
    # stage 3
        # SGR in fg
        sgr_fg = self.sgr5(fg_2,sgr_fg)

        # SGR in bg
        sgr_bg = self.sgr6(bg_2,sgr_bg)

        # FBRE in bg: fg指导bg
        bg_3 = self.fbre3(sgr_bg, bg_2,F.interpolate(fg_2,size=sgr_bg.size()[2:],mode='bilinear'))  # 96*96
        # BFRE in fg
        fg_3 = self.bfre3(sgr_fg, fg_2, bg_3)

        pred_fg1 = F.interpolate(self.pre_fg1(fg_1), size=shape, mode='bilinear')
        pred_fg2 = F.interpolate(self.pre_fg2(fg_2), size=shape, mode='bilinear')
        pred_fg3 = F.interpolate(self.pre_fg3(fg_3), size=shape, mode='bilinear')  # final pred

        pred_bg1 = F.interpolate(self.pre_bg1(bg_1), size=shape, mode='bilinear')
        pred_bg2 = F.interpolate(self.pre_bg2(bg_2), size=shape, mode='bilinear')
        pred_bg3 = F.interpolate(self.pre_bg3(bg_3), size=shape, mode='bilinear')

        return pred_fg1, pred_fg2, pred_fg3, pred_bg1, pred_bg2, pred_bg3


if __name__ == '__main__':
    import numpy as np
    from time import time

    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 384, 384)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
