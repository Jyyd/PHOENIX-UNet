import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsEnhanced(nn.Module):
    '''
    A module to fuse image features with metadata features.
    Arguments:
        img_ch: Number of channels in image features (3).
        meta_dim: Dimension of metadata features.
        fuse_dim: Dimension to project metadata features before fusion.
    Returns:
        Fused image features with metadata information.
    '''
    def __init__(self, img_ch, meta_dim, fuse_dim):
        super().__init__()
        self.proj = nn.Linear(meta_dim, fuse_dim)
        self.conv = nn.Conv2d(img_ch + fuse_dim, img_ch, 1)
        self.act = nn.ReLU(inplace=True)
    def forward(self, img_feat, meta_feat):
        meta = self.proj(meta_feat).unsqueeze(-1).unsqueeze(-1) # [B, fuse_dim, 1, 1]
        meta = meta.expand(-1, -1, img_feat.shape[2], img_feat.shape[3]) # [B, fuse_dim, H, W]
        x = torch.cat([img_feat, meta], dim=1) # [B, img_ch + fuse_dim, H, W]
        return self.act(self.conv(x))

# =======================
# A) PHENUNet pyhsics-enhanced U-Net
# =======================
class PHENUNet(nn.Module):
    '''
    U-Net architecture with PhysicsEnhanced fusion at each level.
    Arguments:
        in_ch: Number of input image channels.
        out_ch: Number of output image channels.
        base_ch: Base number of channels for U-Net.
        meta_dim: Dimension of metadata features.
        fuse_dim: Dimension to project metadata features before fusion.
        depth: Depth of the U-Net.
    '''
    def __init__(self, in_ch=3, out_ch=1, base_ch=32, meta_dim=22, fuse_dim=16, depth=4):
        super().__init__()
        self.depth = depth
        self.base_ch = base_ch

        # ========== Encoder ==========
        self.enc_convs = nn.ModuleList()
        self.meta_fuse = nn.ModuleList()
        self.pools = nn.ModuleList()

        enc_channels = [in_ch] + [base_ch * 2 ** i for i in range(depth)]
        for i in range(depth):
            self.enc_convs.append(nn.Sequential(
                nn.Conv2d(enc_channels[i], enc_channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(enc_channels[i + 1]),
                nn.ReLU(inplace=True)
            ))
            self.meta_fuse.append(PhysicsEnhanced(enc_channels[i + 1], meta_dim, fuse_dim))
            self.pools.append(nn.MaxPool2d(2, 2, ceil_mode=True))

        # ========== Bottleneck ==========
        bottleneck_ch = base_ch * 2 ** depth
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[-1], bottleneck_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True)
        )
        self.meta_bottleneck = PhysicsEnhanced(bottleneck_ch, meta_dim, fuse_dim)

        # ========== Decoder ==========
        self.upconvs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.meta_fuse_dec = nn.ModuleList()
        skip_chs = enc_channels[1:][::-1]
        in_chs = [bottleneck_ch] + [base_ch * 2 ** i for i in reversed(range(1, depth))]
        for i in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(in_chs[i], skip_chs[i], kernel_size=2, stride=2))
            self.dec_convs.append(nn.Sequential(
                nn.Conv2d(skip_chs[i]*2, skip_chs[i], 3, padding=1),
                nn.BatchNorm2d(skip_chs[i]),
                nn.ReLU(inplace=True)
            ))
            self.meta_fuse_dec.append(PhysicsEnhanced(skip_chs[i], meta_dim, fuse_dim))

        self.conv_out = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    @staticmethod
    def crop_to_match(x, ref):
        h, w = ref.shape[2], ref.shape[3]
        return x[..., :h, :w]

    def forward(self, x, meta):
        enc_feats = []
        x_in = x
        # Encoder
        for i in range(self.depth):
            x_in = self.enc_convs[i](x_in)
            x_in = self.meta_fuse[i](x_in, meta)
            enc_feats.append(x_in)
            x_in = self.pools[i](x_in)
        # Bottleneck
        x_b = self.bottleneck(x_in)
        x_b = self.meta_bottleneck(x_b, meta)
        # Decoder
        x_dec = x_b
        for i in range(self.depth):
            x_dec = self.upconvs[i](x_dec)
            x_dec = self.crop_to_match(x_dec, enc_feats[self.depth - 1 - i])
            x_dec = torch.cat([x_dec, enc_feats[self.depth - 1 - i]], dim=1)
            x_dec = self.dec_convs[i](x_dec)
            x_dec = self.meta_fuse_dec[i](x_dec, meta)
        out = self.conv_out(x_dec)
        out = self.crop_to_match(out, x)
        return out
    
# =======================
# B) OnlyCNN
# =======================
class OnlyCNN(nn.Module):
    '''
    A simple CNN without physics enhancement.
    Arguments:
        in_ch: Number of input image channels.
        out_ch: Number of output image channels.
        base_ch: Base number of channels for CNN.
        depth: Number of convolutional layers.
    '''
    def __init__(self, in_ch=3, out_ch=1, base_ch=32, depth=4):
        super().__init__()
        self.depth = depth
        enc_chs = [in_ch] + [base_ch * 2 ** i for i in range(depth)]

        blocks = []
        for i in range(depth):
            blocks += [
                nn.Conv2d(enc_chs[i], enc_chs[i + 1], 3, padding=1),
                nn.BatchNorm2d(enc_chs[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv2d(enc_chs[-1], out_ch, kernel_size=1)

    def forward(self, x, meta):
        h = self.backbone(x)
        out = self.head(h)
        return out
    
# =======================
# C) Physics Enhanced CNN
# =======================
class PHENCNN(nn.Module):
    '''
    A simple CNN with PhysicsEnhanced fusion at each layer.
    Arguments:
        in_ch: Number of input image channels.
        out_ch: Number of output image channels.
        base_ch: Base number of channels for CNN.
        depth: Number of convolutional layers.
        meta_dim: Dimension of metadata features.
        fuse_dim: Dimension to project metadata features before fusion.
    '''
    def __init__(self, in_ch=3, out_ch=1, base_ch=32, depth=4, meta_dim=22, fuse_dim=16):
        super().__init__()
        self.depth = depth
        enc_chs = [in_ch] + [base_ch * (2 ** i) for i in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(enc_chs[i], enc_chs[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(enc_chs[i + 1]),
                nn.ReLU(inplace=True),
            ))

        # PhysicsEnhanced
        self.meta_fuse = nn.ModuleList([
            PhysicsEnhanced(img_ch=enc_chs[i + 1], meta_dim=meta_dim, fuse_dim=fuse_dim)
            for i in range(depth)
        ])

        self.head = nn.Conv2d(enc_chs[-1], out_ch, kernel_size=1)

    def forward(self, x, meta):
        h = x
        for i in range(self.depth):
            h = self.blocks[i](h)
            h = self.meta_fuse[i](h, meta)
        out = self.head(h)
        return out

# =======================
# D) OnlyUNet
# =======================
class OnlyUNet(nn.Module):
    '''
    Standard U-Net architecture without physics enhancement.
    Arguments:
        in_ch: Number of input image channels.
        out_ch: Number of output image channels.
        base_ch: Base number of channels for U-Net.
        depth: Depth of the U-Net.
    '''
    def __init__(self, in_ch=3, out_ch=1, base_ch=32, depth=4):
        super().__init__()
        self.depth = depth
        self.base_ch = base_ch

        enc_channels = [in_ch] + [base_ch * 2 ** i for i in range(depth)]
        bottleneck_ch = base_ch * 2 ** depth
        skip_chs = enc_channels[1:][::-1]
        in_chs = [bottleneck_ch] + [base_ch * 2 ** i for i in reversed(range(1, depth))]

        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(depth):
            self.enc_convs.append(nn.Sequential(
                nn.Conv2d(enc_channels[i], enc_channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(enc_channels[i + 1]),
                nn.ReLU(inplace=True)
            ))
            self.pools.append(nn.MaxPool2d(2, 2, ceil_mode=True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[-1], bottleneck_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True)
        )

        self.upconvs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for i in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(in_chs[i], skip_chs[i], kernel_size=2, stride=2))
            self.dec_convs.append(nn.Sequential(
                nn.Conv2d(skip_chs[i]*2, skip_chs[i], 3, padding=1),
                nn.BatchNorm2d(skip_chs[i]),
                nn.ReLU(inplace=True)
            ))

        self.conv_out = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    @staticmethod
    def crop_to_match(x, ref):
        return x[..., :ref.shape[2], :ref.shape[3]]

    def forward(self, x, meta):
        enc_feats = []
        h = x
        for i in range(self.depth):
            h = self.enc_convs[i](h)
            enc_feats.append(h)
            h = self.pools[i](h)

        h = self.bottleneck(h)

        for i in range(self.depth):
            h = self.upconvs[i](h)
            skip = enc_feats[self.depth - 1 - i]
            h = self.crop_to_match(h, skip)
            h = torch.cat([h, skip], dim=1)
            h = self.dec_convs[i](h)

        out = self.conv_out(h)
        out = self.crop_to_match(out, x)
        return out