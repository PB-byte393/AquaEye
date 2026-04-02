import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        
        # ENCODER
        self.d1 = self.conv_block(n_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.d2 = self.conv_block(64, 128)
        self.d3 = self.conv_block(128, 256)
        
        # BOTTLENECK (Named 'bot' in your weights file)
        self.bot = self.conv_block(256, 512)
        
        # DECODER
        # Layer 1 Upscale
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.u1 = self.conv_block(512, 256) # Input: 256(up) + 256(d3) = 512
        
        # Layer 2 Upscale
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.u2 = self.conv_block(256, 128) # Input: 128(up) + 128(d2) = 256
        
        # Layer 3 Upscale
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.u3 = self.conv_block(128, 64)  # Input: 64(up) + 64(d1) = 128
        
        # FINAL
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encode
        c1 = self.d1(x)
        p1 = self.pool(c1)
        
        c2 = self.d2(p1)
        p2 = self.pool(c2)
        
        c3 = self.d3(p2)
        p3 = self.pool(c3)
        
        # Bottleneck
        bn = self.bot(p3)
        
        # Decode 1
        u1 = self.up1(bn)
        # Safety crop for padding issues
        if u1.size() != c3.size():
             u1 = F.interpolate(u1, size=c3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([c3, u1], dim=1)
        x = self.u1(x)
        
        # Decode 2
        u2 = self.up2(x)
        if u2.size() != c2.size():
             u2 = F.interpolate(u2, size=c2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([c2, u2], dim=1)
        x = self.u2(x)
        
        # Decode 3
        u3 = self.up3(x)
        if u3.size() != c1.size():
             u3 = F.interpolate(u3, size=c1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([c1, u3], dim=1)
        x = self.u3(x)
        
        return self.final(x)