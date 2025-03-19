import torch
import torch.nn as nn
from .model_part import *  # 假设 model_part 包含必要的组件

class reseg(nn.Module):
    def __init__(self, 
                 inp_channels=3, 
                 out_channels=3, 
                 seg_classes=1,
                 dim=96):
        super(reseg, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.block = Transformer(dim)

        self.pa1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(dim, dim, 3, 1, groups=dim),
            nn.AdaptiveMaxPool2d(3),
            nn.Conv2d(dim, dim, 3, 1, groups=dim)
        )
        self.pa2 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(dim, dim, 3, 1, groups=dim),
            nn.AdaptiveMaxPool2d(3),
            nn.Conv2d(dim, dim, 3, 1, groups=dim)
        )
        self.pa3 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(dim, dim, 3, 1, groups=dim),
            nn.AdaptiveMaxPool2d(3),
            nn.Conv2d(dim, dim, 3, 1, groups=dim)
        )
        
        self.restoration_output = nn.Conv2d(int(dim), out_channels, kernel_size=3, padding=1)
        self.segmentation_output = nn.Conv2d(int(dim), seg_classes, kernel_size=3, padding=1)

    def forward(self, inp_img):
        ## restoration ##
        x = self.patch_embed(inp_img)

        x1 = self.block(x)
        x2 = self.block(x1)
        x3 = self.block(x2)

        x4 = self.block(x3)
        t1, t2, t3 = self.pa1(x1), self.pa2(x2), self.pa3(x3)

        x5 = self.block(x4)
        x5 = x5 * t1

        x6 = self.block(x5)
        x6 = x6 * t2

        x7 = self.block(x6)
        x7 = x7 * t3

        restoration = self.restoration_output(x7) + inp_img
        ## restoration ##
        #---------------------
        ## segmentation ##
        y = self.patch_embed(restoration)

        y1 = self.block(y)
        y2 = self.block(y1)
        y3 = self.block(y2)

        y4 = self.block(y3)
        t1, t2, t3 = self.pa1(y1), self.pa2(y2), self.pa3(y3)

        y5 = self.block(y4)
        y5 = y5 * t1

        y6 = self.block(y5)
        y6 = y6 * t2

        y7 = self.block(y6)
        y7 = y7 * t3

        segmentation = self.segmentation_output(y7)
        ## segmentation


        return restoration, segmentation