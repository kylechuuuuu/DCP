import torch
import torch.nn as nn
from .model_part import *


class reseg(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        seg_classes=1,
        dim=72,
        num_blocks=[1, 1, 1, 1],
        num_stages=2,
    ):
        super(reseg, self).__init__()

        self.num_stages = num_stages

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed2 = OverlapPatchEmbed2(seg_classes, dim)

        self.encoder_level1 = nn.Sequential(*[Transformer(dim) for _ in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[Transformer(dim * 2**1) for _ in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(*[Transformer(dim * 2**2) for _ in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(*[Transformer(dim * 2**3) for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1)
        self.decoder_level3 = nn.Sequential(*[Transformer(dim * 2**2) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1)
        self.decoder_level2 = nn.Sequential(*[Transformer(dim * 2**1) for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.Sequential(*[Transformer(dim * 2**1) for _ in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[Transformer(dim * 2**1) for _ in range(num_blocks[0])])

        self.restoration_output = nn.Conv2d(int(dim * 2**1), out_channels, kernel_size=3, padding=1)
        self.segmentation_output = nn.Conv2d(int(dim * 2**1), seg_classes, kernel_size=3, padding=1)

    def forward_stage(self, inp_feature):
        out_enc_level1 = self.encoder_level1(inp_feature)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3_restoration = self.up4_3(latent)
        inp_dec_level3_restoration = torch.cat(
            [inp_dec_level3_restoration, out_enc_level3], 1
        )
        inp_dec_level3_restoration = self.reduce_chan_level3(inp_dec_level3_restoration)
        out_dec_level3_restoration = self.decoder_level3(inp_dec_level3_restoration)

        inp_dec_level2_restoration = self.up3_2(out_dec_level3_restoration)
        inp_dec_level2_restoration = torch.cat(
            [inp_dec_level2_restoration, out_enc_level2], 1
        )
        inp_dec_level2_restoration = self.reduce_chan_level2(inp_dec_level2_restoration)
        out_dec_level2_restoration = self.decoder_level2(inp_dec_level2_restoration)

        inp_dec_level1_restoration = self.up2_1(out_dec_level2_restoration)
        inp_dec_level1_restoration = torch.cat(
            [inp_dec_level1_restoration, out_enc_level1], 1
        )
        out_dec_level1_restoration = self.decoder_level1(inp_dec_level1_restoration)

        out_dec_level1_restoration = self.refinement(out_dec_level1_restoration)
        segmentation_out = self.segmentation_output(out_dec_level1_restoration)

        return segmentation_out

    def forward(self, inp_img):
        # Stage one for segmentation
        inp_enc_level = self.patch_embed(inp_img)
        segmentation = self.forward_stage(inp_enc_level)

        segmentations = [segmentation]
        for _ in range(1, self.num_stages):
            inp_enc_level = self.patch_embed2(segmentations[-1])
            segmentation = self.forward_stage(inp_enc_level) + segmentations[-1]
            segmentations.append(segmentation)

        return segmentations
        
class DualReseg(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        seg_classes=1,
        dim=72,
        num_blocks=[1, 1, 1, 1],
        num_stages=4,
    ):
        super(DualReseg, self).__init__()
        
        self.model1 = reseg(inp_channels, out_channels, seg_classes, dim, num_blocks, num_stages)
        self.model2 = reseg(inp_channels, out_channels, seg_classes, dim, num_blocks, num_stages)


    def forward(self, inp_img):
        segmentation1 = self.model1(inp_img)
        segmentation2 = self.model2(inp_img)

        output = torch.max(segmentation1[-1], segmentation2[-1])

        return segmentation1[0], segmentation1[1], segmentation1[2], segmentation1[3], segmentation2[0], segmentation2[1], segmentation2[2], segmentation2[3], output


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    model = reseg(3)
    print(model(x))
