import torch
import torch.nn as nn
from .layernorm import LayerNorm2d
from .unet_model import UNet
from .unetplus import NestedUNet


class Channel(nn.Module):
	def __init__(self, channels):
		super(Channel, self).__init__()
		self.maxpool = nn.AdaptiveMaxPool2d(3)
		self.mlp = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=3, groups=channels) for _ in range(12)])

	def forward(self, x):
		avg_out = self.maxpool(x)
		output = [nn.Sigmoid()(nn.LeakyReLU()(mlp(avg_out))) for mlp in self.mlp]
		add = nn.LeakyReLU()(sum(output))
		out = nn.LeakyReLU()(add * x)

		return out


class Spatial(nn.Module):
	def __init__(self, channels):
		super(Spatial, self).__init__()
		self.project_in = nn.Conv2d(channels, channels*3, kernel_size=1)
		self.dwconv = nn.Conv2d(channels*3, channels*3, kernel_size=3, padding=1, groups=channels*3)
		self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

	def forward(self, x):

		x = self.project_in(x)
		x1, x2, x3 = self.dwconv(x).chunk(3, dim=1)
		a = nn.LeakyReLU()(x1 * x2)
		b = nn.LeakyReLU()(a * x3)
		out = self.project_out(b)
		out = nn.LeakyReLU()(out)
		return out


class MLP(nn.Module):
	def __init__(self, channels):
		super(MLP, self).__init__()

		self.body = nn.Sequential(
				nn.Conv2d(channels, channels, kernel_size=1),
				nn.LeakyReLU(),
				nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, groups=channels)
				)

	def forward(self, x):

		return self.body(x)


class Transformer(nn.Module):
	def __init__(self, channels):
		super(Transformer, self).__init__()

		self.norm1 = LayerNorm2d(channels)
		self.norm2 = LayerNorm2d(channels)

		self.spatial_blocks = Spatial(channels)
		self.channel_blocks = Channel(channels)

		self.mlp = MLP(channels)

	def forward(self, img):

		x = self.norm1(img)

		x_1 = self.spatial_blocks(x)
		x_2 = nn.LeakyReLU()(x_1)
		x_3 = self.channel_blocks(x_2)
		x_4 = nn.LeakyReLU()(x_3)
		y = x_4 + img

		y_1 = self.norm2(y)

		y_2 = self.mlp(y_1)

		out = y_2 + y

		return out


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class RestorationAndSegmentation(nn.Module):
    def __init__(self, inp_channels=3, out_channels=1, dim=64, num_blocks=[1, 1, 1, 1]):
        super(RestorationAndSegmentation, self).__init__()
        
        # Encoder part
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[Transformer(dim) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[Transformer(dim*2**1) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[Transformer(dim*2**2) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[Transformer(dim*2**3) for i in range(num_blocks[3])])

        # Restoration Decoder part
        self.up4_3_restoration = Upsample(int(dim*2**3))
        self.reduce_chan_level3_restoration = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1)
        self.decoder_level3_restoration = nn.Sequential(*[Transformer(dim*2**2) for i in range(num_blocks[2])])
        self.up3_2_restoration = Upsample(int(dim*2**2))
        self.reduce_chan_level2_restoration = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1)
        self.decoder_level2_restoration = nn.Sequential(*[Transformer(dim*2**1) for i in range(num_blocks[1])])
        self.up2_1_restoration = Upsample(int(dim*2**1))
        self.decoder_level1_restoration = nn.Sequential(*[Transformer(dim*2**1) for i in range(num_blocks[0])])
        self.refinement_restoration = nn.Sequential(*[Transformer(dim*2**1) for i in range(num_blocks[0])])
        self.output_restoration = nn.Conv2d(int(dim*2**1), inp_channels, kernel_size=3, stride=1, padding=1)

        # Segmentation Decoder part
        self.up4_3_segmentation = Upsample(int(dim*2**3))
        self.reduce_chan_level3_segmentation = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1)
        self.decoder_level3_segmentation = nn.Sequential(*[Transformer(dim*2**2) for i in range(num_blocks[2])])
        self.up3_2_segmentation = Upsample(int(dim*2**2))
        self.reduce_chan_level2_segmentation = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1)
        self.decoder_level2_segmentation = nn.Sequential(*[Transformer(dim*2**1) for i in range(num_blocks[1])])
        self.up2_1_segmentation = Upsample(int(dim*2**1))
        self.decoder_level1_segmentation = nn.Sequential(*[Transformer(dim*2**1) for i in range(num_blocks[0])])
        self.refinement_segmentation = nn.Sequential(*[Transformer(dim*2**1) for i in range(num_blocks[0])])
        self.output_segmentation = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inp_img):
        # Encoder
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # Restoration Decoder
        inp_dec_level3_restoration = self.up4_3_restoration(latent)
        inp_dec_level3_restoration = torch.cat([inp_dec_level3_restoration, out_enc_level3], 1)
        inp_dec_level3_restoration = self.reduce_chan_level3_restoration(inp_dec_level3_restoration)
        out_dec_level3_restoration = self.decoder_level3_restoration(inp_dec_level3_restoration)
        inp_dec_level2_restoration = self.up3_2_restoration(out_dec_level3_restoration)
        inp_dec_level2_restoration = torch.cat([inp_dec_level2_restoration, out_enc_level2], 1)
        inp_dec_level2_restoration = self.reduce_chan_level2_restoration(inp_dec_level2_restoration)
        out_dec_level2_restoration = self.decoder_level2_restoration(inp_dec_level2_restoration)
        inp_dec_level1_restoration = self.up2_1_restoration(out_dec_level2_restoration)
        inp_dec_level1_restoration = torch.cat([inp_dec_level1_restoration, out_enc_level1], 1)
        out_dec_level1_restoration = self.decoder_level1_restoration(inp_dec_level1_restoration)
        out_dec_level1_restoration = self.refinement_restoration(out_dec_level1_restoration)
        out_restoration = self.output_restoration(out_dec_level1_restoration)

        # Segmentation Decoder
        inp_dec_level3_segmentation = self.up4_3_segmentation(latent)
        inp_dec_level3_segmentation = torch.cat([inp_dec_level3_segmentation, out_enc_level3], 1)
        inp_dec_level3_segmentation = self.reduce_chan_level3_segmentation(inp_dec_level3_segmentation)
        out_dec_level3_segmentation = self.decoder_level3_segmentation(inp_dec_level3_segmentation)
        inp_dec_level2_segmentation = self.up3_2_segmentation(out_dec_level3_segmentation)
        inp_dec_level2_segmentation = torch.cat([inp_dec_level2_segmentation, out_enc_level2], 1)
        inp_dec_level2_segmentation = self.reduce_chan_level2_segmentation(inp_dec_level2_segmentation)
        out_dec_level2_segmentation = self.decoder_level2_segmentation(inp_dec_level2_segmentation)
        inp_dec_level1_segmentation = self.up2_1_segmentation(out_dec_level2_segmentation)
        inp_dec_level1_segmentation = torch.cat([inp_dec_level1_segmentation, out_enc_level1], 1)
        out_dec_level1_segmentation = self.decoder_level1_segmentation(inp_dec_level1_segmentation)
        out_dec_level1_segmentation = self.refinement_segmentation(out_dec_level1_segmentation)
        out_segmentation = self.output_segmentation(out_dec_level1_segmentation)

        return out_restoration, out_segmentation