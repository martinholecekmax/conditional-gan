import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        # features_g will be 64
        self.gen = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(
                channels_noise + embed_size, features_g * 16, 4, 1, 0
            ),  # img: Output 4x4 (N x f_g * 16 x 4 x 4)
            self._block(
                features_g * 16, features_g * 8, 4, 2, 1
            ),  # img: Output 8x8 (N x f_g * 8 x 8 x 8)
            self._block(
                features_g * 8, features_g * 4, 4, 2, 1
            ),  # img: Output 16x16 (N x f_g * 4 x 16 x 16)
            self._block(
                features_g * 4, features_g * 2, 4, 2, 1
            ),  # img: Output 32x32 (N x f_g * 2 x 32 x 32)
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  # Output 64x64 (N x channels_img x 64 x 64)
            nn.Tanh(),  # [-1, 1] range (We will normalize images to be in this range)
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,  # Need to be false because we are using batchnorm layer
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # Add dimensions. Latent vector z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
