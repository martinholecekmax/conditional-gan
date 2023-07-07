""" 
Discriminator for the WGAN-GP model 
"""
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Critic, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # Input: N x (channels_img + label)  x 64 x 64
            nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),
            # Output 32x32
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),  # Output 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # Output 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # Output 4x4
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)

        # N (num examples in batch) x C (channels) x img_size (H) x img_size (W)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)
