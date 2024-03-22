import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaborUNet(nn.Module):
    def __init__(self, kernel_size, in_channels, num_orientations, num_scales, height, width):
        super(GaborUNet, self).__init__()
        self.in_channels = in_channels
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.height = height
        self.width = width

        # Encoder (Contraction path)
        self.enc_conv1 = self.gabor_layer(in_channels, 64, kernel_size, num_orientations, num_scales)
        self.enc_conv2 = self.gabor_layer(64, 128, kernel_size, num_orientations, num_scales)
        self.enc_conv3 = self.gabor_layer(128, 256, kernel_size, num_orientations, num_scales)
        self.enc_conv4 = self.gabor_layer(256, 512, kernel_size, num_orientations, num_scales)

        # Decoder (Expansion path)
        self.dec_conv1 = self.gabor_layer(512, 256, kernel_size, num_orientations, num_scales)
        self.dec_conv2 = self.gabor_layer(512, 128, kernel_size, num_orientations, num_scales)
        self.dec_conv3 = self.gabor_layer(256, 64, kernel_size, num_orientations, num_scales)
        self.dec_conv4 = self.gabor_layer(128, 64, kernel_size, num_orientations, num_scales)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.enc_conv3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.enc_conv4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        # Decoder
        dec1 = self.dec_conv1(F.interpolate(enc4, scale_factor=2, mode='bilinear', align_corners=True))
        dec1 = torch.cat((dec1, enc3), dim=1)
        dec2 = self.dec_conv2(F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=True))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec3 = self.dec_conv3(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True))
        dec3 = torch.cat((dec3, enc1), dim=1)
        dec4 = self.dec_conv4(dec3)

        out = self.out_conv(dec4)
        return torch.sigmoid(out)

    def gabor_layer(self, in_channels, out_channels, kernel_size, num_orientations, num_scales):
        return nn.Sequential(
            GaborConv2d(in_channels, out_channels, kernel_size, num_orientations, num_scales),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class GaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_orientations, num_scales):
        super(GaborConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.num_scales = num_scales

        # Generate Gabor filter parameters
        self.sigma, self.theta, self.Lambda, self.psi, self.gamma, self.bias = self.generate_parameters(out_channels, in_channels)
        self.filter_cos = self.whole_filter(True, self.sigma, self.theta, self.Lambda, self.psi, self.gamma)
        self.filter_sin = self.whole_filter(False, self.sigma, self.theta, self.Lambda, self.psi, self.gamma)

    def forward(self, x):
        x_cos = F.conv2d(x, self.filter_cos, bias=self.bias)
        x_sin = F.conv2d(x, self.filter_sin, bias=self.bias)
        return torch.cat((x_cos, x_sin), 1)

    def generate_parameters(self, dim_out, dim_in):
        torch.manual_seed(1)
        # Adjusted to initialize parameters more appropriately for Gabor filters
        sigma = nn.Parameter(torch.rand(dim_out, 1) * 2.0 + 0.5) # Random values between 0.5 and 2.5
        theta = nn.Parameter(torch.rand(dim_out, 1) * np.pi) # Random values between 0 and π
        Lambda = nn.Parameter(torch.rand(dim_out, 1) * 3.0 + 1.0) # Random values between 1.0 and 4.0, how Lambda is good for the detection?
        psi = nn.Parameter(torch.rand(dim_out, 1) * 2 * np.pi) # Random values between 0 and 2π
        gamma = nn.Parameter(torch.rand(dim_out, 1) * 2.0 + 0.5) # Random values between 0.5 and 2.5
        bias = nn.Parameter(torch.randn(dim_out)) # to avoid division by zero
        return sigma, theta, Lambda, psi, gamma, bias


    def whole_filter(self, cos=True):
        # Creating a tensor to hold the Gabor filters for all orientations and scales
        result = torch.zeros(self.num_orientations*self.num_scales, self.in_channels, self.kernel_size, self.kernel_size)
        for i in range(self.num_orientations):
            for j in range(self.num_scales):
                index = i * self.num_scales + j
                # Adjusting parameters for scale and orientation
                sigma = self.sigma[index] * (2.1 ** j) # Adjusting sigma for scale
                theta = self.theta[index] + i * 2 * np.pi / self.num_orientations # Adjusting theta for orientation
                Lambda = self.Lambda[index] # Keeping Lambda constant
                psi = self.psi[index] # Keeping psi constant
                gamma = self.gamma[index] # Keeping gamma constant
                # Generating the Gabor filter for each channel
                for k in range(self.in_channels):
                    result[index, k] = self.gabor_fn(sigma, theta, Lambda, psi, gamma, self.kernel_size, cos)
        return nn.Parameter(result)

    def gabor_fn(self, sigma, theta, Lambda, psi, gamma, kernel_size, cos=True):
        n = kernel_size // 2
        y, x = np.ogrid[-n:n+1, -n:n+1]
        y = torch.FloatTensor(y)
        x = torch.FloatTensor(x)

        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        if cos:
            gb = torch.exp(-.5 * (x_theta ** 2 / sigma ** 2 + y_theta ** 2 / sigma ** 2 / gamma ** 2)) * torch.cos(2 * np.pi / Lambda * x_theta + psi)
        else:
            gb = torch.exp(-.5 * (x_theta ** 2 / sigma ** 2 + y_theta ** 2 / sigma ** 2 / gamma ** 2)) * torch.sin(2 * np.pi / Lambda * x_theta + psi)
        
        return gb