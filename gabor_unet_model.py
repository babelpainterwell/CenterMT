import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaborUNet(nn.Module):
    def __init__(self, kernel_size, in_channels=1, out_channels=1, num_orientations=8, num_scales=5):
        super(GaborUNet, self).__init__()
        self.in_channels = in_channels
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        # Encoder (Contraction path)
        self.enc_conv1 = self.doubleGaborConv(in_channels, kernel_size, num_orientations, num_scales)
        self.enc_conv2 = self.doubleConv3x3(80, 32)
        self.enc_conv3 = self.doubleConv3x3(32, 64)
        self.enc_conv4 = self.doubleConv3x3(64, 128)

        # Decoder (Expansion path)
        self.dec_conv1 = self.doubleConv3x3(128, 64)
        self.dec_conv2 = self.doubleConv3x3(128, 32)
        self.dec_conv3 = self.doubleConv3x3(64, 16)

        self.out_conv = nn.Conv2d(96, 1, kernel_size=1)

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
        out = self.out_conv(dec3)
        return torch.sigmoid(out)

    def doubleGaborConv(self, in_channels, kernel_size, num_orientations, num_scales):
        return nn.Sequential(
            GaborConv2d(in_channels, kernel_size, num_orientations, num_scales),
            nn.BatchNorm2d(2 * num_orientations * num_scales), 
            nn.ReLU(inplace=True),
            GaborConv2d(2 * num_orientations * num_scales, kernel_size, num_orientations, num_scales),
            nn.BatchNorm2d(2 * num_orientations * num_scales), 
            nn.ReLU(inplace=True)
        )
    
    def doubleConv3x3(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class GaborConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, num_orientations, num_scales):
        super(GaborConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * num_orientations * num_scales # 80
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.padding = kernel_size // 2
        

        # Generate Gabor filter parameters
        self.sigma, self.theta, self.Lambda, self.psi, self.gamma, self.bias = self.generate_parameters(self.out_channels // 2) # 40
        self.filter_cos = self.whole_filter(True)
        self.filter_sin = self.whole_filter(False)

    def forward(self, x):
        x_cos = F.conv2d(x, self.filter_cos, padding = self.padding, bias=self.bias)
        x_sin = F.conv2d(x, self.filter_sin, padding = self.padding, bias=self.bias)
        return torch.cat((x_cos, x_sin), 1)

    def generate_parameters(self, dim_out):
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