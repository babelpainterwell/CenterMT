import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaborSimpleModel(nn.Module):
    def __init__(self, kernel_size, in_channels, num_orientations, num_scales, output_channels):
        super(GaborSimpleModel, self).__init__()
        self.in_channels = in_channels
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.output_channels = output_channels  
        
        # First Gabor Filter Layer, using padding to avoid downsampling?
        self.sigma1, self.theta1, self.Lambda1, self.psi1, self.gamma1, self.bias1 = self.generate_parameters(num_orientations*num_scales, in_channels)
        self.filter_cos1 = self.whole_filter(True, self.sigma1, self.theta1, self.Lambda1, self.psi1, self.gamma1)
        self.filter_sin1 = self.whole_filter(False, self.sigma1, self.theta1, self.Lambda1, self.psi1, self.gamma1)

        # Second Gabor Filter Layer
        self.sigma2, self.theta2, self.Lambda2, self.psi2, self.gamma2, self.bias2 = self.generate_parameters(num_orientations*num_scales, num_orientations*num_scales*2)
        self.filter_cos2 = self.whole_filter(True, self.sigma2, self.theta2, self.Lambda2, self.psi2, self.gamma2)
        self.filter_sin2 = self.whole_filter(False, self.sigma2, self.theta2, self.Lambda2, self.psi2, self.gamma2)
        
        # Adjusted Final Convolutional Layer to match input image channels
        self.final_conv = nn.Conv2d(num_orientations*num_scales*2*2, output_channels, kernel_size=1)

    def forward(self, x):
        # Input shape: [batch_size, in_channels, height, width] - [64, 1, 65, 65]

        # First Gabor Filter Layer
        x_cos1 = F.conv2d(x, self.filter_cos1, bias=self.bias1)
        x_sin1 = F.conv2d(x, self.filter_sin1, bias=self.bias1)
        x_comb1 = torch.cat((x_cos1, x_sin1), 1) # shape: [64, num_orientations*num_scales*2, (65 - k + 1), (65 - k + 1)]

        # Second Gabor Filter Layer
        x_cos2 = F.conv2d(x_comb1, self.filter_cos2, bias=self.bias2)
        x_sin2 = F.conv2d(x_comb1, self.filter_sin2, bias=self.bias2)
        x_comb2 = torch.cat((x_cos2, x_sin2), 1) # shape: [64, num_orientations*num_scales*2*2, (65 - k + 1) - k + 1, (65 - k + 1) - k + 1]
        
        # Final Convolutional Layer
        output = self.final_conv(x_comb2)  # This layer ensures the output has the same spatial dimensions as the input

        # Apply sigmoid activation to ensure output values are in the range [0, 1]
        output = torch.sigmoid(output)

        # upsampling??
        
        return output 

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