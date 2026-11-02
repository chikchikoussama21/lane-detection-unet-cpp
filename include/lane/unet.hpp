#pragma once
#include <torch/torch.h>
namespace lane {
struct DoubleConvImpl : torch::nn::Module {
 torch::nn::Sequential conv{nullptr};
 DoubleConvImpl(int in_channels,int out_channels);
 torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DoubleConv);

struct UNetSmallImpl : torch::nn::Module {
 DoubleConv enc1{nullptr},enc2{nullptr},bottleneck{nullptr},dec2{nullptr},dec1{nullptr};
 torch::nn::Conv2d head{nullptr};
 UNetSmallImpl();
 torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(UNetSmall);
}
