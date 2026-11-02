#include "lane/unet.hpp"
#include <torch/nn/functional.h>
#include <vector>
namespace lane {
DoubleConvImpl::DoubleConvImpl(int in_channels,int out_channels){
 conv=register_module("conv",torch::nn::Sequential(
  torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,out_channels,3).padding(1)),
  torch::nn::BatchNorm2d(out_channels),
  torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
  torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels,out_channels,3).padding(1)),
  torch::nn::BatchNorm2d(out_channels),
  torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))));
}
torch::Tensor DoubleConvImpl::forward(torch::Tensor x){return conv->forward(x);}
UNetSmallImpl::UNetSmallImpl(){
 enc1=register_module("enc1",DoubleConv(3,8));
 enc2=register_module("enc2",DoubleConv(8,16));
 bottleneck=register_module("bottleneck",DoubleConv(16,32));
 dec2=register_module("dec2",DoubleConv(32+16,16));
 dec1=register_module("dec1",DoubleConv(16+8,8));
 head=register_module("head",torch::nn::Conv2d(torch::nn::Conv2dOptions(8,1,1)));
}
torch::Tensor UNetSmallImpl::forward(torch::Tensor x){
 auto e1=enc1->forward(x);
 auto e2=enc2->forward(torch::max_pool2d(e1,2));
 auto b=bottleneck->forward(torch::max_pool2d(e2,2));
 auto up2=torch::nn::functional::interpolate(b,torch::nn::functional::InterpolateFuncOptions()
   .size(std::vector<int64_t>{e2.size(2),e2.size(3)}).mode(torch::kBilinear).align_corners(false));
 auto d2=dec2->forward(torch::cat({up2,e2},1));
 auto up1=torch::nn::functional::interpolate(d2,torch::nn::functional::InterpolateFuncOptions()
   .size(std::vector<int64_t>{e1.size(2),e1.size(3)}).mode(torch::kBilinear).align_corners(false));
 return head->forward(dec1->forward(torch::cat({up1,e1},1)));
}
}
