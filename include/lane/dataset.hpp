#pragma once
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>
#include <vector>
namespace lane {
struct Sample {
 std::filesystem::path image;
 std::filesystem::path mask;
};
struct TensorPair {
 torch::Tensor image; // RGB float32 [3,H,W], [0,1]
 torch::Tensor mask;  // float32 [1,H,W], {0,1}
};
std::vector<Sample> read_split(const std::filesystem::path& root,
                               const std::filesystem::path& csv);
TensorPair load_sample(const Sample& s,int h,int w,bool flip=false);
torch::Tensor image_to_tensor(const cv::Mat& bgr,int h,int w);
cv::Mat probability_to_mask(torch::Tensor probs,int original_width,int original_height,double threshold);
struct Counts { double tp=0,fp=0,fn=0,tn=0; };
void update_counts(Counts& c,const torch::Tensor& logits,const torch::Tensor& truth,double threshold);
double precision(const Counts& c);
double recall(const Counts& c);
double iou(const Counts& c);
double dice(const Counts& c);
torch::Tensor segmentation_loss(torch::Tensor logits,torch::Tensor truth);
cv::Mat overlay_mask(const cv::Mat& bgr,const cv::Mat& binary_mask);
std::string device_description(const torch::Device& d);
torch::Device select_device(const std::string& requested);
}
