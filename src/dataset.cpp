#include "lane/dataset.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <stdexcept>
#include <cmath>
namespace fs=std::filesystem;
namespace lane {
std::vector<Sample> read_split(const fs::path& root,const fs::path& csv){
 std::ifstream f(csv);
 if(!f)throw std::runtime_error("Impossible d'ouvrir le split: "+csv.string());
 std::string line; std::getline(f,line);
 if(line!="image_rel,mask_rel" && line!="image_rel,mask_rel\r")throw std::runtime_error("Entete CSV invalide: "+csv.string());
 std::vector<Sample> result;
 while(std::getline(f,line)){
  if(line.empty()||line=="\r")continue;
  if(!line.empty()&&line.back()=='\r')line.pop_back();
  const size_t comma=line.find(',');
  if(comma==std::string::npos||line.find(',',comma+1)!=std::string::npos)throw std::runtime_error("Ligne CSV invalide: "+line);
  fs::path im=root/line.substr(0,comma),ma=root/line.substr(comma+1);
  if(!fs::exists(im)||!fs::exists(ma))throw std::runtime_error("Fichier absent: "+line);
  result.push_back({im,ma});
 }
 if(result.empty())throw std::runtime_error("Split vide: "+csv.string());
 return result;
}
torch::Tensor image_to_tensor(const cv::Mat& bgr,int h,int w){
 if(bgr.empty()||bgr.type()!=CV_8UC3)throw std::runtime_error("Image BGR8 absente ou invalide");
 if(h<=0||w<=0)throw std::runtime_error("Taille image invalide");
 cv::Mat resized,rgb;
 cv::resize(bgr,resized,cv::Size(w,h),0,0,(h<bgr.rows&&w<bgr.cols)?cv::INTER_AREA:cv::INTER_LINEAR);
 cv::cvtColor(resized,rgb,cv::COLOR_BGR2RGB);
 return torch::from_blob(rgb.data,{h,w,3},torch::kUInt8).clone().permute({2,0,1}).to(torch::kFloat32).div_(255.0);
}
TensorPair load_sample(const Sample& s,int h,int w,bool flip){
 cv::Mat im=cv::imread(s.image.string(),cv::IMREAD_COLOR);
 cv::Mat ma=cv::imread(s.mask.string(),cv::IMREAD_UNCHANGED);
 if(im.empty()||ma.empty()||ma.type()!=CV_8UC1||im.size()!=ma.size())
  throw std::runtime_error("Image/masque illisible, format non 8-bit ou dimensions differentes: "+s.image.string());
 if(flip){cv::flip(im,im,1);cv::flip(ma,ma,1);}
 cv::Mat mask;
 cv::resize(ma,mask,cv::Size(w,h),0,0,cv::INTER_NEAREST);
 auto x=image_to_tensor(im,h,w);
 auto y=torch::from_blob(mask.data,{h,w},torch::kUInt8).clone().gt(0).to(torch::kFloat32).unsqueeze(0);
 return {x,y};
}
cv::Mat probability_to_mask(torch::Tensor probs,int original_width,int original_height,double threshold){
 auto cpu=probs.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
 if(cpu.dim()==4)cpu=cpu.squeeze(0);
 if(cpu.dim()==3)cpu=cpu.squeeze(0);
 if(cpu.dim()!=2)throw std::runtime_error("Probabilites attendues [H,W] ou [1,1,H,W]");
 cv::Mat p(static_cast<int>(cpu.size(0)),static_cast<int>(cpu.size(1)),CV_32F,cpu.data_ptr<float>());
 cv::Mat resized,mask;
 cv::resize(p,resized,cv::Size(original_width,original_height),0,0,cv::INTER_LINEAR);
 cv::threshold(resized,mask,threshold,255,cv::THRESH_BINARY);
 mask.convertTo(mask,CV_8U);
 return mask;
}
void update_counts(Counts& c,const torch::Tensor& logits,const torch::Tensor& truth,double threshold){
 auto pred=logits.sigmoid()>=threshold;
 auto gt=truth>0.5;
 c.tp+=torch::logical_and(pred,gt).sum().item<int64_t>();
 c.fp+=torch::logical_and(pred,torch::logical_not(gt)).sum().item<int64_t>();
 c.fn+=torch::logical_and(torch::logical_not(pred),gt).sum().item<int64_t>();
 c.tn+=torch::logical_and(torch::logical_not(pred),torch::logical_not(gt)).sum().item<int64_t>();
}
double precision(const Counts&c){return c.tp+c.fp==0?1.0:c.tp/(c.tp+c.fp);}
double recall(const Counts&c){return c.tp+c.fn==0?1.0:c.tp/(c.tp+c.fn);}
double iou(const Counts&c){return c.tp+c.fp+c.fn==0?1.0:c.tp/(c.tp+c.fp+c.fn);}
double dice(const Counts&c){return 2*c.tp+c.fp+c.fn==0?1.0:(2*c.tp)/(2*c.tp+c.fp+c.fn);}
torch::Tensor segmentation_loss(torch::Tensor logits,torch::Tensor truth){
 auto bce=torch::nn::functional::binary_cross_entropy_with_logits(logits,truth);
 auto prob=logits.sigmoid();
 auto intersection=(prob*truth).sum({1,2,3});
 auto denominator=prob.sum({1,2,3})+truth.sum({1,2,3});
 auto dice_loss=1-( (2*intersection+1)/(denominator+1) ).mean();
 return bce+dice_loss;
}
cv::Mat overlay_mask(const cv::Mat&bgr,const cv::Mat&mask){
 if(bgr.empty()||mask.empty()||bgr.size()!=mask.size()||mask.type()!=CV_8UC1)
  throw std::runtime_error("Dimensions superposition invalides");
 cv::Mat red=bgr.clone(),blended,out=bgr.clone();
 red.setTo(cv::Scalar(0,0,255),mask);
 cv::addWeighted(bgr,.65,red,.35,0,blended);
 blended.copyTo(out,mask);
 return out;
}
std::string device_description(const torch::Device& d){return d.is_cpu()?"CPU":"autre";}
torch::Device select_device(const std::string& requested){
 if(requested=="cpu"||requested=="auto")return torch::Device(torch::kCPU);
 throw std::runtime_error("Edition CPU uniquement: utiliser --device cpu (aucun CUDA requis)");
}
}
