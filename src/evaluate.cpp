#include "lane/unet.hpp"
#include "lane/dataset.hpp"
#include "lane/args.hpp"
#include "lane/cpu.hpp"
#include <torch/torch.h>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
namespace fs=std::filesystem;
int main(int argc,char**argv){
 try{
  lane::Args a(argc,argv);
  fs::path root=a.get("--root","data/raw"),split=a.get("--split","data/splits/test.csv"),checkpoint=a.get("--model","checkpoints/best.pt");
  fs::path out=a.get("--out","reports/evaluation");
  const auto [h,w]=lane::read_resolution(checkpoint);
  const int previews=a.number("--previews",8);
  const double threshold=a.real("--threshold",.5);
  if(threshold<0||threshold>1||previews<0)throw std::runtime_error("Arguments invalides");
  lane::configure_cpu_threads(a.number("--threads",2));
  auto samples=lane::read_split(root,split);auto device=lane::select_device(a.get("--device","cpu"));
  lane::UNetSmall net;torch::load(net,checkpoint.string());net->to(device);net->eval();
  torch::NoGradGuard guard;lane::Counts counts;double sumloss=0;
  fs::create_directories(out);size_t n=0;
  for(const auto& s:samples){
   auto pair=lane::load_sample(s,h,w);
   auto y=pair.mask.unsqueeze(0).to(device),x=pair.image.unsqueeze(0).to(device);
   auto logits=net->forward(x);
   lane::update_counts(counts,logits,y,threshold);
   sumloss+=lane::segmentation_loss(logits,y).item<double>();
   if(static_cast<int>(n)<previews){
    cv::Mat original=cv::imread(s.image.string(),cv::IMREAD_COLOR);
    auto mask=lane::probability_to_mask(logits.sigmoid(),original.cols,original.rows,threshold);
    auto overlay=lane::overlay_mask(original,mask);
    cv::imwrite((out/("prediction_"+std::to_string(n)+".png")).string(),overlay);
   }
   ++n;
  }
  std::ofstream report(out/"metrics.txt");
  const auto show=[&](std::ostream&stream){
   stream<<"Images: "<<n<<"\nLoss: "<<sumloss/n<<"\nIoU: "<<lane::iou(counts)<<"\nDice: "<<lane::dice(counts)<<"\nPrecision: "<<lane::precision(counts)<<"\nRecall: "<<lane::recall(counts)<<"\nTP: "<<counts.tp<<" FP: "<<counts.fp<<" FN: "<<counts.fn<<" TN: "<<counts.tn<<"\n";
  };
  show(std::cout);show(report);
  std::cout<<"Rapport: "<<out/"metrics.txt"<<'\n';
 }catch(const std::exception&e){std::cerr<<"ERREUR: "<<e.what()<<'\n';return 1;}
}
