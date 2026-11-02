#include "lane/unet.hpp"
#include "lane/dataset.hpp"
#include "lane/args.hpp"
#include "lane/cpu.hpp"
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
namespace fs=std::filesystem;
struct Result{double loss;lane::Counts counts;};
Result validation(lane::UNetSmall&net,const std::vector<lane::Sample>&samples,
                  int height,int width,int batch,torch::Device device){
 net->eval();torch::NoGradGuard guard;
 Result r{0,{}};size_t total=0;
 for(size_t start=0;start<samples.size();start+=batch){
  std::vector<torch::Tensor> xs,ys;
  for(size_t j=start;j<std::min(start+batch,samples.size());++j){
   auto pair=lane::load_sample(samples[j],height,width);
   xs.push_back(pair.image);ys.push_back(pair.mask);
  }
  auto x=torch::stack(xs).to(device),y=torch::stack(ys).to(device);
  auto logits=net->forward(x);
  const auto loss=lane::segmentation_loss(logits,y);
  r.loss+=loss.item<double>()*xs.size();total+=xs.size();
  lane::update_counts(r.counts,logits,y,.5);
 }
 r.loss/=total;return r;
}
int main(int argc,char**argv){
 try{
  lane::Args args(argc,argv);
  const fs::path root=args.get("--root","data/raw"),tr=args.get("--train","data/splits/train.csv"),va=args.get("--val","data/splits/val.csv");
  const fs::path out=args.get("--out","checkpoints/best.pt"),log=args.get("--log","reports/training.csv");
  const int h=args.number("--height",128),w=args.number("--width",224),batch=args.number("--batch",1),epochs=args.number("--epochs",15),seed=args.number("--seed",42);
  const double lr=args.real("--lr",0.001),flip_prob=args.real("--flip-prob",0.5);
  const double brightness=args.real("--brightness",0.0),contrast=args.real("--contrast",0.0);
  const int patience=args.number("--patience",0);
  if(h<32||w<32||batch<1||epochs<1||lr<=0||flip_prob<0||flip_prob>1||brightness<0||brightness>0.5||contrast<0||contrast>0.5||patience<0)throw std::runtime_error("Hyperparametres invalides");
  lane::configure_cpu_threads(args.number("--threads",2));
  auto training=lane::read_split(root,tr),val=lane::read_split(root,va);
  auto device=lane::select_device(args.get("--device","cpu"));
  std::cout<<"Train="<<training.size()<<" validation="<<val.size()<<" device="<<lane::device_description(device)<<'\n';
  torch::manual_seed(seed);std::mt19937 rng(seed);std::bernoulli_distribution flip(flip_prob);
  std::uniform_real_distribution<double> bright_dist(-brightness,brightness),contrast_dist(1.0-contrast,1.0+contrast);
  lane::UNetSmall net;net->to(device);
  torch::optim::Adam optimizer(net->parameters(),torch::optim::AdamOptions(lr));
  if(!out.parent_path().empty())fs::create_directories(out.parent_path());
  if(!log.parent_path().empty())fs::create_directories(log.parent_path());
  std::ofstream csv(log);if(!csv)throw std::runtime_error("Impossible d'ouvrir fichier log");
  csv<<"epoch,train_loss,val_loss,val_iou,val_dice,val_precision,val_recall\n";
  double best=-1;int best_epoch=0;
  std::vector<size_t> indices(training.size());std::iota(indices.begin(),indices.end(),0);
  int stale=0;
  for(int epoch=1;epoch<=epochs;++epoch){
   net->train();std::shuffle(indices.begin(),indices.end(),rng);
   double sum_loss=0;size_t processed=0;
   for(size_t start=0;start<indices.size();start+=batch){
    std::vector<torch::Tensor> xs,ys;
    for(size_t j=start;j<std::min(start+batch,indices.size());++j){
     auto pair=lane::load_sample(training[indices[j]],h,w,flip(rng));
     if(brightness>0||contrast>0){
      const double gain=contrast>0?contrast_dist(rng):1.0;
      const double bias=brightness>0?bright_dist(rng):0.0;
      pair.image= (pair.image*gain+bias).clamp(0.0,1.0);
     }
     xs.push_back(pair.image);ys.push_back(pair.mask);
    }
    auto x=torch::stack(xs).to(device),y=torch::stack(ys).to(device);
    optimizer.zero_grad();auto logits=net->forward(x);
    auto loss=lane::segmentation_loss(logits,y);
    if(!std::isfinite(loss.item<double>()))throw std::runtime_error("Loss NaN/Inf: verifier donnees et learning rate");
    loss.backward();optimizer.step();
    sum_loss+=loss.item<double>()*xs.size();processed+=xs.size();
   }
   auto metrics=validation(net,val,h,w,batch,device);
   auto viou=lane::iou(metrics.counts);
   csv<<epoch<<','<<sum_loss/processed<<','<<metrics.loss<<','<<viou<<','<<lane::dice(metrics.counts)<<','<<lane::precision(metrics.counts)<<','<<lane::recall(metrics.counts)<<'\n';csv.flush();
   std::cout<<"Epoch "<<epoch<<'/'<<epochs<<" train_loss="<<sum_loss/processed<<" val_loss="<<metrics.loss<<" IoU="<<viou<<" Dice="<<lane::dice(metrics.counts)<<'\n';
   if(viou>best){
    best=viou;best_epoch=epoch;
    torch::save(net,out.string());lane::save_resolution(out,h,w);
    std::cout<<"  -> Meilleur modele sauvegarde : "<<out<<'\n';
    stale=0;
   }else{
    ++stale;
   }
   if(patience>0&&stale>=patience){
    std::cout<<"Arret anticipe apres "<<stale<<" epochs sans amelioration.\n";
    break;
   }
  }
  std::cout<<"Meilleur epoch="<<best_epoch<<" IoU="<<best<<" checkpoint="<<out<<'\n';
 }catch(const std::exception&e){std::cerr<<"ERREUR: "<<e.what()<<'\n';return 1;}
}
