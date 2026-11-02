#include "lane/unet.hpp"
#include <torch/torch.h>
#include <torch/version.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(){
 try{
  std::cout<<"LibTorch C++ version "<<TORCH_VERSION<<'\n';
  at::set_num_threads(2);
  lane::UNetSmall model;
  model->eval();
  torch::NoGradGuard guard;
  auto x=torch::rand({1,3,128,224});
  auto y=model->forward(x);
  std::cout<<"Entree: "<<x.sizes()<<" | logits: "<<y.sizes()<<" | execution: CPU | threads: "<<at::get_num_threads()<<'\n';
  return (y.dim()==4 && y.size(0)==1 && y.size(1)==1 && y.size(2)==128 && y.size(3)==224)?0:1;
 }catch(const std::exception& e){std::cerr<<"ERREUR: "<<e.what()<<'\n';return 1;}
}
