#include "lane/psu_index.hpp"
#include "lane/args.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <cmath>
namespace fs=std::filesystem;

// Masque binaire : seules les classes 1, 2 et 3 sont conservees.
// Le sens semantique des classes doit etre valide visuellement.
static cv::Mat polygon_mask(const std::vector<lane::Polygon>& polys,int width,int height){
 cv::Mat mask(height,width,CV_8UC1,cv::Scalar(0));
 for(const auto& poly:polys){
  if(poly.cls != 1 && poly.cls != 2 && poly.cls != 3)continue;
  std::vector<cv::Point> vertices;
  for(size_t i=0;i<poly.xy.size();i+=2){
   int x=std::clamp(static_cast<int>(std::lround(poly.xy[i]*(width-1))),0,width-1);
   int y=std::clamp(static_cast<int>(std::lround(poly.xy[i+1]*(height-1))),0,height-1);
   vertices.emplace_back(x,y);
  }
  cv::fillPoly(mask,std::vector<std::vector<cv::Point>>{vertices},cv::Scalar(255),cv::LINE_8);
 }
 return mask;
}
int main(int argc,char**argv){
 try{
  lane::Args args(argc,argv);
  const fs::path src=args.get("--src","data/external/PSU/extracted");
  const fs::path out=args.get("--out","data/processed");
  const fs::path splits=args.get("--splits","data/splits");
  const int width=args.number("--width",256),height=args.number("--height",144);
  const int train_limit=args.number("--train-limit",550),val_limit=args.number("--val-limit",100),test_limit=args.number("--test-limit",350);
  const int preview_limit=args.number("--preview-each",3);
  if(width<64||height<64||width%4||height%4||train_limit<1||val_limit<1||test_limit<1||preview_limit<0)
   throw std::runtime_error("Taille >=64 multiple de 4 et limites positives exigees");
  const auto records=lane::discover_psu(src);
  fs::create_directories(out);fs::create_directories(splits);
  for(const char* s:{"train.csv","val.csv","test.csv"})
   if(fs::exists(splits/s))throw std::runtime_error("Fichier split deja present: "+(splits/s).string()+". Utiliser un autre dossier --splits.");
  std::map<std::string,std::vector<std::pair<fs::path,fs::path>>> manifests;
  std::map<std::string,int> count, empty_masks;
  std::map<std::string,int> previews;
  const fs::path preview_dir=out.parent_path()/(out.filename().string()+"_previews");
  fs::create_directories(preview_dir);
  for(const auto& r:records){
   const auto split=lane::psu_split(r.id);
   if(split=="outside")continue;
   const int limit=split=="train"?train_limit:split=="val"?val_limit:test_limit;
   if(count[split]>=limit)continue;
   const auto polys=lane::parse_psu_polygons(r.label);
   cv::Mat image=cv::imread(r.image.string(),cv::IMREAD_COLOR);
   if(image.empty())throw std::runtime_error("Lecture image impossible: "+r.image.string());
   cv::Mat resized;
   cv::resize(image,resized,cv::Size(width,height),0,0,cv::INTER_AREA);
   cv::Mat mask=polygon_mask(polys,width,height);
   if(cv::countNonZero(mask)==0)++empty_masks[split];
   fs::path img_rel=fs::path("images")/split/(r.image.stem().string()+".jpg");
   fs::path msk_rel=fs::path("masks")/split/(r.image.stem().string()+".png");
   fs::create_directories((out/img_rel).parent_path());
   fs::create_directories((out/msk_rel).parent_path());
   if(!cv::imwrite((out/img_rel).string(),resized,{cv::IMWRITE_JPEG_QUALITY,93}) ||
      !cv::imwrite((out/msk_rel).string(),mask))
      throw std::runtime_error("Echec ecriture pour image "+r.image.string());
   manifests[split].push_back({img_rel,msk_rel});++count[split];
   if(previews[split]<preview_limit){
    cv::Mat red=resized.clone(),mix=resized.clone();
    red.setTo(cv::Scalar(0,0,255),mask);
    cv::addWeighted(resized,0.65,red,0.35,0,mix);
    cv::imwrite((preview_dir/(split+"_"+r.image.stem().string()+".jpg")).string(),mix);
    ++previews[split];
   }
   if((count[split]%100)==0)std::cout<<"Prepare "<<split<<" : "<<count[split]<<" images\n";
  }
  for(const auto& split:{"train","val","test"})if(manifests[split].empty())
    throw std::runtime_error("Partition vide: "+std::string(split));
  for(const auto& split:{"train","val","test"}){
   std::ofstream csv(splits/(std::string(split)+".csv"));
   if(!csv)throw std::runtime_error("Impossible d'ecrire le CSV de "+std::string(split));
   csv<<"image_rel,mask_rel\n";
   for(const auto& paths:manifests[split])csv<<paths.first.generic_string()<<","<<paths.second.generic_string()<<"\n";
   std::cout<<split<<": "<<manifests[split].size()<<" couples ; masques vides="<<empty_masks[split]<<"\n";
  }
  std::cout<<"OK. Images+masques: "<<out<<" | CSV: "<<splits<<" | Apercus: "<<preview_dir<<"\n";
  std::cout<<"IMPORTANT: classes 1,2,3 retenues ; verifier visuellement les masques avant entrainement.\n";
 }catch(const std::exception& e){std::cerr<<"ERREUR: "<<e.what()<<'\n';return 1;}
}
