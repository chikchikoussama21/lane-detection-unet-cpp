#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <map>
#include <array>
#include <string>
#include <algorithm>
#include <cctype>
#include <stdexcept>
namespace fs=std::filesystem;
using Files=std::map<std::string,fs::path>;
Files scan(fs::path root){
 if(!fs::exists(root))throw std::runtime_error("Dossier absent: "+root.string());
 Files out;
 for(const auto& item:fs::recursive_directory_iterator(root)){
  if(!item.is_regular_file())continue;
  auto ext=item.path().extension().string();std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
  if(ext!=".jpg"&&ext!=".jpeg"&&ext!=".png"&&ext!=".bmp")continue;
  fs::path rel=fs::relative(item.path(),root);rel.replace_extension();
  if(!out.emplace(rel.generic_string(),item.path()).second)throw std::runtime_error("Cle dupliquee: "+rel.string());
 }
 return out;
}
int main(int argc,char**argv){
 try{
  if(argc!=4){std::cerr<<"Usage: audit_dataset RACINE_IMAGES RACINE_MASQUES DOSSIER_RAPPORTS\n";return 2;}
  fs::path imgs=argv[1],masks=argv[2],reports=argv[3];auto I=scan(imgs),M=scan(masks);
  fs::create_directories(reports/"previews");
  std::ofstream csv(reports/"audit.csv");csv<<"cle;etat;largeur;hauteur;valeurs_masque\n";
  size_t good=0,notfound=0,invalid=0,extra=0;std::array<std::uint64_t,256> hist{};
  for(const auto& [k,ip]:I){
   const auto it=M.find(k);
   if(it==M.end()){++notfound;csv<<k<<";MASQUE_MANQUANT;;;\n";continue;}
   cv::Mat im=cv::imread(ip.string(),cv::IMREAD_COLOR),mask=cv::imread(it->second.string(),cv::IMREAD_UNCHANGED);
   if(im.empty()||mask.empty()||im.size()!=mask.size()||mask.type()!=CV_8UC1){
    ++invalid;csv<<k<<";ERREUR_FORMAT_OU_DIMENSIONS;;;\n";continue;
   }
   std::array<std::uint64_t,256> local{};
   for(int y=0;y<mask.rows;++y){const auto* row=mask.ptr<uchar>(y);for(int x=0;x<mask.cols;++x){local[row[x]]++;hist[row[x]]++;}}
   std::string unique;
   for(size_t v=0;v<local.size();++v)if(local[v])unique+=(unique.empty()?"":",")+std::to_string(v);
   csv<<k<<";OK;"<<im.cols<<';'<<im.rows<<';'<<unique<<'\n';
   if(good<8){
    cv::Mat overlay=im.clone(), m=(mask>0), red=im.clone();red.setTo(cv::Scalar(0,0,255),m);
    cv::Mat mix;cv::addWeighted(im,.65,red,.35,0,mix);mix.copyTo(overlay,m);
    cv::imwrite((reports/"previews"/("preview_"+std::to_string(good)+".jpg")).string(),overlay);
   }
   ++good;
  }
  for(const auto& [k,p]:M)if(!I.count(k)){++extra;csv<<k<<";IMAGE_MANQUANTE;;;\n";}
  std::cout<<"Images="<<I.size()<<" masques="<<M.size()<<" couples_OK="<<good<<" masques_manquants="<<notfound<<" invalides="<<invalid<<" images_manquantes="<<extra<<'\n';
  std::cout<<"Valeurs pixels des masques OK: ";for(size_t k=0;k<256;++k)if(hist[k])std::cout<<k<<"("<<hist[k]<<") ";std::cout<<'\n';
  std::cout<<"Rapport: "<<(reports/"audit.csv")<<'\n';
  return(notfound||invalid||extra)?1:0;
 }catch(const std::exception& e){std::cerr<<"ERREUR: "<<e.what()<<'\n';return 1;}
}
