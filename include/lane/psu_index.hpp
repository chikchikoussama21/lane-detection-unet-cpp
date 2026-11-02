#pragma once
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace lane {
namespace fs = std::filesystem;
struct PsuRecord { int id=0; fs::path image, label; };
struct Polygon { int cls=0; std::vector<double> xy; };
inline int psu_frame_id(const fs::path& p) {
 const std::string stem=p.stem().string();
 if(stem.rfind("frame_",0)!=0 || stem.size()==6) return -1;
 for(size_t i=6;i<stem.size();++i)if(!std::isdigit(static_cast<unsigned char>(stem[i]))) return -1;
 try { const long long n=std::stoll(stem.substr(6));return n>0&&n<=1000000?static_cast<int>(n):-1; }
 catch(...) {return -1;}
}
inline std::string lower_ext(fs::path p){
 std::string s=p.extension().string();
 std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});return s;
}
inline std::vector<PsuRecord> discover_psu(const fs::path& extracted) {
 if(!fs::is_directory(extracted)) throw std::runtime_error("Repertoire absent: "+extracted.string());
 std::map<int,fs::path> images,labels;
 for(auto const& entry:fs::recursive_directory_iterator(extracted)){
  if(!entry.is_regular_file())continue;
  const auto& file=entry.path();const int id=psu_frame_id(file);if(id<0)continue;
  const auto ext=lower_ext(file);
  if(ext==".jpg"||ext==".jpeg"||ext==".png"){
   if(!images.emplace(id,file).second)throw std::runtime_error("Images dupliquees pour frame_"+std::to_string(id)+": "+file.string());
  }else if(ext==".txt"){
   if(!labels.emplace(id,file).second)throw std::runtime_error("Annotations dupliquees pour frame_"+std::to_string(id)+": "+file.string());
  }
 }
 if(images.empty())throw std::runtime_error("Aucune frame_XXXX.jpg/png trouvee sous "+extracted.string()+" : consulter unzip -l");
 std::vector<PsuRecord> result;
 for(const auto& kv:images){
  auto it=labels.find(kv.first);
  if(it==labels.end())throw std::runtime_error("Annotation absente pour "+kv.second.string());
  result.push_back({kv.first,kv.second,it->second});
 }
 for(const auto& kv:labels)if(!images.count(kv.first))
   throw std::runtime_error("Image absente pour l'annotation "+kv.second.string());
 return result;
}
inline std::vector<Polygon> parse_psu_polygons(const fs::path& label) {
 std::ifstream file(label);if(!file)throw std::runtime_error("Fichier TXT illisible: "+label.string());
 std::vector<Polygon> polys; std::string line;int line_no=0;
 while(std::getline(file,line)){
  ++line_no;if(line.find_first_not_of(" \t\r\n")==std::string::npos)continue;
  std::istringstream ss(line);std::vector<double> vals;double v;
  while(ss>>v)vals.push_back(v);
  if(!ss.eof())throw std::runtime_error("Valeur non numerique dans "+label.string()+" ligne "+std::to_string(line_no));
  if(vals.size()<7 || vals.size()%2!=1)throw std::runtime_error("Polygone invalide (attendu: classe x1 y1 x2 y2 x3 y3...): "+label.string()+" ligne "+std::to_string(line_no));
  if(!std::isfinite(vals[0]) || std::floor(vals[0])!=vals[0] || vals[0]<0 || vals[0]>4)
   throw std::runtime_error("Classe inconnue (attendu 0..4): "+label.string()+" ligne "+std::to_string(line_no));
  Polygon poly; poly.cls=static_cast<int>(vals[0]);
  for(size_t i=1;i<vals.size();++i){
   if(!std::isfinite(vals[i]) || vals[i]<0 || vals[i]>1)
    throw std::runtime_error("Coordonnee hors [0,1] : "+label.string()+" ligne "+std::to_string(line_no));
   poly.xy.push_back(vals[i]);
  }
  polys.push_back(std::move(poly));
 }
 return polys;
}
inline std::string psu_split(int id){
 if(id>=1 && id<=550)return "train";
 if(id>=551 && id<=650)return "val";
 if(id>=651 && id<=1000)return "test";
 return "outside";
}
} // namespace lane
