#pragma once
#include <map>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <fstream>
namespace lane {
class Args {
 std::map<std::string,std::string> options;
public:
 Args(int argc,char**argv){
  for(int i=1;i<argc;++i){
   const std::string key=argv[i];
   if(key=="--help"||key=="-h")throw std::runtime_error("Consulter README.md pour la syntaxe complete.");
   if(key.rfind("--",0)!=0||i+1>=argc)throw std::runtime_error("Argument attendu: --nom valeur");
   if(!options.emplace(key,argv[++i]).second)throw std::runtime_error("Argument duplique: "+key);
  }
 }
 std::string get(const std::string&k,const std::string&d)const{
  auto it=options.find(k);return it==options.end()?d:it->second;
 }
 int number(const std::string&k,int d)const{return std::stoi(get(k,std::to_string(d)));}
 double real(const std::string&k,double d)const{return std::stod(get(k,std::to_string(d)));}
};
inline void save_resolution(const std::filesystem::path& checkpoint,int height,int width){
 std::ofstream f(checkpoint.string()+".cfg");
 if(!f)throw std::runtime_error("Echec sauvegarde configuration");
 f<<height<<' '<<width<<'\n';
}
inline std::pair<int,int> read_resolution(const std::filesystem::path& checkpoint){
 std::ifstream f(checkpoint.string()+".cfg");
 int h=0,w=0;if(!(f>>h>>w)||h<=0||w<=0)
  throw std::runtime_error("Fichier .cfg de resolution absent/invalide pour le modele: "+checkpoint.string());
 return {h,w};
}
}
