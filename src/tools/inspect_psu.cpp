#include "lane/psu_index.hpp"
#include <iostream>
int main(int argc,char**argv){
 try{
  if(argc!=2){std::cerr<<"Usage: ./build/inspect_psu data/external/PSU/extracted\n";return 2;}
  const auto records=lane::discover_psu(argv[1]);
  size_t t=0,v=0,e=0,other=0,empty=0,polys=0;
  for(const auto& r:records){
   auto ps=lane::parse_psu_polygons(r.label);polys+=ps.size();if(ps.empty())++empty;
   auto split=lane::psu_split(r.id);
   if(split=="train")++t;else if(split=="val")++v;else if(split=="test")++e;else ++other;
  }
  std::cout<<"Couples corrects: "<<records.size()<<" | train="<<t<<" val="<<v<<" test="<<e
    <<" hors-plage="<<other<<" | polygones="<<polys<<" | annotations vides="<<empty<<"\n";
  std::cout<<"Exemple image: "<<records.front().image<<"\nAnnotation: "<<records.front().label<<"\n";
  if(t==0||v==0||e==0)throw std::runtime_error("Au moins une des partitions est vide. Verifier le contenu extrait.");
  if(records.size()!=1000)std::cout<<"ATTENTION: 1000 images annoncees, "<<records.size()<<" trouvees. Archive incomplete ou differente ?\n";
  std::cout<<"TEST D'INDEX OK : aucune conversion image n'est encore faite.\n";
 }catch(const std::exception&e){std::cerr<<"ERREUR: "<<e.what()<<'\n';return 1;}
}
