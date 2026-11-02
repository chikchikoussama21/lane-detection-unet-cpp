#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cmath>
namespace fs=std::filesystem;
int main(int argc,char**argv){
 try{
  if(argc!=2){std::cerr<<"Usage: make_smoke_psu DOSSIER_SORTIE\n";return 2;}
  fs::path root=argv[1];fs::create_directories(root/"images");fs::create_directories(root/"labels");
  std::vector<int> ids;for(int i=1;i<=16;++i)ids.push_back(i);
  for(int i=551;i<=554;++i)ids.push_back(i);
  for(int i=651;i<=654;++i)ids.push_back(i);
  cv::VideoWriter writer((root/"video_smoke.avi").string(),cv::VideoWriter::fourcc('M','J','P','G'),10,cv::Size(320,180));
  for(int id:ids){
   cv::Mat img(180,320,CV_8UC3,cv::Scalar(75,75,75));
   cv::rectangle(img,cv::Rect(0,0,320,70),cv::Scalar(180,125,85),-1);
   cv::line(img,{120,179},{152,85},cv::Scalar(230,230,230),5);
   cv::line(img,{205,179},{171,85},cv::Scalar(240,240,240),5);
   cv::Mat jitter=img.clone();cv::putText(jitter,"SMOKE DATA - NOT REAL",{18,22},cv::FONT_HERSHEY_SIMPLEX,.50,{0,0,0},1);
   std::ostringstream stem;stem<<"frame_"<<std::setw(4)<<std::setfill('0')<<id;
   cv::imwrite((root/"images"/(stem.str()+".jpg")).string(),jitter);
   std::ofstream label(root/"labels"/(stem.str()+".txt"));
   label<<"0 0.357 0.995 0.395 0.995 0.488 0.472 0.463 0.472\n";
   label<<"0 0.629 0.995 0.655 0.995 0.545 0.472 0.529 0.472\n";
   if(writer.isOpened())writer.write(jitter);
  }
  std::cout<<"Donnees artificielles creees: 16 train, 4 val, 4 test; ceci ne valide PAS les performances routieres.\n";
 }catch(const std::exception& e){std::cerr<<"ERREUR: "<<e.what()<<'\n';return 1;}
}
