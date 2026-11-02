#include "lane/unet.hpp"
#include "lane/dataset.hpp"
#include "lane/args.hpp"
#include "lane/cpu.hpp"
#include <torch/torch.h>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
namespace fs=std::filesystem;

static cv::VideoWriter open_writer(const fs::path& path, double fps, cv::Size size) {
 if(!path.parent_path().empty())fs::create_directories(path.parent_path());
 const bool mp4=path.extension()==".mp4";
 const int fourcc=mp4?cv::VideoWriter::fourcc('m','p','4','v'):cv::VideoWriter::fourcc('M','J','P','G');
 cv::VideoWriter writer(path.string(),fourcc,fps,size,true);
 if(!writer.isOpened())throw std::runtime_error("Impossible de creer la video: "+path.string());
 return writer;
}
static cv::Mat filter_small_components(const cv::Mat& mask,int min_area){
 if(min_area<=0)return mask;
 cv::Mat labels,stats,centroids;
 const int count=cv::connectedComponentsWithStats(mask,labels,stats,centroids,8,CV_32S);
 cv::Mat filtered=cv::Mat::zeros(mask.size(),CV_8UC1);
 for(int id=1;id<count;++id){
  if(stats.at<int>(id,cv::CC_STAT_AREA)>=min_area)filtered.setTo(255,labels==id);
 }
 return filtered;
}
static cv::Mat apply_road_roi(const cv::Mat& mask, double top_y, double top_left,
                              double top_right, double bottom_left, double bottom_right) {
 const int W=mask.cols,H=mask.rows;
 auto px=[W](double x){return cv::saturate_cast<int>(x*(W-1));};
 auto py=[H](double y){return cv::saturate_cast<int>(y*(H-1));};
 std::vector<cv::Point> polygon={
  {px(top_left),py(top_y)}, {px(top_right),py(top_y)},
  {px(bottom_right),H-1}, {px(bottom_left),H-1}
 };
 cv::Mat region=cv::Mat::zeros(mask.size(),CV_8UC1),result;
 cv::fillConvexPoly(region,polygon,cv::Scalar(255));
 cv::bitwise_and(mask,region,result);
 return result;
}
static cv::Mat bridge_vertical_gaps(const cv::Mat& mask,int gap){
 if(gap<=0)return mask;
 // Fermeture morphologique verticale uniquement, sans dilatation horizontale.
 cv::Mat result;
 cv::Mat kernel=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(1,gap));
 cv::morphologyEx(mask,result,cv::MORPH_CLOSE,kernel);
 return result;
}
static cv::Mat green_overlay(const cv::Mat& frame,const cv::Mat& mask){
 cv::Mat color=frame.clone(),result=frame.clone(),blended;
 color.setTo(cv::Scalar(0,255,0),mask);
 cv::addWeighted(frame,0.35,color,0.65,0,blended);
 blended.copyTo(result,mask);
 return result;
}
// Hough probabiliste : segments candidats, non estimation certifiee de voie.
static std::vector<cv::Vec4i> lane_segments(const cv::Mat& mask,
    int votes, int min_length, int max_gap, double min_angle_deg,
    int min_y, int max_segments) {
 cv::Mat edges;
 cv::Canny(mask,edges,50,150);
 std::vector<cv::Vec4i> raw, kept;
 cv::HoughLinesP(edges,raw,1.0,CV_PI/180.0,votes,min_length,max_gap);
 for(const auto& seg:raw){
  const double dx=static_cast<double>(seg[2]-seg[0]);
  const double dy=static_cast<double>(seg[3]-seg[1]);
  const double angle=std::atan2(std::abs(dy),std::abs(dx))*180.0/CV_PI;
  const double mid_y=(seg[1]+seg[3])/2.0;
  if(angle<min_angle_deg||mid_y<min_y)continue;
  kept.push_back(seg);
 }
 std::sort(kept.begin(),kept.end(),[](const auto& a,const auto& b){
  return cv::norm(cv::Point(a[2]-a[0],a[3]-a[1]))>
         cv::norm(cv::Point(b[2]-b[0],b[3]-b[1]));
 });
 if(static_cast<int>(kept.size())>max_segments)kept.resize(max_segments);
 return kept;
}
static cv::Mat draw_geometry(const cv::Mat& frame,
                             const std::vector<cv::Vec4i>& segments) {
 cv::Mat result=frame.clone();
 for(const auto& s:segments){
  const double dx=s[2]-s[0],dy=s[3]-s[1];
  const cv::Scalar color=dx*dy>=0?cv::Scalar(0,255,255):cv::Scalar(255,255,0);
  cv::line(result,{s[0],s[1]},{s[2],s[3]},color,3,cv::LINE_AA);
 }
 return result;
}
int main(int argc,char**argv){
 try{
  lane::Args a(argc,argv);
  const fs::path input=a.get("--video","data/demo_input.avi");
  const fs::path checkpoint=a.get("--model","checkpoints/best_v2.pt");
  const fs::path output=a.get("--out","reports/videos/v3_overlay.avi");
  const fs::path mask_output=a.get("--mask-out","reports/videos/v3_mask.avi");
  const std::string mode=a.get("--mode","both");
  const double threshold=a.real("--threshold",0.5);
  const int min_area=a.number("--min-area",0);
  const int close_vertical=a.number("--close-vertical",0);
  const bool display=a.number("--display",0)!=0;
  const bool hough=a.number("--hough",0)!=0;
  const fs::path geometry_output=a.get("--geometry-out","");
  const int hough_votes=a.number("--hough-votes",25);
  const int hough_min_length=a.number("--hough-min-length",25);
  const int hough_max_gap=a.number("--hough-max-gap",15);
  const double hough_min_angle=a.real("--hough-min-angle",25.0);
  const double hough_min_y=a.real("--hough-min-y",0.55);
  const int hough_max_segments=a.number("--hough-max-segments",30);
  const int roi=a.number("--roi",0);
  const double roi_top_y=a.real("--roi-top-y",0.55);
  const double roi_top_left=a.real("--roi-top-left",0.42);
  const double roi_top_right=a.real("--roi-top-right",0.58);
  const double roi_bottom_left=a.real("--roi-bottom-left",0.08);
  const double roi_bottom_right=a.real("--roi-bottom-right",0.92);
  if(!std::isfinite(threshold)||threshold<0||threshold>1||min_area<0||close_vertical<0||close_vertical>31||(close_vertical>0&&close_vertical%2==0))throw std::runtime_error("Seuil ou aire minimale invalide");
  auto valid_fraction=[](double v){return std::isfinite(v)&&v>=0.0&&v<=1.0;};
  if((roi!=0&&roi!=1)||!valid_fraction(roi_top_y)||
     !valid_fraction(roi_top_left)||!valid_fraction(roi_top_right)||
     !valid_fraction(roi_bottom_left)||!valid_fraction(roi_bottom_right)||
     roi_top_y>=1.0||roi_top_left>=roi_top_right||
     roi_bottom_left>=roi_bottom_right)
   throw std::runtime_error("Parametres ROI invalides: fractions dans [0,1], gauche < droite, top-y < 1");
  if(hough_votes<1||hough_min_length<1||hough_max_gap<0||
     !std::isfinite(hough_min_angle)||hough_min_angle<0||hough_min_angle>90||
     !std::isfinite(hough_min_y)||hough_min_y<0||hough_min_y>1||
     hough_max_segments<1||hough_max_segments>1000)
   throw std::runtime_error("Parametres Hough invalides");
  if(!geometry_output.empty()&&!hough)
   throw std::runtime_error("--geometry-out exige --hough 1");
  if(!geometry_output.empty()&&
     (fs::absolute(geometry_output).lexically_normal()==fs::absolute(output).lexically_normal()||
      (mode=="both"&&fs::absolute(geometry_output).lexically_normal()==fs::absolute(mask_output).lexically_normal())))
   throw std::runtime_error("La sortie geometrie doit etre distincte des autres videos");
  if(mode!="overlay"&&mode!="mask"&&mode!="both")throw std::runtime_error("--mode: overlay, mask ou both");
  if(mode=="both"&&fs::absolute(output).lexically_normal()==fs::absolute(mask_output).lexically_normal())
   throw std::runtime_error("--out et --mask-out doivent etre differents");
  lane::configure_cpu_threads(a.number("--threads",2));
  const auto [h,w]=lane::read_resolution(checkpoint);
  auto device=lane::select_device(a.get("--device","cpu"));
  lane::UNetSmall net;torch::load(net,checkpoint.string());net->to(device);net->eval();
  cv::VideoCapture video(input.string());
  if(!video.isOpened())throw std::runtime_error("Video introuvable ou format non supporte: "+input.string());
  const int W=static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
  const int H=static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps=video.get(cv::CAP_PROP_FPS);if(fps<=0||fps>240)fps=25;
  if(W<=0||H<=0)throw std::runtime_error("Dimensions video invalides");
  cv::VideoWriter overlay_writer,mask_writer,geometry_writer;
  if(mode=="overlay"||mode=="both")overlay_writer=open_writer(output,fps,cv::Size(W,H));
  if(mode=="mask"||mode=="both")mask_writer=open_writer(mode=="mask"?output:mask_output,fps,cv::Size(W,H));
  if(!geometry_output.empty())geometry_writer=open_writer(geometry_output,fps,cv::Size(W,H));
  torch::NoGradGuard guard;
  cv::Mat frame;std::size_t frames=0;double total_ms=0;
  while(video.read(frame)){
   auto start=std::chrono::steady_clock::now();
   auto x=lane::image_to_tensor(frame,h,w).unsqueeze(0).to(device);
   auto probs=net->forward(x).sigmoid();
   auto mask=lane::probability_to_mask(probs,frame.cols,frame.rows,threshold);
   if(roi)mask=apply_road_roi(mask,roi_top_y,roi_top_left,roi_top_right,
                               roi_bottom_left,roi_bottom_right);
   mask=bridge_vertical_gaps(mask,close_vertical);
   mask=filter_small_components(mask,min_area);
   cv::Mat geometry;
   if(hough){
    const auto segments=lane_segments(mask,hough_votes,hough_min_length,
      hough_max_gap,hough_min_angle,cvRound(hough_min_y*(frame.rows-1)),hough_max_segments);
    if(geometry_writer.isOpened())geometry=draw_geometry(frame,segments);
   }
   cv::Mat mask_bgr,overlay;
   if(mode!="overlay")cv::cvtColor(mask,mask_bgr,cv::COLOR_GRAY2BGR);
   if(mode!="mask")overlay=green_overlay(frame,mask);
   auto finish=std::chrono::steady_clock::now();
   total_ms+=std::chrono::duration<double,std::milli>(finish-start).count();
   if(geometry_writer.isOpened())geometry_writer.write(geometry);
   if(mode!="mask")overlay_writer.write(overlay);
   if(mode!="overlay")mask_writer.write(mask_bgr);
   ++frames;
   if(display){cv::imshow("Lane CNN V3",mode=="mask"?mask_bgr:overlay);if((cv::waitKey(1)&255)==27)break;}
  }
  if(overlay_writer.isOpened())overlay_writer.release();
  if(mask_writer.isOpened())mask_writer.release();
  if(geometry_writer.isOpened())geometry_writer.release();
  if(display)cv::destroyAllWindows();
  if(frames==0)throw std::runtime_error("Aucune frame decodable");
  std::cout<<"Frames: "<<frames<<"\nMode: "<<mode<<"\nSeuil: "<<threshold
           <<"\nAire minimale: "<<min_area<<" pixels\nFermeture verticale: "<<close_vertical<<" pixels\nROI: "<<(roi?"activee":"desactivee")<<"\n";
  if(roi)std::cout<<"ROI top-y="<<roi_top_y<<" top-left="<<roi_top_left
                  <<" top-right="<<roi_top_right<<" bottom-left="<<roi_bottom_left
                  <<" bottom-right="<<roi_bottom_right<<"\n";
  std::cout<<"Hough: "<<(hough?"active":"inactif")<<"\n";
  if(!geometry_output.empty())std::cout<<"Video segments Hough: "<<geometry_output<<'\n';
  if(mode!="mask")std::cout<<"Video superposition: "<<output<<'\n';
  if(mode=="mask")std::cout<<"Video masque: "<<output<<'\n';
  if(mode=="both")std::cout<<"Video masque: "<<mask_output<<'\n';
  std::cout<<"Temps moyen traitement/image (hors encodage): "<<std::fixed<<std::setprecision(2)<<total_ms/frames<<" ms\n";
  std::cout<<"NOTE: performance temps reel non demontree.\n";
 }catch(const std::exception&e){std::cerr<<"ERREUR: "<<e.what()<<'\n';return 1;}
}
