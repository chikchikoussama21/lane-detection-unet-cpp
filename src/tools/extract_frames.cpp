#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace fs = std::filesystem;
int main(int argc, char** argv) {
 try {
  if(argc != 4) { std::cerr << "Usage: extract_frames VIDEO DOSSIER_SORTIE PAS_EN_FRAMES\nExemple: extract_frames route.mp4 data/raw/images/video01 15\n"; return 2; }
  const int every = std::stoi(argv[3]);
  if(every <= 0) throw std::runtime_error("PAS_EN_FRAMES doit etre > 0");
  cv::VideoCapture capture(argv[1]);
  if(!capture.isOpened()) throw std::runtime_error("Impossible d'ouvrir la video");
  fs::create_directories(argv[2]);
  std::size_t frame_idx = 0, saved = 0;
  cv::Mat frame;
  while(capture.read(frame)) {
   if(frame_idx % static_cast<std::size_t>(every) == 0) {
    std::ostringstream name;
    name << "frame_" << std::setw(7) << std::setfill('0') << frame_idx << ".png";
    if(!cv::imwrite((fs::path(argv[2]) / name.str()).string(),frame)) throw std::runtime_error("Ecriture image echouee");
    ++saved;
   }
   ++frame_idx;
  }
  std::cout << "Video: " << frame_idx << " frames ; images sauvegardees: " << saved << " ; FPS d'origine: " << capture.get(cv::CAP_PROP_FPS) << '\n';
 } catch(const std::exception& e) { std::cerr << "ERREUR: " << e.what() << '\n'; return 1; }
}
