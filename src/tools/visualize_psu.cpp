
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " image.jpg labels.txt sortie.jpg\n";
        return 1;
    }

    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Image introuvable\n";
        return 1;
    }

    std::ifstream file(argv[2]);
    if (!file) {
        std::cerr << "Annotations introuvables\n";
        return 1;
    }

    // Couleurs BGR pour les classes 0 à 4
    const std::vector<cv::Scalar> colors = {
        {0, 0, 255},    // 0 : rouge
        {0, 255, 0},    // 1 : vert
        {255, 0, 0},    // 2 : bleu
        {0, 255, 255},  // 3 : jaune
        {255, 0, 255}   // 4 : magenta
    };

    cv::Mat overlay = image.clone();
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream stream(line);
        int cls;
        if (!(stream >> cls)) continue;

        if (cls < 0 || cls >= static_cast<int>(colors.size())) {
            std::cerr << "Classe inconnue: " << cls << "\n";
            continue;
        }

        std::vector<double> coords;
        double value;
        while (stream >> value) coords.push_back(value);

        if (coords.size() < 6 || coords.size() % 2 != 0) {
            std::cerr << "Polygone invalide\n";
            continue;
        }

        std::vector<cv::Point> points;
        for (size_t i = 0; i < coords.size(); i += 2) {
            int x = std::clamp(
                static_cast<int>(std::lround(coords[i] * (image.cols - 1))),
                0, image.cols - 1
            );
            int y = std::clamp(
                static_cast<int>(std::lround(coords[i + 1] * (image.rows - 1))),
                0, image.rows - 1
            );
            points.emplace_back(x, y);
        }

        cv::fillPoly(overlay,
                     std::vector<std::vector<cv::Point>>{points},
                     colors[cls]);

        cv::polylines(image,
                      std::vector<std::vector<cv::Point>>{points},
                      true, colors[cls], 2);
    }

    cv::addWeighted(overlay, 0.30, image, 0.70, 0, image);

    cv::imwrite(argv[3], image);
    std::cout << "Image generee: " << argv[3] << "\n";
    return 0;
}
