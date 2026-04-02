#include <iostream>
#include <filesystem>
#include "DegradationEngine.h"

namespace fs = std::filesystem;

int main() {
    // HARDCODED PATHS based on your folder structure
    // Ensure backslashes are escaped (\\) or use forward slashes (/)
    std::string inputPath = "C:/Users/pranj/OneDrive/Desktop/AquaEye_Pro/datasets/AquaEye_Training_Set/images";
    std::string outputPath = "C:/Users/pranj/OneDrive/Desktop/AquaEye_Pro/datasets/AquaEye_Degraded_Set/images";

    if (!fs::exists(inputPath)) {
        std::cerr << "CRITICAL ERROR: Input directory not found: " << inputPath << std::endl;
        return -1;
    }

    if (!fs::exists(outputPath)) {
        fs::create_directories(outputPath);
        std::cout << "Created output directory: " << outputPath << std::endl;
    }

    DegradationEngine engine;
    
    int count = 0;
    std::cout << "Starting Physics-Based Degradation..." << std::endl;

    for (const auto& entry : fs::directory_iterator(inputPath)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            
            // 1. Load Clean "Ground Truth"
            cv::Mat img = cv::imread(entry.path().string());
            if (img.empty()) continue;

            // 2. Destroy it
            cv::Mat degraded = engine.processImage(img);

            // 3. Save it
            std::string outName = outputPath + "/" + entry.path().filename().string();
            cv::imwrite(outName, degraded);

            count++;
            if (count % 100 == 0) std::cout << "Processed " << count << " images..." << std::endl;
        }
    }

    std::cout << "Complete. Degraded " << count << " images." << std::endl;
    std::cout << "These images represent your 'Mobile' domain." << std::endl;
    return 0;
}