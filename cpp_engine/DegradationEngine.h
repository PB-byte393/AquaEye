#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class DegradationEngine {
public:
    DegradationEngine();

    // The Main Pipeline
    cv::Mat processImage(const cv::Mat& cleanImage);

private:
    cv::Mat mapX, mapY;
    bool mapsInitialized;

    // We pre-calculate the distortion map once
    void buildDistortionMap(cv::Size size, double k1, double k2);

    // Optical Faults
    cv::Mat applyGeometricDistortion(const cv::Mat& src);
    cv::Mat applyChromaticAberration(const cv::Mat& src, double spread);
    cv::Mat applyVignette(const cv::Mat& src, float strength);
    cv::Mat applySensorNoise(const cv::Mat& src, double sigma);
};