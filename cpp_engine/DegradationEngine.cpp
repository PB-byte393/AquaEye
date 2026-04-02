#include "DegradationEngine.h"
#include <iostream>
#include <random>

DegradationEngine::DegradationEngine() : mapsInitialized(false) {
}

void DegradationEngine::buildDistortionMap(cv::Size size, double k1, double k2) {
    mapX = cv::Mat(size, CV_32FC1);
    mapY = cv::Mat(size, CV_32FC1);

    float cx = size.width / 2.0f;
    float cy = size.height / 2.0f;

    // THE PHYSICS LOOP (Forward Distortion Model)
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            
            // Normalize coordinates (-1 to +1)
            float normX = (x - cx) / cx;
            float normY = (y - cy) / cy;
            
            // Radius squared
            float r2 = normX*normX + normY*normY;

            // Barrel Distortion Formula: r_distorted = r * (1 + k1*r^2 + k2*r^4)
            // We use a slight zoom (0.9) to cut off black edges
            float factor = 1.0 + k1 * r2 + k2 * r2 * r2;
            
            // Map back to pixel coordinates
            mapX.at<float>(y, x) = cx + (normX * factor * cx) * 0.9;
            mapY.at<float>(y, x) = cy + (normY * factor * cy) * 0.9;
        }
    }
    mapsInitialized = true;
}

cv::Mat DegradationEngine::processImage(const cv::Mat& cleanImage) {
    if (cleanImage.empty()) return cleanImage;

    // 1. Initialize maps on first run
    if (!mapsInitialized) {
        // k1 = 0.2 (Positive = Barrel Distortion in this forward model)
        buildDistortionMap(cleanImage.size(), 0.2, 0.05);
    }

    cv::Mat processed = cleanImage.clone();

    // 2. Geometric Distortion (Using our manual map)
    processed = applyGeometricDistortion(processed);

    // 3. Chromatic Aberration (Color Bleed)
    processed = applyChromaticAberration(processed, 0.005);

    // 4. Vignette (Dark corners)
    processed = applyVignette(processed, 0.7f);

    // 5. Sensor Noise (Grain)
    processed = applySensorNoise(processed, 8.0); 

    return processed;
}

cv::Mat DegradationEngine::applyGeometricDistortion(const cv::Mat& src) {
    cv::Mat distorted;
    // Remap pixels using our pre-calculated Physics Map
    cv::remap(src, distorted, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return distorted;
}

cv::Mat DegradationEngine::applyChromaticAberration(const cv::Mat& src, double spread) {
    std::vector<cv::Mat> channels;
    cv::split(src, channels); 

    // Green is anchor. Red expands, Blue shrinks.
    cv::Mat r_scaled, b_scaled;
    
    // Resize Red (Zoom IN)
    double sR = 1.0 + spread;
    cv::resize(channels[2], r_scaled, cv::Size(), sR, sR);
    
    // Resize Blue (Zoom OUT)
    double sB = 1.0 - spread;
    cv::resize(channels[0], b_scaled, cv::Size(), sB, sB);

    // Crop Red center
    int cx = r_scaled.cols / 2;
    int cy = r_scaled.rows / 2;
    int w = src.cols;
    int h = src.rows;
    cv::getRectSubPix(r_scaled, cv::Size(w, h), cv::Point2f(cx, cy), channels[2]);

    // Pad Blue border
    int top = (h - b_scaled.rows) / 2;
    int left = (w - b_scaled.cols) / 2;
    cv::copyMakeBorder(b_scaled, channels[0], top, h - b_scaled.rows - top, left, w - b_scaled.cols - left, cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::Mat result;
    cv::merge(channels, result);
    return result;
}

cv::Mat DegradationEngine::applyVignette(const cv::Mat& src, float strength) {
    cv::Mat mask(src.size(), CV_32F);
    float cx = src.cols / 2.0f;
    float cy = src.rows / 2.0f;
    float maxR = std::sqrt(cx*cx + cy*cy);

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float dx = x - cx;
            float dy = y - cy;
            float r = std::sqrt(dx*dx + dy*dy);
            float factor = 1.0f - strength * (r / maxR);
            mask.at<float>(y, x) = std::max(0.1f, factor);
        }
    }

    cv::Mat floatSrc, result;
    src.convertTo(floatSrc, CV_32F);
    
    std::vector<cv::Mat> ch;
    cv::split(floatSrc, ch);
    for(int i=0; i<3; i++) cv::multiply(ch[i], mask, ch[i]);
    cv::merge(ch, result);
    
    result.convertTo(result, CV_8U);
    return result;
}

cv::Mat DegradationEngine::applySensorNoise(const cv::Mat& src, double sigma) {
    cv::Mat noise(src.size(), src.type());
    cv::randn(noise, 0, sigma);
    return src + noise;
}