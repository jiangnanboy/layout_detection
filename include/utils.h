#pragma once
#include <codecvt>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
struct Yolov8Result {
    cv::Rect box;
    cv::Mat boxMask; // mask in box
    float conf{};
    int classId{};
};

namespace utils {
    static vector<cv::Scalar> colors;

    size_t vectorProduct(const vector<int64_t> &vector);
    wstring charToWstring(const char *str);
    vector<string> loadNames(const string &path);
    void visualizeDetection(cv::Mat &image, vector<Yolov8Result> &results,
                            const vector<string> &classNames);

    void letterbox(const cv::Mat &image, cv::Mat &outImage,
                   const cv::Size &newShape,
                   const cv::Scalar &color,
                   bool auto_,
                   bool scaleFill,
                   bool scaleUp,
                   int stride);

    void scaleCoords(cv::Rect &coords, cv::Mat &mask,
                     const float maskThreshold,
                     const cv::Size &imageShape, const cv::Size &imageOriginalShape);

    template <typename T>
    T clip(const T &n, const T &lower, const T &upper);
}
