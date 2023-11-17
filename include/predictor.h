#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>

#include "utils.h"

using namespace std;

class YOLOPredictor {
public:
    explicit YOLOPredictor(nullptr_t){};
    YOLOPredictor(const string &modelPath,
                  const bool &isGPU,
                  float confThreshold,
                  float iouThreshold,
                  float maskThreshold);
    // ~YOLOPredictor();
    vector<Yolov8Result> predict(cv::Mat &image);
    int classNums = 10;

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    void preprocessing(cv::Mat &image, float *&blob, vector<int64_t> &inputTensorShape);
    vector<Yolov8Result> postprocessing(const cv::Size &resizedImageShape,
                                             const cv::Size &originalImageShape,
                                             vector<Ort::Value> &outputTensors);

    static void getBestClassInfo(vector<float>::iterator it,
                                 float &bestConf,
                                 int &bestClassId,
                                 const int _classNums);
    cv::Mat getMask(const cv::Mat &maskProposals, const cv::Mat &maskProtos);
    bool isDynamicInputShape{};

    vector<const char *> inputNames;
    vector<Ort::AllocatedStringPtr> input_names_ptr;

    vector<const char *> outputNames;
    vector<Ort::AllocatedStringPtr> output_names_ptr;

    vector<vector<int64_t>> inputShapes;
    vector<vector<int64_t>> outputShapes;
    float confThreshold = 0.3f;
    float iouThreshold = 0.4f;

    bool hasMask = false;
    float maskThreshold = 0.5f;
};