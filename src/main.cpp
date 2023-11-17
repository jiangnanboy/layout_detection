#include <regex>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <ctime>
#include "utils.h"
#include "predictor.h"

YOLOPredictor initModel(const string modelPath, const bool isGPU, const float confThreshold, const float iouThreshold, const float maskThreshold);
const vector<string> initLabelName(const string lableNamePath);
using namespace std;

int main() {
    const string modelPath = "E:\\clion_project\\layout_onnx_cplusplus\\models\\model_det.onnx";
    const string labelNamesPath = "E:\\clion_project\\layout_onnx_cplusplus\\models\\label.names";
    filesystem::path imagePath = "E:\\clion_project\\layout_onnx_cplusplus\\test_img\\test.jpeg";
    const string savePath = "E:\\clion_project\\layout_onnx_cplusplus\\test_img_result";

    float confThreshold = 0.4f;
    float iouThreshold = 0.4f;
    float maskThreshold = 0.5f;
    bool isGPU = false;

    const vector<string> labelNames = initLabelName(labelNamesPath);

    if (labelNames.empty())
    {
        cerr << "Error: Label names file is empty!" << endl;
        return -1;
    }
    if (!filesystem::exists(modelPath))
    {
        cerr << "Error: Model path is empty!" << endl;
        return -1;
    }
    YOLOPredictor predictor = initModel(modelPath, isGPU, confThreshold, iouThreshold, maskThreshold);
    assert(labelNames.size() == predictor.classNums);
    regex pattern(".+\\.(jpg|jpeg|png|gif)$");
    cout << "Start inferring..." << endl;

    clock_t startTime, endTime;
    startTime = clock();
    if(filesystem::is_regular_file(imagePath) && regex_match(imagePath.string(), pattern)) {
        cv::Mat image = cv::imread(imagePath.string());
        vector<Yolov8Result> result = predictor.predict(image);
        utils::visualizeDetection(image, result, labelNames);
        cv::imwrite(savePath + "\\" + imagePath.filename().string(), image);
    }
    endTime = clock();
    cout << "Time consumption : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "seconds" << endl;
    cout << "##########DONE################" << endl;

    return 0;
}

YOLOPredictor initModel(const string modelPath, const bool isGPU, const float confThreshold, const float iouThreshold, const float maskThreshold) {
    YOLOPredictor predictor{nullptr};
    try {
        predictor = YOLOPredictor(modelPath, isGPU,
                                  confThreshold,
                                  iouThreshold,
                                  maskThreshold);
        cout << "Model was initialized." << endl;
    } catch (const exception &e) {
        cerr << e.what() << endl;
    }
    return predictor;
}

const vector<string> initLabelName(const string lableNamePath) {
    const vector<string> labelNames = utils::loadNames(lableNamePath);
    return labelNames;
}
