#include "utils.h"
using namespace std;

size_t utils::vectorProduct(const vector<int64_t> &vector) {
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto &element : vector)
        product *= element;

    return product;
}

wstring utils::charToWstring(const char *str) {
    typedef codecvt_utf8<wchar_t> convert_type;
    wstring_convert<convert_type, wchar_t> converter;

    return converter.from_bytes(str);
}

vector<string> utils::loadNames(const string &path) {
    // load class names
    vector<string> classNames;
    ifstream infile(path);
    if (infile.good()) {
        string line;
        while (getline(infile, line)) {
            if (line.back() == '\r')
                line.pop_back();
            classNames.emplace_back(line);
        }
        infile.close();
    } else {
        cerr << "ERROR: Failed to access class name path: " << path << endl;
    }
    // set color
    srand(time(0));

    for (int i = 0; i < 2 * classNames.size(); i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        colors.push_back(cv::Scalar(b, g, r));
    }
    return classNames;
}

void utils::visualizeDetection(cv::Mat &im, vector<Yolov8Result> &results,
                               const vector<string> &classNames) {
    cv::Mat image = im.clone();
    for (const Yolov8Result &result : results) {

        int x = result.box.x;
        int y = result.box.y;

        int conf = (int)round(result.conf * 100);
        int classId = result.classId;
        string label = classNames[classId] + " 0." + to_string(conf);
        cout << "label name " << label <<endl;
        int baseline = 0;
        cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.4, 1, &baseline);
        image(result.box).setTo(colors[classId + classNames.size()], result.boxMask);
        cv::rectangle(image, result.box, colors[classId], 2);
        cv::rectangle(image,
                      cv::Point(x, y), cv::Point(x + size.width, y + 12),
                      colors[classId], -1);
        cv::putText(image, label,
                    cv::Point(x, y - 3 + 12), cv::FONT_ITALIC,
                    0.4, cv::Scalar(0, 0, 0), 1);
    }
    cv::addWeighted(im, 0.4, image, 0.6, 0, im);

}

void utils::letterbox(const cv::Mat &image, cv::Mat &outImage,
                      const cv::Size &newShape = cv::Size(640, 640),
                      const cv::Scalar &color = cv::Scalar(114, 114, 114),
                      bool auto_ = true,
                      bool scaleFill = false,
                      bool scaleUp = true,
                      int stride = 32) {
    cv::Size shape = image.size();
    float r = min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = min(r, 1.0f);

    float ratio[2]{r, r};
    int newUnpad[2]{(int)round((float)shape.width * r),
                    (int)round((float)shape.height * r)};

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);
    if (auto_) {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    } else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1]) {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(round(dh - 0.1f));
    int bottom = int(round(dh + 0.1f));
    int left = int(round(dw - 0.1f));
    int right = int(round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void utils::scaleCoords(cv::Rect &coords,
                        cv::Mat &mask,
                        const float maskThreshold,
                        const cv::Size &imageShape,
                        const cv::Size &imageOriginalShape) {
    float gain = min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = {(int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

    coords.x = (int)round(((float)(coords.x - pad[0]) / gain));
    coords.x = max(0, coords.x);
    coords.y = (int)round(((float)(coords.y - pad[1]) / gain));
    coords.y = max(0, coords.y);

    coords.width = (int)round(((float)coords.width / gain));
    coords.width = min(coords.width, imageOriginalShape.width - coords.x);
    coords.height = (int)round(((float)coords.height / gain));
    coords.height = min(coords.height, imageOriginalShape.height - coords.y);
    mask = mask(cv::Rect(pad[0], pad[1], imageShape.width - 2 * pad[0], imageShape.height - 2 * pad[1]));

    cv::resize(mask, mask, imageOriginalShape, cv::INTER_LINEAR);

    mask = mask(coords) > maskThreshold;
}
template <typename T>
T utils::clip(const T &n, const T &lower, const T &upper) {
    return max(lower, min(n, upper));
}
