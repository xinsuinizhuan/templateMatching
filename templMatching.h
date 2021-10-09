/**************************************
 * FILENAME: TEMPLMATCHING.h
 *
 * AUTHORS: Jin Yun
 *
 * START DATE: 2021/10/09
 *
 * CONTACT: yun.jin@smartmore.com
 **************************************/
#ifndef __SMARTMORE_TEMPLMATCHING_H__
#define __SMARTMORE_TEMPLMATCHING_H__

#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "yaml-cpp/yaml.h"

struct Feature {
    float x;
    float y;
    int label;
    float theta;

    Feature() : x(0), y(0), label(0) {}
    Feature(float x, float y) : x(x), y(y), label(0) {}
    Feature(float x, float y, int label) : x(x), y(y), label(label) {}
};

struct Template {
    int width;
    int height;
    int tl_x;
    int tl_y;
    int pyramid_level;
    std::vector<Feature> features;
    Template(std::vector<Feature> _features)
        : width(0), height(0), tl_x(0), tl_y(0), pyramid_level(0), features(_features) {}
    Template() {}
};

class ColorGradientPyramid {
   public:
    ColorGradientPyramid(const cv::Mat& _src, const cv::Mat& _mask, float _weak_threshold, size_t _num_features,
                         float _strong_threshold)
        : src(_src),
          mask(_mask),
          pyramid_level(0),
          weak_threshold(_weak_threshold),
          num_features(_num_features),
          strong_threshold(_strong_threshold) {}

    void quantize(cv::Mat& dst) const;

    bool extractTemplate(Template& templ, cv::Mat drwa) const;

    void pyrDown(int index);

   public:
    void update();
    /// Candidate feature with a score
    struct Candidate {
        Candidate(int x, int y, int label, float score);

        /// Sort candidates with high score to the front
        bool operator<(const Candidate& rhs) const { return score > rhs.score; }

        Feature f;
        float score;
    };

    cv::Mat src;
    cv::Mat mask;

    int pyramid_level;
    cv::Mat angle;
    cv::Mat magnitude;
    cv::Mat angle_ori;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    static bool selectScatteredFeatures(const std::vector<Candidate>& candidates, std::vector<Feature>& features,
                                        size_t num_features, float distance);
};
inline ColorGradientPyramid::Candidate::Candidate(int x, int y, int label, float _score)
    : f(x, y, label), score(_score) {}

class ColorGradient {
   public:
    ColorGradient() : weak_threshold(30.0f), num_features(63), strong_threshold(60.0f) {}

    ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
        : weak_threshold(_weak_threshold), num_features(_num_features), strong_threshold(_strong_threshold) {}

    float weak_threshold;
    size_t num_features;
    float strong_threshold;

    cv::Ptr<ColorGradientPyramid> process(const cv::Mat src, const cv::Mat& mask = cv::Mat()) const {
        return cv::makePtr<ColorGradientPyramid>(src, mask, weak_threshold, num_features, strong_threshold);
    }
};

class Detector {
   public:
    typedef std::vector<Template> TemplatePyramid;
    typedef std::vector<cv::Mat> LinearMemories;
    typedef std::vector<std::vector<LinearMemories>> LinearMemoryPyramid;
    Detector();

    Detector(std::vector<int> T);
    Detector(int num_features, float weak_thresh, float strong_thresh, std::vector<int> vRotate);
    Detector(int num_features, std::vector<int> T, float weak_thresh = 70.0f, float strong_thresh = 100.0f,
             std::vector<int> vRotate = {0, 360, 1});

    bool addTemplate(const cv::Mat& sources, const cv::Mat& object_mask);

    bool matchTemplate(const cv::Mat& source, int nameNumb);

    cv::Mat transform(cv::Mat src, float angle, float scale);
    bool save(std::string& dir);
    bool load(std::string& dir);
    cv::Size srcSize;
    std::vector<TemplatePyramid> getTemplatePyramid() { return class_templates; }

   protected:
    cv::Ptr<ColorGradient> modality;
    cv::Mat srcImage;
    std::vector<int> vRotate;
    int pyramid_levels;
    std::vector<int> T_at_level;
    std::vector<int> sumVal;
    std::vector<TemplatePyramid> class_templates;
};

// toolFunctions
bool isExitPoint(cv::Point2f p, std::vector<cv::Point2f>& vp, int index);
#endif  // __SMARTMORE_TEMPLMATCHING_H__
