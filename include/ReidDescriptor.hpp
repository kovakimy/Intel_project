#pragma once
#include <opencv2/tracking/tracking_by_matching.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <ie_core.hpp>
#include "../include/ReidNetwork.hpp"

class ReidDescriptor : public cv::detail::tracking::tbm::IImageDescriptor
{
public:
    ReidDescriptor(const std::string&, const std::string&, const InferenceEngine::Core&);
    ~ReidDescriptor();
    void compute(const cv::Mat& mat, CV_OUT cv::Mat& descr);
    void compute(const std::vector<cv::Mat>& mats,
        CV_OUT std::vector<cv::Mat>& descrs);
    cv::Size size() const override { return descr_size_; }
private:
    std::string FLAGS_mReidentification;
    std::string FLAGS_cReidentification;
    InferenceEngine::Core ie;
    ReidentificationNet *reidnet = nullptr;
    cv::Size descr_size_;
};