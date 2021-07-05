#include "ReidDescriptor.hpp"

ReidDescriptor::ReidDescriptor(const std::string& FLAGS_mReidentification, const std::string& FLAGS_cReidentification, const InferenceEngine::Core& ie) :
    FLAGS_mReidentification(FLAGS_mReidentification),
    FLAGS_cReidentification(FLAGS_cReidentification),
    ie(ie)
{
    reidnet = new ReidentificationNet(FLAGS_mReidentification, FLAGS_cReidentification, ie);
    descr_size_ = {1,256};
}

void ReidDescriptor::compute(const cv::Mat& mat, CV_OUT cv::Mat& descr)
{
    cv::Mat denseCopy;
    mat.copyTo(denseCopy);
    descr = reidnet->doEverything(denseCopy);
}

void ReidDescriptor::compute(const std::vector<cv::Mat>& mats,
    CV_OUT std::vector<cv::Mat>& descrs)
{
    descrs.resize(mats.size());
    for (size_t i = 0; i < mats.size(); i++) {
        compute(mats[i], descrs[i]);
    }
}


ReidDescriptor::~ReidDescriptor()
{
}
