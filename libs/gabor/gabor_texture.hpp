#pragma once

#include <opencv2/opencv.hpp>

class GaborTexture{
public:
    GaborTexture(uchar lambdaMethod = LAMBDA_JAIN);
    GaborTexture(const cv::Mat& inImage, uchar lambdaMethod = LAMBDA_JAIN);
    GaborTexture(std::vector<double> thetas, uchar lambdaMethod = LAMBDA_JAIN);
    GaborTexture(cv::Mat& thetas, cv::Mat& waveLengths);
    GaborTexture(std::vector<int> waveLengths, uchar lambdaMethod = CIRCULAR);
    GaborTexture(const GaborTexture& gabor);
    virtual ~GaborTexture();

    int kernelsSize() const;

    void filterKernelListByDirection(std::vector<int> &directionsToKeep);

    void saveKernels(std::string path);

    cv::Mat convolution(const cv::Mat& inImage, int kernelIndex, cv::Point2i p) const;
    void convolution(const cv::Mat& inImage, int kernelIndex, cv::Mat& outImage) const;

    int getKernelMaxSize() const;

    enum{LAMBDA_JAIN = 0, LAMBDA_ZHANG = 1, CIRCULAR = 2};

private:

    void computeLambdaJZhang(unsigned int imageWidth);
    void computeLambdaJain(unsigned int imageWidth);
    cv::Mat createGaborKernel(float lambda, float theta, float gamma, float phi, float b);
    cv::Mat createSmoothKernel(float beta, float lambda);
    void computeNonlinearity(cv::Mat& image) const;


    int kernelMaxSize;

    uchar lambdaMethod;

    int lambdaSize;
    int directions;
    std::vector<double> thetas;
    std::vector<double> waveLengths;
    std::vector<double> lambdas;
    std::vector<cv::Mat> kernels;
    std::vector<cv::Mat> smoothKernels;


};


