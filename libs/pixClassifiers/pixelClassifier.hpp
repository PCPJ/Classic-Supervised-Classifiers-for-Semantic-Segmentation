#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

#include "imageFeatures/imageFeatures.hpp"

using namespace std;
using namespace cv;

class PixelClassifier{
public:
    PixelClassifier(ImageFeatures* calculator);
    virtual ~PixelClassifier();

    //sampleImage - rgb NxM image
    //samplePixCords - coordinates of samples pixel, for each class
    //Clean the train data, and adds this imputs on the training data set.
    virtual void setTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords) = 0;
    //sampleImage - rgb NxM image
    //imageLabelslabels - 32SC1 mat, same size as sampleImage
    //Clean the train data, and adds this imputs on the training data set.
    virtual void setTrainData(Mat& imageSample, Mat& imageLabels) = 0;

    //sampleImage - rgb NxM image
    //samplePixCords - coordinates of samples, for each class
    //Adds this imputs on the training data set.
    virtual void addTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords) = 0;
    //sampleImage - rgb NxM image
    //imageLabelslabels - 32SC1 mat, same size as sampleImage
    //Adds this imputs on the training data set.
    virtual void addTrainData(Mat& imageSamples, Mat& imageLabels) = 0;

    virtual bool loadTrainData(string trainDataConfFilePath, string colorsConfFilePath);

    virtual void train() = 0;

    virtual void run(Mat& inImage, Mat& outLabelImage);


protected:
    virtual void classify(Mat& inImage, Mat& outLabelImage) = 0;

    ImageFeatures* featureCalculator;
};
