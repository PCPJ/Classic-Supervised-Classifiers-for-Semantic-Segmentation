#include "gaborFeatures.hpp"
#include "utils/util.hpp"
#include "debug.h"
#include <omp.h>

using namespace std;
using namespace cv;

GaborFeature::GaborFeature(const GaborTexture& gabor, ImageFeatures *seedFeature)
    : gabor(gabor), seedFeature(seedFeature)
{
    assert(seedFeature != 0);
    dimensions = seedFeature->getDimentions() * gabor.kernelsSize() + seedFeature->getDimentions();
}

GaborFeature::~GaborFeature()
{

}

void GaborFeature::getFeature(vector<Point2i> &pixCoordinates, vector<double *> &outFeatures)
{

}

void GaborFeature::getFeature(vector<double *> &outFeatures)
{

}

void GaborFeature::getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outFeatures.data == NULL){
        outFeatures = Mat(pixCoordinates.size(), dimensions, CV_32FC1);
    }
    else{
        assert(outFeatures.rows == pixCoordinates.size());
        assert(outFeatures.cols == dimensions);
        assert(outFeatures.type() == CV_32FC1);
    }

    PRINT_DEBUG("Gabor Features Start!");
    static const int numThread = omp_get_max_threads();

    Mat seedImage;
    seedFeature->getFeatures(image, seedImage);

    vector<Mat> coordinatesFeatures(gabor.kernelsSize()+1);
    Mat seedCoordsF = Mat(pixCoordinates.size(), seedFeature->getDimentions(), CV_32FC1);
    for(int p = 0; p < pixCoordinates.size(); p++){
        const Point2i& coord = pixCoordinates[p];
        float* seedPixPtr = seedImage.ptr<float>(coord.y, coord.x*seedFeature->getDimentions());
        float* coordinatesPtr = seedCoordsF.ptr<float>(p, 0);
        memcpy(coordinatesPtr, seedPixPtr, sizeof(float)*seedFeature->getDimentions());
    }
    coordinatesFeatures[0] = seedCoordsF;

    seedImage = seedImage.reshape(seedFeature->getDimentions(), image.rows);
    #pragma omp parallel for num_threads(numThread)
    for(int k = 0; k < gabor.kernelsSize(); k++){
        Mat convolved;
        gabor.convolution(seedImage, k, convolved);
        convolved = convolved.reshape(1, image.rows);
        Mat coordsF = Mat(pixCoordinates.size(), seedFeature->getDimentions(), CV_32FC1);
        for(int p = 0; p < pixCoordinates.size(); p++){
            const Point2i& coord = pixCoordinates[p];
            float* convolvedPixPtr = convolved.ptr<float>(coord.y, coord.x*seedFeature->getDimentions());
            float* coordinatesPtr = coordsF.ptr<float>(p, 0);
            memcpy(coordinatesPtr, convolvedPixPtr, sizeof(float)*seedFeature->getDimentions());
        }
        coordinatesFeatures[k+1] = coordsF;
    }
    cv::hconcat(coordinatesFeatures, outFeatures);
    assert(outFeatures.cols == dimensions);
    assert(outFeatures.rows == pixCoordinates.size());
    PRINT_DEBUG("Gabor Features End!");
}

void GaborFeature::getFeatures(const Mat &image, Mat &outFeatureImage)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outFeatureImage.data == NULL){
        outFeatureImage = Mat(image.rows, image.cols*dimensions, CV_32FC1);
    }
    else{
        assert(outFeatureImage.rows == image.rows);
        assert(outFeatureImage.cols == image.cols*dimensions);
        assert(outFeatureImage.type() == CV_32FC1);
    }

    PRINT_DEBUG("Gabor Features Start!");
    static const int numThread = omp_get_max_threads();
    vector<Mat> featuresImages(gabor.kernelsSize()+1);

    Mat seedImage;
    seedFeature->getFeatures(image, seedImage);
    seedImage = seedImage.reshape(1, image.total());
    featuresImages[0] = seedImage.clone();
    seedImage = seedImage.reshape(seedFeature->getDimentions(), image.rows);
    debugSaveImage("convolved/seed.png", seedImage);

    vector<Mat> convolved(gabor.kernelsSize());
    static int debugConvolvedCount = 0;
    #pragma omp parallel for num_threads(numThread)
    for(int k = 0; k < gabor.kernelsSize(); k++){
        gabor.convolution(seedImage, k, convolved[k]);
        debugSaveImage("convolved/"+to_string(k)+".png", convolved[k]);
        convolved[k] = convolved[k].reshape(1, image.total());
        featuresImages[k+1] = convolved[k].clone();
    }
    cv::hconcat(featuresImages, outFeatureImage);
    assert(outFeatureImage.rows = image.total());
    assert(outFeatureImage.cols = dimensions);
    outFeatureImage = outFeatureImage.reshape(1, image.rows);
    assert(outFeatureImage.rows = image.rows);
    assert(outFeatureImage.cols = image.cols*dimensions);
    PRINT_DEBUG("Gabor Features End!");
}


