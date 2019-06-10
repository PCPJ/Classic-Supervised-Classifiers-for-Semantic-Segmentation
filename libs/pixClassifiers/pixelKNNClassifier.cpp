#include "pixelKNNClassifier.hpp"

PixelKNNClassifier::PixelKNNClassifier(ImageFeatures *featureCalculator, uint k)
    : PixelClassifier(featureCalculator), knn(), k(k)
{
    knn = ml::KNearest::create();
//    knn->setAlgorithmType(ml::KNearest::KDTREE); // OPENCV BUG!!! DONT WORKS!!!
    knn->setAlgorithmType(ml::KNearest::BRUTE_FORCE);
    knn->setIsClassifier(true);
    knn->setDefaultK(k);
//    knn->setEmax(std::numeric_limits<int>::max()); // JUST FOR KDTREE!
}

PixelKNNClassifier::~PixelKNNClassifier()
{

}

void PixelKNNClassifier::setTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords)
{
    assert(sampleImage.data);
    assert(sampleImage.type() == CV_8UC3);

    trainData.release();
    trainDataLabels.release();
    addTrainData(sampleImage, samplePixCords);
}

void PixelKNNClassifier::setTrainData(Mat& imageSample, Mat& imageLabel)
{
    assert(imageSample.type() == CV_8UC3);
    assert(imageLabel.type() == CV_32SC1);
    assert(imageSample.total() == imageLabel.total());

    trainData.release();
    trainDataLabels.release();
    addTrainData(imageSample, imageLabel);
}

void PixelKNNClassifier::addTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords)
{
    assert(sampleImage.data);
    assert(sampleImage.type() == CV_8UC3);

    uint dimensions = featureCalculator->getDimentions();

    uint samplesSize = 0;
    for(vector<Point2i>& v : samplePixCords)
        samplesSize += v.size();
    vector<Point2i> allCords;
    allCords.reserve(samplesSize);
    for(vector<Point2i>& v : samplePixCords)
        allCords.insert(allCords.end(), v.begin(), v.end());
    assert(samplesSize == allCords.size());
    Mat newTrainData = Mat(samplesSize, dimensions, CV_32FC1);
    Mat newTrainDataLabels = Mat(samplesSize, 1, CV_32SC1);

    featureCalculator->getFeatures(sampleImage, allCords, newTrainData);
    int sampleCount = 0;
    for(int i = 0; i < samplePixCords.size(); i++){
        Mat trainDataLabelsI = Mat(newTrainDataLabels, Rect(0, sampleCount, 1, samplePixCords[i].size()));
        trainDataLabelsI.setTo(i);
        sampleCount += samplePixCords[i].size();
    }

    if(trainData.data){
        vconcat(trainData, newTrainData, trainData);
        vconcat(trainDataLabels, newTrainDataLabels, trainDataLabels);
    }else{
        trainData = newTrainData;
        trainDataLabels = newTrainDataLabels;
    }


    assert(sampleCount == samplesSize);
    assert(trainData.rows == trainDataLabels.rows);
    assert(trainData.cols == featureCalculator->getDimentions());
    assert(trainDataLabels.cols == 1);
}

void PixelKNNClassifier::addTrainData(Mat& imageSamples, Mat& imageLabels)
{
    assert(imageSamples.type() == CV_8UC3);
    assert(imageLabels.type() == CV_32SC1);
    assert(imageSamples.total() == imageLabels.total());

    Mat newTrainData;
    featureCalculator->getFeatures(imageSamples, newTrainData);
    newTrainData = newTrainData.reshape(1, imageSamples.total());
    Mat newTrainDataLabels = imageLabels.reshape(1, imageSamples.total());

    if(trainData.data){
        vconcat(trainData, newTrainData, trainData);
        vconcat(trainDataLabels, newTrainDataLabels, trainDataLabels);
    }else{
        trainData = newTrainData;
        trainDataLabels = newTrainDataLabels.clone();
    }

    assert(trainData.rows == trainDataLabels.rows);
    assert(trainData.cols == featureCalculator->getDimentions());
    assert(trainDataLabels.cols == 1);
}

void PixelKNNClassifier::train()
{
    knn->train(trainData, ml::ROW_SAMPLE, trainDataLabels);
}

void PixelKNNClassifier::classify(Mat& inImage, Mat& outLabelImage)
{
    assert(inImage.data);
    assert(inImage.type() == CV_8UC3);

    Mat featureImage;
    featureCalculator->getFeatures(inImage, featureImage);

    Mat dataInImage = featureImage.reshape(1, inImage.total());
    Mat outLabels;

    knn->findNearest(dataInImage, k, outLabels);

    outLabelImage = outLabels.reshape(1, inImage.rows);
    outLabelImage.convertTo(outLabelImage, CV_32SC1);

    assert(outLabelImage.rows = inImage.rows);
    assert(outLabelImage.cols = inImage.cols);
    assert(dataInImage.rows == inImage.total());
    assert(dataInImage.cols == featureCalculator->getDimentions());
}

void PixelKNNClassifier::setK(uint k)
{
    this->k = k;
}
