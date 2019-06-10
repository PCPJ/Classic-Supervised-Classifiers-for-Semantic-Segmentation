#include "pixelNNClassifier.hpp"
#include "debug.h"

using namespace std;
using namespace cv;

PixelNNClassifier::PixelNNClassifier(ImageFeatures *featureCalculator)
    : PixelClassifier(featureCalculator), nn(), classesLabels()
{
    nn = ml::ANN_MLP::create();

    nn->setActivationFunction(ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM);
    nn->setTrainMethod(ml::ANN_MLP::TrainingMethods::BACKPROP);
    nn->setBackpropWeightScale(0.1);//learning rate? i think so.
    nn->setBackpropMomentumScale(0.1);//learning rate changes over iterations?
    nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 1e-6));
}

PixelNNClassifier::~PixelNNClassifier()
{

}

void PixelNNClassifier::setTrainData(Mat &sampleImage, vector<vector<Point2i> > &samplePixCords)
{
    assert(sampleImage.data);
    assert(sampleImage.type() == CV_8UC3);

    trainData.release();
    trainDataLabels.release();
    classesLabels.clear();
    addTrainData(sampleImage, samplePixCords);
}

void PixelNNClassifier::setTrainData(Mat &imageSample, Mat &imageLabel)
{
    assert(imageSample.type() == CV_8UC3);
    assert(imageLabel.type() == CV_32SC1);
    assert(imageSample.total() == imageLabel.total());

    trainData.release();
    trainDataLabels.release();
    classesLabels.clear();
    addTrainData(imageSample, imageLabel);
}

void PixelNNClassifier::addTrainData(Mat &sampleImage, vector<vector<Point2i> > &samplePixCords)
{
    assert(sampleImage.data);
    assert(sampleImage.type() == CV_8UC3);

    uint dimensions = featureCalculator->getDimentions();

    uint samplesSize = 0;
    int labelCounts = 0;
    for(vector<Point2i>& v : samplePixCords){
        samplesSize += v.size();
        classesLabels.insert(labelCounts++);
    }
    Mat newTrainData = Mat(samplesSize, dimensions, CV_32FC1);
    Mat newTrainDataLabels = Mat(samplesSize, 1, CV_32SC1);

    int sampleCount = 0;
    for(int i = 0; i < samplePixCords.size(); i++){
        Mat features = Mat(newTrainData, Rect(0, sampleCount, dimensions, samplePixCords[i].size()));
        Mat trainDataLabelsI = Mat(newTrainDataLabels, Rect(0, sampleCount, 1, samplePixCords[i].size()));
        featureCalculator->getFeatures(sampleImage, samplePixCords[i], features);
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

    ///////////////////////////////
    //setting the output layer size
    int inputLayerSize = featureCalculator->getDimentions();
    vector<int> layers = {inputLayerSize, inputLayerSize*10, inputLayerSize*10, (int)classesLabels.size()};
    nn->setLayerSizes(layers);
    nn->setActivationFunction(ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM);
    //opencv requires to reset the activation function if you modify the network layers.
    //why? how knows!?
    ///////////////////////////////

    assert(sampleCount == samplesSize);
    assert(trainData.rows == trainDataLabels.rows);
    assert(trainData.cols == featureCalculator->getDimentions());
    assert(trainDataLabels.cols == 1);
}

void PixelNNClassifier::addTrainData(Mat &imageSamples, Mat &imageLabels)
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

    ///////////////////////////////////////////////////////////////////////////
    //storing all classes labels, so we can keep tracking the number of classes
    for(int i = 0; i < trainDataLabels.rows; i++){
        classesLabels.insert(trainDataLabels.at<int>(i,0));
    }
    ///////////////////////////////////////////////////////////////////////////
    //setting the output layer size
    int inputLayerSize = featureCalculator->getDimentions();
    vector<int> layers = {inputLayerSize, inputLayerSize*10, inputLayerSize*10, (int)classesLabels.size()};
    nn->setLayerSizes(layers);
    nn->setActivationFunction(ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM);
    //opencv requires to reset the activation function if you modify the network layers.
    //why? how knows!?
    ///////////////////////////////

    assert(trainData.rows == trainDataLabels.rows);
    assert(trainData.cols == featureCalculator->getDimentions());
    assert(trainDataLabels.cols == 1);
}

void PixelNNClassifier::train()
{
    cout << "method = " << (nn->getTrainMethod() == 0 ? "BACK_PROP" : "RPROP") << endl;
    cout << "term crit iterations = " << nn->getTermCriteria().maxCount << endl;
    cout << "term crit accur = " << nn->getTermCriteria().epsilon << endl;
    cout << "learning rate = " << nn->getBackpropWeightScale() << endl;
    cout << "momentum = " << nn->getBackpropMomentumScale() << endl;
    cout << "layers = " << nn->getLayerSizes() << endl;
    nn->train(trainData, ml::ROW_SAMPLE, toOneHot());
}

void PixelNNClassifier::classify(Mat &inImage, Mat &outLabelImage)
{
    assert(inImage.data);
    assert(inImage.type() == CV_8UC3);

    Mat featureImage;
    featureCalculator->getFeatures(inImage, featureImage);
    Mat dataInImage = featureImage.reshape(1, inImage.total());
    Mat oneHotPrediction;
    PRINT_DEBUG("NN PREDICTION START!");
    nn->predict(dataInImage, oneHotPrediction);
    PRINT_DEBUG("NN PREDICTION END!");
    outLabelImage = fromOneHot(oneHotPrediction);
    outLabelImage = outLabelImage.reshape(1, inImage.rows);

    assert(outLabelImage.rows = inImage.rows);
    assert(outLabelImage.cols = inImage.cols);
    assert(dataInImage.rows == inImage.total());
    assert(dataInImage.cols == featureCalculator->getDimentions());
}

Mat PixelNNClassifier::toOneHot() const
{
    Mat oneHot = Mat::zeros(trainDataLabels.rows, classesLabels.size(), CV_32FC1);
    for(int i = 0; i < trainDataLabels.rows; i++){
        int label = trainDataLabels.at<int>(i,0);
        oneHot.at<float>(i, label) = 1.0;
    }
    return oneHot;
}

Mat PixelNNClassifier::fromOneHot(const Mat &oneHotLabels) const
{
    Mat labels = Mat(oneHotLabels.rows, 1, CV_32SC1);
    for(int i = 0; i < oneHotLabels.rows; i++){
        const Mat& row = oneHotLabels.row(i);
        Point maxLoc;
        cv::minMaxLoc(row, 0, 0, 0, &maxLoc);
        labels.at<int>(i, 0) = maxLoc.x;
    }
    return labels;
}


