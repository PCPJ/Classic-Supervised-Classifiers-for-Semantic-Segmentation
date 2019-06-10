#include "pixelSVMClassifier.hpp"

#include "debug.h"

using namespace std;
using namespace cv;

PixelSVMClassifier::PixelSVMClassifier(ImageFeatures *featureCalculator, Kernel k)
    : PixelClassifier(featureCalculator), svm()
{
    svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
//    svm->setType(ml::SVM::NU_SVC);
//    svm->setKernel(ml::SVM::LINEAR);//RUIM - RAPIDO
//    svm->setKernel(ml::SVM::POLY);
//    svm->setKernel(ml::SVM::CHI2);
//    svm->setKernel(ml::SVM::RBF);
//    svm->setKernel(ml::SVM::SIGMOID);
    switch (k) {
    case LINEAR:
        svm->setKernel(ml::SVM::LINEAR);
        break;
    case RBF:
        svm->setKernel(ml::SVM::RBF);
        break;
    default:
        svm->setKernel(ml::SVM::LINEAR);
        break;
    }
    svm->setC(0.1);
    svm->setGamma(2);
    svm->setP(0.01);
    svm->setNu(0.01);
    svm->setCoef0(0.1);
    svm->setDegree(0.1);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 1e-6));
}

PixelSVMClassifier::~PixelSVMClassifier()
{

}

void PixelSVMClassifier::setTrainData(Mat &sampleImage, vector<vector<Point2i> > &samplePixCords)
{
    assert(sampleImage.data);
    assert(sampleImage.type() == CV_8UC3);

    trainData.release();
    trainDataLabels.release();
    addTrainData(sampleImage, samplePixCords);
}

void PixelSVMClassifier::setTrainData(Mat &imageSample, Mat &imageLabel)
{
    assert(imageSample.type() == CV_8UC3);
    assert(imageLabel.type() == CV_32SC1);
    assert(imageSample.total() == imageLabel.total());

    trainData.release();
    trainDataLabels.release();
    addTrainData(imageSample, imageLabel);
}

void PixelSVMClassifier::addTrainData(Mat &sampleImage, vector<vector<Point2i> > &samplePixCords)
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

void PixelSVMClassifier::addTrainData(Mat &imageSamples, Mat &imageLabels)
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

void PixelSVMClassifier::train()
{
    Ptr<ml::ParamGrid> cGrid = ml::SVM::getDefaultGridPtr(ml::SVM::C);
    Ptr<ml::ParamGrid> gammaGrid = ml::SVM::getDefaultGridPtr(ml::SVM::GAMMA);
    Ptr<ml::ParamGrid> pGrid = ml::SVM::getDefaultGridPtr(ml::SVM::P);
    Ptr<ml::ParamGrid> nuGrid = ml::SVM::getDefaultGridPtr(ml::SVM::NU);
    Ptr<ml::ParamGrid> coefGrid = ml::SVM::getDefaultGridPtr(ml::SVM::COEF);
    Ptr<ml::ParamGrid> degreeGrid = ml::SVM::getDefaultGridPtr(ml::SVM::DEGREE);

//    cGrid->logStep = 0;
//    gammaGrid->logStep = 0;
    pGrid->logStep = 0;
    nuGrid->logStep = 0;
    coefGrid->logStep = 0;
    degreeGrid->logStep = 0;

    cGrid->logStep = 2;
    cGrid->minVal = 0.1;
    cGrid->maxVal = 1000;
    gammaGrid->logStep = 2;
    gammaGrid->minVal = 1e-5;
    gammaGrid->maxVal = 100;

    cout << "C grid - logStep =  " << cGrid->logStep << ", minVal =  " << cGrid->minVal << ", maxVal = " << cGrid->maxVal << endl;
    cout << "Gamma grid - logStep =  " << gammaGrid->logStep << ", minVal =  " << gammaGrid->minVal << ", maxVal = " << gammaGrid->maxVal << endl;

    cout << "trainData - " << trainData.rows << " x " << trainData.cols << endl;

//    svm->train(trainData, ml::ROW_SAMPLE, trainDataLabels);
    svm->trainAuto(trainData, ml::ROW_SAMPLE, trainDataLabels, 10,
                   cGrid, gammaGrid, pGrid, nuGrid, coefGrid, degreeGrid,
                   false);

    cout << "Trained C = " << svm->getC() << endl;
    cout << "Trained Gamma = " << svm->getGamma() << endl;
//    cout << "Trained P = " << svm->getP() << endl;
//    cout << "Trained Nu = " << svm->getNu() << endl;
//    cout << "Trained Coef0 = " << svm->getCoef0() << endl;
//    cout << "Trained Degree = " << svm->getDegree() << endl;

    svm->save("svmTrained.yml");
}

void PixelSVMClassifier::classify(Mat &inImage, Mat &outLabelImage)
{
    assert(inImage.data);
    assert(inImage.type() == CV_8UC3);

    Mat featureImage;
    featureCalculator->getFeatures(inImage, featureImage);
    Mat dataInImage = featureImage.reshape(1, inImage.total());
    PRINT_DEBUG("SVM PREDICTION START!");
    svm->predict(dataInImage, outLabelImage);
    PRINT_DEBUG("SVM PREDICTION END!");
    outLabelImage = outLabelImage.reshape(1, inImage.rows);
    outLabelImage.convertTo(outLabelImage, CV_32S);

    assert(outLabelImage.rows = inImage.rows);
    assert(outLabelImage.cols = inImage.cols);
    assert(dataInImage.rows == inImage.total());
    assert(dataInImage.cols == featureCalculator->getDimentions());
}
