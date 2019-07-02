#include "pixelRandomForestClassifier.hpp"

#include "debug.h"

using namespace std;
using namespace cv;

PixelRandomForestClassifier::PixelRandomForestClassifier(ImageFeatures *featureCalculator, uint n_tree, uint max_deep, uint max_categorie)
    : PixelClassifier(featureCalculator), rf()
{
    rf = ml::RTrees::create();
    rf->setMaxDepth(max_deep);//A low value will likely underfit and conversely a high value will likely overfit.
    rf->setMinSampleCount(1);//minimum samples required at a leaf node for it to be split. A reasonable value is a small percentage of the total data e.g. 1%.
    rf->setActiveVarCount(0);//The size of the randomly selected subset of features at each tree node and that are used to find the best split(s).
                             //If you set it to 0 then the size will be set to the square root of the total number of features.
    rf->setMaxCategories(max_categorie);//Cluster possible values of a categorical variable into K <= maxCategories clusters to find a suboptimal split.
                                        //If a discrete variable, on which the training procedure tries to make a split, takes more than max_categories values,
                                        //the precise best subset estimation may take a very long time because the algorithm is exponential.
                                        //Instead, many decision trees engines (including ML) try to find sub-optimal split in this case by clustering all the
                                        //samples into maxCategories clusters that is some categories are merged together.
                                        //The clustering is applied only in n>2-class classification problems for categorical variables with N > max_categories possible values.
                                        //In case of regression and 2-class classification the optimal split can be found efficiently without employing clustering, thus the parameter is not used in these cases.
    rf->setCalculateVarImportance(true);//If true then variable importance will be calculated and then it can be retrieved by RTrees::getVarImportance.
    rf->setRegressionAccuracy(0);
    rf->setUseSurrogates(false);
    rf->setPriors(cv::Mat());

    rf->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, n_tree, 1e-6));//50 is the max number of trees.
}

PixelRandomForestClassifier::~PixelRandomForestClassifier()
{

}

void PixelRandomForestClassifier::setTrainData(Mat &sampleImage, vector<vector<Point2i> > &samplePixCords)
{
    assert(sampleImage.data);
    assert(sampleImage.type() == CV_8UC3);

    trainData.release();
    trainDataLabels.release();
    addTrainData(sampleImage, samplePixCords);
}

void PixelRandomForestClassifier::setTrainData(Mat &imageSample, Mat &imageLabel)
{
    assert(imageSample.type() == CV_8UC3);
    assert(imageLabel.type() == CV_32SC1);
    assert(imageSample.total() == imageLabel.total());

    trainData.release();
    trainDataLabels.release();
    addTrainData(imageSample, imageLabel);
}

void PixelRandomForestClassifier::addTrainData(Mat &sampleImage, vector<vector<Point2i> > &samplePixCords)
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


    cv::FileStorage file("rawTrainData.yml", cv::FileStorage::WRITE);
    file << "numberOfClasses" << (int)samplePixCords.size();
    int sampleCount = 0;
    for(int i = 0; i < samplePixCords.size(); i++){
        Mat trainDataLabelsI = Mat(newTrainDataLabels, Rect(0, sampleCount, 1, samplePixCords[i].size()));
        file << "samples"+to_string(i) << newTrainData(Rect(0, sampleCount, dimensions, samplePixCords[i].size()));//remove
        trainDataLabelsI.setTo(i);
        sampleCount += samplePixCords[i].size();
    }
    file.release();

    if(trainData.data){
        vconcat(trainData, newTrainData, trainData);
        vconcat(trainDataLabels, newTrainDataLabels, trainDataLabels);
    }else{
        trainData = newTrainData;
        trainDataLabels = newTrainDataLabels;
    }

    rf->setMinSampleCount(trainData.rows/100);//minimum samples required at a leaf node for it to be split. A reasonable value is a small percentage of the total data e.g. 1%.

    assert(sampleCount == samplesSize);
    assert(trainData.rows == trainDataLabels.rows);
    assert(trainData.cols == featureCalculator->getDimentions());
    assert(trainDataLabels.cols == 1);
}

void PixelRandomForestClassifier::addTrainData(Mat &imageSamples, Mat &imageLabels)
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

    rf->setMinSampleCount(trainData.rows/100);//minimum samples required at a leaf node for it to be split. A reasonable value is a small percentage of the total data e.g. 1%.

    assert(trainData.rows == trainDataLabels.rows);
    assert(trainData.cols == featureCalculator->getDimentions());
    assert(trainDataLabels.cols == 1);
}

void PixelRandomForestClassifier::train()
{
    PRINT_DEBUG("trainData - %d x %d", trainData.rows, trainData.cols);
    rf->train(trainData, ml::ROW_SAMPLE, trainDataLabels);
    rf->save("RTree.yml");
    Mat varImportance = rf->getVarImportance();
    cout << "RTree features importance = " << varImportance << endl;
}

void PixelRandomForestClassifier::classify(Mat &inImage, Mat &outLabelImage)
{
    assert(inImage.data);
    assert(inImage.type() == CV_8UC3);

    Mat featureImage;
    featureCalculator->getFeatures(inImage, featureImage);
    Mat dataInImage = featureImage.reshape(1, inImage.total());
    PRINT_DEBUG("RTree PREDICTION START!");
    rf->predict(dataInImage, outLabelImage);
    PRINT_DEBUG("RTree PREDICTION END!");
    outLabelImage = outLabelImage.reshape(1, inImage.rows);
    outLabelImage.convertTo(outLabelImage, CV_32S);

    assert(outLabelImage.rows = inImage.rows);
    assert(outLabelImage.cols = inImage.cols);
    assert(dataInImage.rows == inImage.total());
    assert(dataInImage.cols == featureCalculator->getDimentions());
}
