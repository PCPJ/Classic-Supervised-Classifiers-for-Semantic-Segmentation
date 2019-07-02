#include "imgIOU.hpp"

using namespace std;
using namespace cv;

ImgIOU::ImgIOU(int numberOfClass)
    : ImgSegPrecisionMeasure(numberOfClass)
{

}

ImgIOU::~ImgIOU()
{

}

std::vector<double> ImgIOU::measure(const Mat &confusionMat)
{
    assert(confusionMat.data);
    assert(confusionMat.rows == confusionMat.cols);
    assert(confusionMat.rows == numberOfClass);
    assert(confusionMat.type() == CV_32SC1);

    double sumIOU = 0;
    vector<double> classIOU;
    for(int c = 0; c < numberOfClass; c++){
        double tp = confusionMat.at<int>(c,c);
        double fn = cv::sum(confusionMat.col(c))[0] - tp;
        double fp = cv::sum(confusionMat.row(c))[0] - tp;
        double iou = 0;
        if(tp != 0)
            iou = tp/(tp + fn + fp);
        classIOU.push_back(iou);
        sumIOU += iou;
    }
    classIOU.push_back(sumIOU/numberOfClass);
    return classIOU;
}

