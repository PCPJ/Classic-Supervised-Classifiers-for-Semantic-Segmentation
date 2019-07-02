#include "imgF1Score.hpp"

using namespace std;
using namespace cv;

ImgF1Score::ImgF1Score(int numberOfClass)
    : ImgSegPrecisionMeasure(numberOfClass)
{

}

ImgF1Score::~ImgF1Score()
{

}

vector<double> ImgF1Score::measure(const Mat& confusionMat)
{
    assert(confusionMat.data);
    assert(confusionMat.rows == confusionMat.cols);
    assert(confusionMat.rows == numberOfClass);
    assert(confusionMat.type() == CV_32SC1);

    double sumF1 = 0;
    vector<double> classF1;
    for(int c = 0; c < numberOfClass; c++){
        double tp = confusionMat.at<int>(c,c);
        double fn = cv::sum(confusionMat.col(c))[0] - tp;
        double fp = cv::sum(confusionMat.row(c))[0] - tp;
        double f1 = 0;
        if(tp != 0){
            double precision = tp/(tp+fp);
            double recall = tp/(tp+fn);
            f1 = 2.0 * (precision*recall) / (precision+recall);
        }
        classF1.push_back(f1);
        sumF1 += f1;
    }
    classF1.push_back(sumF1/numberOfClass);
    return classF1;
}

