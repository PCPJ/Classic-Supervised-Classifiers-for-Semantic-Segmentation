#include <iostream>
#include <memory>
#include <set>
#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>

#include "precisionMeasures/imgIOU.hpp"
#include "precisionMeasures/imgF1Score.hpp"

#include "utils/util.hpp"
#include "debug.h"

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem::v1;

boost::program_options::options_description desc("Allowed options");
boost::program_options::variables_map vm;

string predPath;
string gtPath;
string colorClassesFilePath;
vector<pair<string, string> > predGtPathVec;
Mat classesColors;
string outPath;

void mountParams(int argc, char **argv);
void readParams();
void readPaths();
void readColorClassesFile();

Mat operator ==(const Mat& img, const Vec3b& color){
    assert(img.type() == CV_8UC3);
    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    for(int i = 0; i < img.rows; i++){
        const Vec3b* imgData = img.ptr<Vec3b>(i);
        uchar* maskData = mask.ptr<uchar>(i);
        for(int j = 0; j < img.cols; j++){
            if(imgData[j] == color)
                maskData[j] = 255;
        }
    }
    return mask;
}

int main(int argc, char** argv){
    mountParams(argc, argv);
    readParams();
    readPaths();
    readColorClassesFile();

//    system("echo start >> date.txt && date >> date.txt");

    int numberOfClasses = classesColors.rows;
    ImgIOU iou(numberOfClasses);
    ImgF1Score f1(numberOfClasses);

    Mat confusionMat = Mat::zeros(numberOfClasses, numberOfClasses, CV_32SC1);
    Mat confusionMatFloat = Mat::zeros(numberOfClasses, numberOfClasses, CV_32FC1);
    for(const auto& predGtPathPair : predGtPathVec){
        Mat predImg = imread(predGtPathPair.first);
        Mat gtImg = imread(predGtPathPair.second);
        if(!predImg.data){
            PRINT_DEBUG("ERROR WHILE OPENING IMAGE: %s", predGtPathPair.first);
            exit(-1);
        }
        if(!gtImg.data){
            PRINT_DEBUG("ERROR WHILE OPENING IMAGE: %s", predGtPathPair.second);
            exit(-1);
        }
        Mat predIntLabel = Mat(predImg.size(), CV_32SC1, Scalar(-1));
        Mat gtIntLabel   = Mat(gtImg.size(), CV_32SC1, Scalar(-1));
        for(int i = 0; i < classesColors.rows; i++){
            Mat maskPred = predImg == classesColors.at<Vec3b>(i, 0);
            Mat maskGt = gtImg == classesColors.at<Vec3b>(i, 0);
            predIntLabel.setTo(i, maskPred);
            gtIntLabel.setTo(i, maskGt);
        }
        iou.fillConfusionMat(predIntLabel, gtIntLabel, confusionMat);
    }
    vector<double> iouVec  = iou.measure(confusionMat);
    vector<double> f1Vec   = f1.measure(confusionMat);
    confusionMat.convertTo(confusionMatFloat, CV_32F);
    confusionMatFloat /= cv::sum(confusionMat)[0];

    cv::FileStorage outFile(outPath, cv::FileStorage::WRITE);
    for(int c = 0; c < numberOfClasses; c++)
        outFile << "IOU_Class_"+to_string(c) << iouVec[c];
    outFile << "IOU_Mean" << iouVec[numberOfClasses];
    for(int c = 0; c < numberOfClasses; c++)
        outFile << "F1_Class_"+to_string(c) << f1Vec[c];
    outFile << "F1_Mean" << f1Vec[numberOfClasses];
    outFile.writeComment("--------------");
    outFile << "confusionMat" << confusionMat;
    outFile << "confusionMatPercent" << confusionMatFloat;
    outFile.release();
    PRINT_DEBUG("Saving %s", outPath.c_str());
//    system("echo end >> date.txt && date >> date.txt");
}

void mountParams(int argc, char** argv){
    namespace po = boost::program_options;
    desc.add_options()
            ("help", "describe arguments")
            ("pred", po::value<string>(), "path to predictions image")
            ("gt", po::value<string>(), "path to gt images")
            ("colors", po::value<string>(), "path to classes colors conf file")
            ("out", po::value<string>(), "output file name");

    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
}

void readParams(){
    if(vm.count("help")){
        cout << desc << endl;
        exit(0);
    }
    if(!vm.count("pred")){
        cout << "Missing predictions path" << endl;
        exit(-1);
    }
    if(!vm.count("gt")){
        cout << "Missing gt path" << endl;
        exit(-1);
    }
    if(!vm.count("colors")){
        cout << "Missing classes colors conf file" << endl;
        exit(-1);
    }
    predPath = vm["pred"].as<string>();
    gtPath = vm["gt"].as<string>();
    outPath = vm.count("out") ? vm["out"].as<string>() : "out.yml";
    colorClassesFilePath = vm["colors"].as<string>();
}

void readPaths(){
    vector<string> predPaths;
    vector<string> gtPaths;
    for(auto& p : fs::directory_iterator(predPath)){
        predPaths.push_back(p.path());
    }
    for(auto& p : fs::directory_iterator(gtPath)){
        gtPaths.push_back(p.path());
    }
    assert(predPaths.size() == gtPaths.size());
    std::sort(predPaths.begin(), predPaths.end());
    std::sort(gtPaths.begin(), gtPaths.end());
    for(int i = 0; i < predPaths.size(); i++){
        predGtPathVec.push_back(make_pair(predPaths[i], gtPaths[i]));
    }
}

void readColorClassesFile(){
    cv::FileStorage file(colorClassesFilePath, cv::FileStorage::READ);
    file["colors"] >> classesColors;
}
