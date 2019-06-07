#include "pixelClassifier.hpp"
#include "debug.h"

#include <omp.h>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem::v1;

PixelClassifier::PixelClassifier(ImageFeatures* calculator)
    : featureCalculator(calculator)
{
}

PixelClassifier::~PixelClassifier()
{

}

bool PixelClassifier::loadTrainData(string trainDataConfFilePath, string colorsConfFilePath)
{
    FileStorage trainDataFile(trainDataConfFilePath, FileStorage::READ);
    FileStorage colorsConfFile(colorsConfFilePath, FileStorage::READ);
    if(!trainDataFile.isOpened()){
        PRINT_DEBUG("Can't open file %s", trainDataConfFilePath);
        return false;
    }
    if(!colorsConfFile.isOpened()){
        PRINT_DEBUG("Can't open file %s", colorsConfFilePath);
        return false;
    }
    Mat colors;
    colorsConfFile["colors"] >> colors;

    int numberOfClasses;
    string gtImagesPath;
    string gtPath;
    string annotationsImagesPath;
    string annotationsPath;
    trainDataFile["numberOfClasses"] >> numberOfClasses;
    trainDataFile["gtImagesPath"] >> gtImagesPath;
    trainDataFile["gtPath"] >> gtPath;
    trainDataFile["annotationsImagesPath"] >> annotationsImagesPath;
    trainDataFile["annotationsPath"] >> annotationsPath;

    PRINT_DEBUG("\n\tNumber of classes = %d \n\tGT original images path = %s \n\tGT images path = %s"
                "\n\tAnnotations original images path = %s \n\tAnnotations images path = %s",
                numberOfClasses, gtImagesPath.c_str(), gtPath.c_str(), annotationsImagesPath.c_str(), annotationsPath.c_str());

    vector<pair<fs::path, fs::path> > gtPathsVector;
    vector<pair<fs::path, fs::path> > annotationsPathsVector;
    if(gtPath != "" && gtImagesPath != ""){
        for(const fs::directory_entry& entry : fs::directory_iterator(gtPath)){
            if(entry.status().type() != fs::file_type::directory){
                if(entry.path().extension() == ".png" || entry.path().extension() == ".jpg"){
                    fs::path originalImagePath = gtImagesPath;
                    originalImagePath /= entry.path().filename();
                    if(fs::exists(originalImagePath)){
                        gtPathsVector.push_back(make_pair(originalImagePath, entry.path()));
                    }
                }
            }
        }
    }
    if(annotationsPath != "" && annotationsImagesPath != ""){
        for(const fs::directory_entry& entry : fs::directory_iterator(annotationsPath)){
            if(entry.status().type() != fs::file_type::directory){
                if(entry.path().extension() == ".png" || entry.path().extension() == ".jpg"){
                    fs::path originalImagePath = annotationsImagesPath;
                    originalImagePath /= entry.path().filename();
                    if(fs::exists(originalImagePath)){
                        annotationsPathsVector.push_back(make_pair(originalImagePath, entry.path()));
                    }
                }
            }
        }
    }

    for(const auto& paths : gtPathsVector){
        Mat image = imread(paths.first.string());
        Mat gt = imread(paths.second.string());
        assert(image.data);
        assert(gt.data);
        assert(image.size() == gt.size());
        assert(image.type() == CV_8UC3);
        assert(gt.type() == CV_8UC3);
        Mat label = Mat(image.size(), CV_32SC1);
        for(int i = 0; i < image.rows; i++){
            for(int j = 0; j < image.cols; j++){
                int l = -1;
                for(int c = 0; c < colors.rows; c++){
                    if(gt.at<Vec3b>(i,j) == colors.at<Vec3b>(c,0)){
                        l = c;
                        break;
                    }
                }
                assert(l != -1 && "GT with unknown color.");
                label.at<int>(i,j) = l;
            }
        }
        addTrainData(image, label);
    }

    for(const auto& paths : annotationsPathsVector){
        Mat image = imread(paths.first.string());
        Mat annotation = imread(paths.second.string());
        assert(image.data);
        assert(annotation.data);
        assert(image.size() == annotation.size());
        assert(image.type() == CV_8UC3);
        assert(annotation.type() == CV_8UC3);
        vector<vector<Point> > coords(numberOfClasses);
        for(int i = 0; i < image.rows; i++){
            for(int j = 0; j < image.cols; j++){
                for(int c = 0; c < colors.rows; c++){
                    if(annotation.at<Vec3b>(i,j) == colors.at<Vec3b>(c,0)){
                        coords[c].push_back(Point(j,i));
                        break;
                    }
                }
            }
        }
        addTrainData(image, coords);
    }
}

void PixelClassifier::run(Mat &inImage, Mat &outLabelImage)
{
    assert(inImage.data);
    assert(inImage.type() == CV_8UC3);

    if(outLabelImage.data){
        assert(outLabelImage.size() == inImage.size());
        assert(outLabelImage.type() == CV_32SC1);
    }else{
        outLabelImage = Mat(inImage.size(), CV_32SC1, Scalar(-1));
    }

    static const long int maxMemory = 8L * 1024L * 1024L * 1024L;//bytes

    long int inImageRows = inImage.rows;
    long int inImageCols = inImage.cols;
    long int numberOfFeatures = featureCalculator->getDimentions();
    long int totalValues = inImageRows * inImageCols * numberOfFeatures;
    long int totalMemory =  totalValues * (long int)sizeof(float);

    if(totalValues > std::numeric_limits<int>::max() || totalMemory > maxMemory){
        //Process window by window.
        static const int windowSize = sqrt(std::numeric_limits<int>::max()/((float)numberOfFeatures));
        Mat outWindow;
        for(int i = 0; i < inImage.rows; i+=windowSize){
            for(int j = 0; j < inImage.cols; j+=windowSize){
                Point rectPoint(j, i);
                Size rectSize(std::min(windowSize, inImage.cols-j), std::min(windowSize, inImage.rows-i));
                Rect windowRect(rectPoint, rectSize);
                Mat inWindow = inImage(windowRect);
                PRINT_DEBUG("CLASSIFYING THE WINDOW %dx%d-%dx%d", windowRect.x, windowRect.y, windowRect.width, windowRect.height);
                classify(inWindow, outWindow);
                outWindow.copyTo(outLabelImage(windowRect));
                PRINT_DEBUG("CLASSIFYING WINDOW DONE!");
            }
        }
    }else{
        classify(inImage, outLabelImage);
    }

}


