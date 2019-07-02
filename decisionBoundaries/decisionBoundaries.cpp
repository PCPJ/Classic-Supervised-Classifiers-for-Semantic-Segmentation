#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>

#include "pixClassifiers/pixelKNNClassifier.hpp"
#include "pixClassifiers/pixelMahalanobisClassifier.hpp"
#include "pixClassifiers/pixelSVMClassifier.hpp"
#include "pixClassifiers/pixelNNClassifier.hpp"
#include "pixClassifiers/pixelRandomForestClassifier.hpp"

#include "imageFeatures/imageFeatures.hpp"

using namespace cv;
using namespace std;

Mat drawMat;
bool pressed = false;
Size s(1000, 1000);

uint classCounter = 0;
vector<Vec3b> colors;
vector<Vec3b> colorsBack;
vector<vector<Point2i> > pointSamples;
vector<Mat> results;
vector<string> names;

class PositionFeature : public ImageFeatures{
public:
    PositionFeature(){dimensions = 2;}
    virtual ~PositionFeature(){}

    void getFeature(std::vector<Point2i> &pixCoordinates, std::vector<double *> &outFeatures){}
    void getFeature(std::vector<double *> &outFeatures){}

    void getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures){
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

        for(int i = 0; i < pixCoordinates.size(); i++){
            Point2f coord = pixCoordinates[i];
            Vec2f& featureI = outFeatures.at<Vec2f>(i, 0);
            featureI[0] = coord.x/image.cols;
            featureI[1] = coord.y/image.rows;
        }
    }
    void getFeatures(const Mat &image, Mat &outFeatureImage){
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
        for(int i = 0; i < image.rows; i++){
            for(int j = 0; j < image.cols; j++){
                Vec2f feature(j, i);
                feature[0] /= image.cols;
                feature[1] /= image.rows;
                outFeatureImage.at<Vec2f>(i, j) = feature;
            }
        }
    }
private:
};

void draw(int event, int x, int y, int flags, void* userdata){
    if(event == CV_EVENT_LBUTTONDOWN){
        pressed = true;
    }else if(event == CV_EVENT_LBUTTONUP){
        pressed = false;
    }else if(event == CV_EVENT_MOUSEMOVE && pressed){
        if(drawMat.at<Vec3b>(y, x) != Vec3b(0,0,0))
            return;
        pointSamples[classCounter].push_back(Point2i(x,y));
        cv::circle(drawMat, Point(x,y), 3, Scalar(colors[classCounter]), CV_FILLED);
    }
    imshow("draw", drawMat);
}

void readSamplesFile(string path){
    cv::FileStorage file(path, cv::FileStorage::READ);
    int numberOfClasses;
    file["numberOfClasses"] >> numberOfClasses;
    for(int i = 0; i < numberOfClasses; i++){
        Mat samplesI;
        file["samples"+to_string(i)] >> samplesI;
        for(int j = 0; j < samplesI.rows; j++){
            Vec2i s = samplesI.at<Vec2i>(j,0);
            pointSamples[i].push_back(Point2i(s[0],s[1]));
            cv::circle(drawMat, Point2i(s[0],s[1]), 3, Scalar(colors[i]), CV_FILLED);
        }
        pointSamples.push_back(vector<Point2i>());
        classCounter++;
    }
    imshow("draw", drawMat);
}

void saveSamplesFile(string path){
    cv::FileStorage file(path, cv::FileStorage::WRITE);
    file << "numberOfClasses" << (int)(classCounter+1);
    for(int n = 0; n < classCounter+1; n++){
        Mat samplesN = Mat(pointSamples[n].size(), 2, CV_32SC1);
        for(int i = 0; i < pointSamples[n].size(); i++){
            Point pointI = pointSamples[n][i];
            samplesN.at<Vec2i>(i,0) = Vec2i(pointI.x, pointI.y);
        }
        file << ("samples"+to_string(n)) << samplesN;
    }
}

void generateCirclesSamples(int n, float noise){
    float radius = (s.width/2.0)-50;
    float rStep = radius/n;
    float resolutions = 2*M_PI/1000.0;
    cv::RNG rng(1);
    for(int i = 0; i < n; i++){
        for(float angle = 0; angle < 2*M_PI; angle+=resolutions){
            Point2d p(cos(angle), sin(angle));
            p *= radius;
            p += Point2d(s.width/2.0, s.height/2.0);
            Point pi = p + Point2d(rng.gaussian(noise), rng.gaussian(noise));
            pointSamples[i].push_back(pi);
            cv::circle(drawMat, pi, 3, Scalar(colors[i]), CV_FILLED);
        }
        radius -= rStep;
        pointSamples.push_back(vector<Point2i>());
        classCounter++;
    }
    imshow("draw", drawMat);
}

void generateDoubleSpiralSamples(int n, float noise){

}

int main(int argc, char** argv){

    drawMat = Mat::zeros(s, CV_8UC3);
    pointSamples.push_back(vector<Point2i>());
    names.push_back("Mahala1");
    names.push_back("Mahala2");
    names.push_back("Mahala3");
    names.push_back("KNN3");
    names.push_back("KNN11");
    names.push_back("SVMLinear");
    names.push_back("SVMRBF");
    names.push_back("NN");
    names.push_back("RTree");
    results = vector<Mat>(names.size(), Mat::zeros(s, CV_8UC1));
    colors.push_back(Vec3b(0,255,0));
    colors.push_back(Vec3b(0,255,255));
    colors.push_back(Vec3b(0,0,255));
    colors.push_back(Vec3b(255,0,0));
    colors.push_back(Vec3b(255,0,255));
    colors.push_back(Vec3b(255,255,0));
    colorsBack.push_back(Vec3b(128,255,128));
    colorsBack.push_back(Vec3b(128,255,255));
    colorsBack.push_back(Vec3b(128,128,255));
    colorsBack.push_back(Vec3b(255,128,128));
    colorsBack.push_back(Vec3b(255,128,255));
    colorsBack.push_back(Vec3b(255,255,128));

    cv::namedWindow("draw");
    cv::setMouseCallback("draw", draw);
    imshow("draw", drawMat);
    cv::waitKey(1);

    if(argc > 1){
        string param(argv[1]);
        if(param == "circles")
            generateCirclesSamples(4, 10);
        readSamplesFile(param);
    }

    ImageFeatures* features;
    PositionFeature posFeatures;
    features = &posFeatures;

    vector<PixelClassifier*> classifiers;
    classifiers.push_back(new PixelMahalanobisClassifier(features, 1));
    classifiers.push_back(new PixelMahalanobisClassifier(features, 2));
    classifiers.push_back(new PixelMahalanobisClassifier(features, 3));
    classifiers.push_back(new PixelKNNClassifier(features, 3));
    classifiers.push_back(new PixelKNNClassifier(features, 11));
    classifiers.push_back(new PixelSVMClassifier(features, PixelSVMClassifier::Kernel::LINEAR));
    classifiers.push_back(new PixelSVMClassifier(features, PixelSVMClassifier::Kernel::RBF));
    classifiers.push_back(new PixelNNClassifier(features));
    classifiers.push_back(new PixelRandomForestClassifier(features, 50, 30, 16));

    while(1){
        char key = waitKey(10);
        if(key == 'q')
            break;
        else if(key == ' '){
            classCounter++;
            pointSamples.push_back(vector<Point2i>());
        }
        else if(key == 'c'){
            for(auto& classifier: classifiers){
                classifier->setTrainData(drawMat, pointSamples);
            }
            for(int i = 0; i < classifiers.size(); i++){
                cout << "Training " << names[i] << endl;
                classifiers[i]->train();
                cout << "Training " << names[i] << " Done!" << endl;
            }

            Mat out;
            for(int c = 0; c < classifiers.size(); c++){
                classifiers[c]->run(drawMat, out);
                for(int i = 0; i < drawMat.rows; i++){
                    for(int j = 0; j < drawMat.cols; j++){
                        int outLabel = out.at<int>(i, j);
                        results[c].at<Vec3b>(i, j) = colorsBack[outLabel];
                    }
                }
            }

            for(int i = 0; i < results.size(); i++){
                imshow("Result "+names[i], results[i]);
            }
            waitKey(1);
        }
        else if(key == 's'){
            imwrite("samples.png", drawMat);
            for(Mat& result : results){
                for(int i = 0; i < pointSamples.size(); i++){
                    for(const Point& p : pointSamples[i]){
                        cv::circle(result, p, 6, colors[i], CV_FILLED);
                        cv::circle(result, p, 6, Scalar(0,0,0), 2);
                    }
                }
            }
            for(int i = 0; i < results.size(); i++){
                imwrite("classification"+names[i]+".png", results[i]);
            }
            saveSamplesFile("samplesFileOut.yml");
        }
    }

}
