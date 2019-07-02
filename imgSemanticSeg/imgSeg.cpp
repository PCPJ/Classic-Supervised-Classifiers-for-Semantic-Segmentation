#include <iostream>
#include <memory>
#include <set>
#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>

#include "pixClassifiers/pixelClassifierFactory.hpp"
#include "pixClassifiers/imageSemanticSegmenter.hpp"
#include "imageFeatures/imageFeaturesFactory.hpp"

#include "utils/util.hpp"
#include "debug.h"

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem::v1;

boost::program_options::options_description desc("Allowed options");
boost::program_options::variables_map vm;

Size interfaceSize;
double interfaceScaleFactor;

string inPaths;
vector<fs::path> inImagesPaths;
string outImagePath = "out/";
uint numberOfClasses;
string classifierConfFilePath;
string featuresConfFilePath;
string trainDataFilePath;
string colorClassesFilePath;

ImageSemanticSegmenter* mapper;
PixelClassifier* pixClassifier;
ImageFeatures* features;

Mat classesColors;

void mountParams(int argc, char **argv);
void readParams();
void readInImagesPaths();
void readColorClassesFile();
void setUpImage();

int main(int argc, char** argv){
    mountParams(argc, argv);
    readParams();

    readInImagesPaths();
    readColorClassesFile();

    features = ImageFeaturesFactory::create(featuresConfFilePath);
    pixClassifier = PixelClassifierFactory::create(classifierConfFilePath, features);

    PRINT_DEBUG("Loading Train Data!");
    pixClassifier->loadTrainData(trainDataFilePath, colorClassesFilePath);
    PRINT_DEBUG("Start Training!");
    pixClassifier->train();
    PRINT_DEBUG("Finished Training!");
    mapper = new ImageSemanticSegmenter(pixClassifier);

    system("echo start >> date.txt && date >> date.txt");

    Mat inImage;
    Mat labelImage;
    Mat colorLabelImage;
    for(fs::path& inImagePath : inImagesPaths){

        inImage = imread(inImagePath.c_str());
        if(!inImage.data){
            PRINT_DEBUG("ERROR WHILE OPENING IMAGE: %s", inImagePath.c_str());
            exit(-1);
        }
        PRINT_DEBUG("Image Opened.");
        PRINT_DEBUG("Input image size = %dx%d", inImage.cols, inImage.rows);

        mapper->doSegmentation(inImage, labelImage);
        labelImage.convertTo(colorLabelImage, CV_8UC1);
        cvtColor(colorLabelImage, colorLabelImage, COLOR_GRAY2BGR);
        for(int i = 0; i < inImage.rows; i++){
            for(int j = 0; j < inImage.cols; j++){
                int label = labelImage.at<int>(i, j);
                assert(label >= 0 && label < numberOfClasses);
                colorLabelImage.at<Vec3b>(i, j) = classesColors.at<Vec3b>(label, 0);
            }
        }
        string outPath(outImagePath+inImagePath.stem()+"out.png");
        imwrite(outPath, colorLabelImage);
        PRINT_DEBUG("Saved Image = %s", outPath.c_str());
    }

    system("echo end >> date.txt && date >> date.txt");
}

void mountParams(int argc, char** argv){
    namespace po = boost::program_options;
    desc.add_options()
            ("help", "describe arguments")
            ("in", po::value<string>(), "input file name")
            ("out", po::value<string>(), "output file name")
            ("n", po::value<unsigned int>(), "number of classes")
            ("classifier", po::value<string>(), "path to classifier conf file")
            ("features", po::value<string>(), "path to features conf file")
            ("tdata", po::value<string>(), "path to train data conf file")
            ("colors", po::value<string>(), "path to classes colors conf file");

    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
}

void readParams(){
    if(vm.count("help")){
        cout << desc << endl;
        exit(0);
    }
    if(!vm.count("in")){
        cout << "Missing input file" << endl;
        exit(-1);
    }
    if(!vm.count("n")){
        cout << "Missing number of classes" << endl;
        exit(-1);
    }
    if(!vm.count("colors")){
        cout << "Missing classes colors conf file" << endl;
        exit(-1);
    }
    if(!vm.count("features")){
        cout << "Missing features conf file" << endl;
        exit(-1);
    }
    if(!vm.count("classifier")){
        cout << "Missing classifier conf file" << endl;
        exit(-1);
    }
    if(!vm.count("tdata")){
        cout << "Missing train data conf file" << endl;
        exit(-1);
    }
    inPaths = vm["in"].as<string>();
    outImagePath = vm.count("out") ? vm["out"].as<string>() : "out/";
    numberOfClasses = vm["n"].as<unsigned int>();
    classifierConfFilePath = vm["classifier"].as<string>();
    featuresConfFilePath = vm["features"].as<string>();
    colorClassesFilePath = vm["colors"].as<string>();
    trainDataFilePath = vm.count("tdata") ? vm["tdata"].as<string>() : "";
}

void readInImagesPaths(){
    fs::path path(inPaths);
    vector<fs::path> paths;
    if(fs::status(path).type() == fs::file_type::directory){
        paths.push_back(path);
    }else{
        assert(path.extension() == ".png" || path.extension() == ".tif" || path.extension() == ".jpg" || path.extension() == ".yml");
        if(path.extension() == ".yml"){
            cv::FileStorage ymlFile(inPaths, cv::FileStorage::READ);
            assert(ymlFile.isOpened());
            int numberOfPaths;
            ymlFile["numberOfPaths"] >> numberOfPaths;
            for(int i = 0; i < numberOfPaths; i++){
                string pathI;
                ymlFile["path"+to_string(i)] >> pathI;
                paths.push_back(fs::path(pathI));
            }
        }
        else{
            inImagesPaths.push_back(fs::path(inPaths));
        }
    }
    for(const fs::path& p : paths){
        for(const fs::directory_entry& entry : fs::directory_iterator(p)){
            if(entry.status().type() != fs::file_type::directory){
                if(entry.path().extension() == ".png" || entry.path().extension() == ".jpg"){
                    inImagesPaths.push_back(entry.path());
                }
            }
        }
    }
}

void readColorClassesFile(){
    cv::FileStorage file(colorClassesFilePath, cv::FileStorage::READ);
    file["colors"] >> classesColors;
    assert(classesColors.rows == numberOfClasses);
}
