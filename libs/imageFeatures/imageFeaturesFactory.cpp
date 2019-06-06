#include "imageFeaturesFactory.hpp"
#include "debug.h"

using namespace std;
using namespace cv;

ImageFeatures* ImageFeaturesFactory::create(string confFilePath)
{
    int useRGB = 1;
    int useEXG = 0;
    int useEXR = 0;
    int useEXGEXR = 0;
    int useGaborFilter = 0;
    int useGLCM = 0;
    uint glcmFeats = 1;
    int glcmWindowSize = 9;
    int glcmContrast = 1;
    int glcmDissimilarity = 0;
    int glcmHomogeneity = 0;
    int glcmEnergy = 0;
    int glcmEntropy = 0;
    int glcmMean = 0;
    int glcmStdDev = 0;
    int glcmCorrelation = 0;

    cv::FileStorage file(confFilePath, cv::FileStorage::READ);
    file["useRGB"] >> useRGB;
    file["useEXG"] >> useEXG;
    file["useEXR"] >> useEXR;
    file["useEXGEXR"] >> useEXGEXR;
    file["useGaborFilter"] >> useGaborFilter;
    file["useGLCM"] >> useGLCM;

    assert(useRGB || useEXG || useEXR || useEXGEXR || useGLCM);

    MultiFeature* multiF = new MultiFeature();
    if(useRGB){
        multiF->concatFeat(new RGBFeature());
    }
    if(useEXG){
        multiF->concatFeat(new ExGIndex());
    }
    if(useEXR){
        multiF->concatFeat(new ExRIndex);
    }
    if(useEXGEXR){
        multiF->concatFeat(new ExGExRIndex(ExGExRIndex::Subtraction));
    }
    if(useGLCM){
        file["glcmContrast"] >> glcmContrast;
        file["glcmDissimilarity"] >> glcmDissimilarity;
        file["glcmHomogeneity"] >> glcmHomogeneity;
        file["glcmEnergy"] >> glcmEnergy;
        file["glcmEntropy"] >> glcmEntropy;
        file["glcmMean"] >> glcmMean;
        file["glcmStdDev"] >> glcmStdDev;
        file["glcmCorrelation"] >> glcmCorrelation;

        file["glcmWindowSize"] >> glcmWindowSize;

        if(glcmContrast)
            glcmFeats |= GLCMFeature::Contrast;
        if(glcmDissimilarity)
            glcmFeats |= GLCMFeature::Dissimilarity;
        if(glcmHomogeneity)
            glcmFeats |= GLCMFeature::Homogeneity;
        if(glcmEnergy)
            glcmFeats |= GLCMFeature::Energy;
        if(glcmEntropy)
            glcmFeats |= GLCMFeature::Entropy;
        if(glcmMean)
            glcmFeats |= GLCMFeature::Mean;
        if(glcmStdDev)
            glcmFeats |= GLCMFeature::StdDev;
        if(glcmCorrelation)
            glcmFeats |= GLCMFeature::Correlation;
        vector<GLCM> glcms = {GLCM(GLCM::NORTH|GLCM::SOUTH),
                              GLCM(GLCM::EAST|GLCM::WEST)/*,
                              GLCM(GLCM::NORTH_EAST|GLCM::SOUTH_WEST),
                              GLCM(GLCM::NORTH_WEST|GLCM::SOUTH_EAST)*/};
//        vector<GLCM> glcms = {GLCM(GLCM::ALL)};
        multiF->concatFeat(new GLCMFeature(glcms, glcmWindowSize, glcmFeats));
    }

    ImageFeatures* features = multiF;

    if(useGaborFilter){
        vector<double> thetas;
        thetas.push_back(0 * M_PI/180.0);//here
//        thetas.push_back(15 * M_PI/180.0);
//        thetas.push_back(30 * M_PI/180.0);
//        thetas.push_back(45 * M_PI/180.0);
//        thetas.push_back(60 * M_PI/180.0);
//        thetas.push_back(75 * M_PI/180.0);
        thetas.push_back(90 * M_PI/180.0);//here
//        thetas.push_back(105 * M_PI/180.0);
//        thetas.push_back(120 * M_PI/180.0);
//        thetas.push_back(135 * M_PI/180.0);
//        thetas.push_back(150 * M_PI/180.0);
//        thetas.push_back(165 * M_PI/180.0);
        vector<int> waveLengths;
//        waveLengths.push_back(3);
        waveLengths.push_back(9);//here
        waveLengths.push_back(15);//here
//        waveLengths.push_back(21);
//        waveLengths.push_back(27);
//        waveLengths.push_back(33);
//        waveLengths.push_back(39);
//        waveLengths.push_back(51);
        GaborTexture gabor(thetas, waveLengths);
//        GaborTexture gabor(waveLengths, GaborTexture::CIRCULAR);
        features = new GaborFeature(gabor, features);
//        multiF->concatFeat(new GaborFeature(gabor, new LuminanceFeature()));
    }
    PRINT_DEBUG("Dimensions = %d", features->getDimentions());
}
