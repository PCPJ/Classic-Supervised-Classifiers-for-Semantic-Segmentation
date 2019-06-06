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
        Mat thetas;
        Mat waveLengths;
        file["gaborOrients"] >> thetas;
        file["gaborWaveLengths"] >> waveLengths;
        GaborTexture gabor(thetas, waveLengths);
//        GaborTexture gabor(waveLengths, GaborTexture::CIRCULAR);
        features = new GaborFeature(gabor, features);
    }
    PRINT_DEBUG("Dimensions = %d", features->getDimentions());
}
