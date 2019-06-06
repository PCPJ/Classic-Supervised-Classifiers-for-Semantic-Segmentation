#include "imageFeatures/multiFeatures.hpp"
#include "imageFeatures/filterFeature.hpp"
#include "imageFeatures/rgbFeature.hpp"
#include "imageFeatures/luminanceFeatures.hpp"
#include "imageFeatures/vegetationIndex/exgIndex.hpp"
#include "imageFeatures/vegetationIndex/exrIndex.hpp"
#include "imageFeatures/vegetationIndex/exGExRIndex.hpp"
#include "imageFeatures/rgbVegetationIndex.hpp"
#include "imageFeatures/gaborFeatures.hpp"
#include "imageFeatures/glcm.hpp"
#include "imageFeatures/glcmFeatures.hpp"

using namespace std;
using namespace cv;

class ImageFeaturesFactory{
public:
    static ImageFeatures* create(string confFilePath);
};
