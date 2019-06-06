#pragma once

#include "pixelClassifier.hpp"
#include "imageFeatures/imageFeatures.hpp"

using namespace std;
using namespace cv;

class PixelClassifierFactory{
public:
    static PixelClassifier* create(string confFilePath, ImageFeatures* features);
};
