#include "pixelClassifierFactory.hpp"
#include "pixClassifiers/pixelKNNClassifier.hpp"
#include "pixClassifiers/pixelSVMClassifier.hpp"
#include "pixClassifiers/pixelMahalanobisClassifier.hpp"
#include "pixClassifiers/pixelNNClassifier.hpp"
#include "pixClassifiers/pixelRandomForestClassifier.hpp"

using namespace std;
using namespace cv;

PixelClassifier* PixelClassifierFactory::create(string confFilePath, ImageFeatures* features)
{
    int useMahala = 1;
    int useSVM = 0;
    int useKNN = 0;
    int useNN = 0;
    int useRTree = 0;

    int mahalaOrder = 3;
    int svm_linear = 1;
    int svm_rbf = 0;
    int knn_k = 3;
    int rtree_n_tree = 50;
    int rtree_max_deep = 30;
    int rtree_max_categories = 16;

    FileStorage file(confFilePath, FileStorage::READ);
    file["useMAHALANOBIS"] >> useMahala;
    file["useSVM"] >> useSVM;
    file["useKNN"] >> useKNN;
    file["useNN"] >> useNN;
    file["useRTREE"] >> useRTree;

    file["SVM_LINEAR"] >> svm_linear;
    file["SVM_RBF"] >> svm_rbf;

    file["MAHALANOBIS_ORDER"] >> mahalaOrder;

    file["KNN_K"] >> knn_k;

    file["RTREE_N_TREE"] >> rtree_n_tree;
    file["RTREE_MAX_DEEP"] >> rtree_max_deep;
    file["RTREE_MAX_CATEGORIES"] >> rtree_max_categories;

    PixelClassifier* pixClassifier = 0;
    assert(useMahala || useSVM || useKNN || useNN || useRTree);
    if(useMahala){
        assert(mahalaOrder >= 1);
        pixClassifier = new PixelMahalanobisClassifier(features, mahalaOrder);
    }else if(useSVM){
        assert(svm_linear || svm_rbf);
        if(svm_linear)
            pixClassifier = new PixelSVMClassifier(features, PixelSVMClassifier::LINEAR);
        else if(svm_rbf)
            pixClassifier = new PixelSVMClassifier(features, PixelSVMClassifier::RBF);
    }else if(useKNN){
        assert(knn_k >= 1);
        pixClassifier = new PixelKNNClassifier(features, knn_k);
    }else if(useNN){
        pixClassifier = new PixelNNClassifier(features);
    }else if(useRTree){
        assert(rtree_n_tree >= 1 && rtree_max_deep >= 1 && rtree_max_categories >= 1);
        pixClassifier = new PixelRandomForestClassifier(features, rtree_n_tree, rtree_max_deep, rtree_max_categories);
    }
    return pixClassifier;
}
