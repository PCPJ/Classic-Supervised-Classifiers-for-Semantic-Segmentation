#include <opencv2/opencv.hpp>
#include "gabor_texture.hpp"
#include "debug.h"
#include <omp.h>

using namespace std;
using namespace cv;

GaborTexture::GaborTexture(uchar lambdaMethod)
{

    string debugKernelPath = "gaborKernels";

//    int width = 1000;
    int width = 256;

    //compute lambda values
    this->lambdaMethod = lambdaMethod;
    if(lambdaMethod == LAMBDA_JAIN){
        computeLambdaJain(width);
        debugKernelPath += "JAIN";
    }
    else if(lambdaMethod == LAMBDA_ZHANG){
        computeLambdaJZhang(width);
        debugKernelPath += "ZHANG";
    }

    float gamma = 1;
    float b = 1;
    float phi = 0; //pi/2

    directions=6;
    float acum=0;
    for(int i=0; i<directions; i++)
    {
            thetas.push_back(acum);
            acum+=M_PI/directions;
    }


    if(DEBUG){
        for(int i = 0; i < lambdas.size(); i++){
            cout << "lambda " << i << " = " << lambdas[i] << endl;
        }
        for(int i = 0; i < thetas.size(); i++){
            cout << "theta " << i << " = " << thetas[i] << endl;
        }
    }

    kernelMaxSize = std::numeric_limits<int>::min();
    for(int i = 0; i < lambdaSize; i++){
        for(int j = 0; j < directions; j++){
            Mat kernel = createGaborKernel(lambdas[i], thetas[j], gamma, phi, b);
            kernels.push_back(kernel);
            Mat smoothKernel = createSmoothKernel(b, lambdas[i]);
            smoothKernels.push_back(smoothKernel);
            kernelMaxSize = std::max(kernelMaxSize, kernel.rows);
        }
    }

    kernelMaxSize = kernelMaxSize;

    if(DEBUG)
        saveKernels(debugKernelPath);

}

GaborTexture::GaborTexture(const Mat &inImage, uchar lambdaMethod)
{
    string debugKernelPath = "gaborKernels";

//    int width = inImage.cols;
//    int width = 1000;
    int width = 256;

    //compute lambda values
    this->lambdaMethod = lambdaMethod;
    if(lambdaMethod == LAMBDA_JAIN){
        computeLambdaJain(width);
        debugKernelPath += "JAIN";
    }
    else if(lambdaMethod == LAMBDA_ZHANG){
        computeLambdaJZhang(width);
        debugKernelPath += "ZHANG";
    }

    float gamma = 1;
    float b = 1;
    float phi = 0; //pi/2

    directions=6;
    float acum=0;
    for(int i=0; i<directions; i++)
    {
            thetas.push_back(acum);
            acum+=M_PI/directions;
    }


    if(DEBUG){
        for(int i = 0; i < lambdas.size(); i++){
            cout << "lambda " << i << " = " << lambdas[i] << endl;
        }
        for(int i = 0; i < thetas.size(); i++){
            cout << "theta " << i << " = " << thetas[i] << endl;
        }
    }

    kernelMaxSize = std::numeric_limits<int>::min();
    for(int i = 0; i < lambdaSize; i++){
        for(int j = 0; j < directions; j++){
            Mat kernel = createGaborKernel(lambdas[i], thetas[j], gamma, phi, b);
            kernels.push_back(kernel);
            Mat smoothKernel = createSmoothKernel(b, lambdas[i]);
            smoothKernels.push_back(smoothKernel);
            kernelMaxSize = std::max(kernelMaxSize, kernel.rows);
        }
    }

    kernelMaxSize = kernelMaxSize;

    if(DEBUG)
        saveKernels(debugKernelPath);
}

GaborTexture::GaborTexture(std::vector<double> thetas, uchar lambdaMethod)
{

    string debugKernelPath = "gaborKernels";

//    int width = inImage.cols;
//    int width = 1000;
    int width = 256;

    //compute lambda values
    this->lambdaMethod = lambdaMethod;
    if(lambdaMethod == LAMBDA_JAIN){
        computeLambdaJain(width);
        debugKernelPath += "JAIN";
    }
    else if(lambdaMethod == LAMBDA_ZHANG){
        computeLambdaJZhang(width);
        debugKernelPath += "ZHANG";
    }

    float gamma = 1;
    float b = 1;
    float phi = 0; //pi/2

    directions = thetas.size();
    this->thetas = thetas;


    if(DEBUG){
        for(int i = 0; i < lambdas.size(); i++){
            cout << "lambda " << i << " = " << lambdas[i] << endl;
        }
        for(int i = 0; i < thetas.size(); i++){
            cout << "theta " << i << " = " << thetas[i] << endl;
        }
    }

    kernelMaxSize = std::numeric_limits<int>::min();
//    for(int i = 0; i < lambdaSize; i++){
//        for(int j = 0; j < directions; j++){
//            Mat kernel = createGaborKernel(lambdas[i], thetas[j], gamma, phi, b);
//            kernels.push_back(kernel);
//            Mat smoothKernel = createSmoothKernel(b, lambdas[i]);
//            smoothKernels.push_back(smoothKernel);
//            kernelMaxSize = std::max(kernelMaxSize, kernel.rows);
//        }
//    }
    static vector<int> kSizes = {101};
    for(int i = 0; i < kSizes.size(); i++){
        for(int j = 0; j < directions; j++){
            Mat kernel = cv::getGaborKernel(Size(kSizes[i], kSizes[i]), kSizes[i], thetas[j], kSizes[i], gamma, phi, CV_32F);
            kernels.push_back(kernel);
            kernelMaxSize = std::max(kernelMaxSize, kernel.rows);
        }
    }

    kernelMaxSize = kernelMaxSize;

    if(DEBUG)
        saveKernels(debugKernelPath);
}

GaborTexture::GaborTexture(cv::Mat& thetas, cv::Mat& waveLengths)
{
    assert(thetas.data && waveLengths.data);
    assert(thetas.type() == CV_32FC1);
    assert(waveLengths.type() == CV_32FC1);

    string debugKernelPath = "gaborKernels";

    float gamma = 1.0;
    float b = 1;
    float phi = 0; //pi/2

    directions = thetas.rows;
    for(int i = 0; i < thetas.rows; i++)
        this->thetas.push_back(thetas.at<float>(i,0));
    for(int i = 0; i < waveLengths.rows; i++)
        this->waveLengths.push_back(waveLengths.at<float>(i,0));

    if(DEBUG){
        for(int i = 0; i < waveLengths.rows; i++){
            cout << "Wave lengths " << i << " = " << waveLengths.at<float>(i,0) << endl;
        }
        for(int i = 0; i < thetas.rows; i++){
            cout << "theta " << i << " = " << thetas.at<float>(i,0) << endl;
        }
    }

    float ratioSigLamb = (1.0/M_PI) * 0.588705011257737 * ((pow(2.0,(double)b)+1) / (pow(2.0,(double)b)-1));

    kernelMaxSize = std::numeric_limits<int>::min();
    for(int i = 0; i < waveLengths.rows; i++){
        int size = this->waveLengths[i]*4 + 1;
        double sigma = ratioSigLamb * this->waveLengths[i];
        for(int j = 0; j < directions; j++){
            Mat kernel = cv::getGaborKernel(Size(size, size), sigma, this->thetas[j], this->waveLengths[i], gamma, phi, CV_32F);
            Mat gausKernel = cv::getGaussianKernel(sigma*6+1, sigma*2, CV_32F);
            gausKernel = gausKernel * gausKernel.t();
            kernels.push_back(kernel);
            smoothKernels.push_back(gausKernel);
            kernelMaxSize = std::max(kernelMaxSize, kernel.rows);
        }
    }

    if(DEBUG)
        saveKernels(debugKernelPath);
}

GaborTexture::GaborTexture(std::vector<int> waveLengths, uchar lambdaMethod)
{
    string debugKernelPath = "gaborKernels";
    float b = 1;
    float phi = 0; //pi/2
    if(DEBUG){
        for(int i = 0; i < waveLengths.size(); i++){
            cout << "Wave Lengths " << i << " = " << waveLengths[i] << endl;
        }
    }
//    int f = 500;
//    int d = 500;
//    int size = 1000;
//    size += size%2==0 ? 1 : 0;
//    cv::namedWindow("kernel");
//    cv::createTrackbar("freq", "kernel", &f, 10000);
//    cv::createTrackbar("dev", "kernel", &d, 10000);
//    Mat kernel = Mat(size,size, CV_32FC1);
//    Mat kernelShow = Mat(size,size, CV_8UC1);
//    while(1){
//        float freq = (f+1)/10.0;
//        float dev = (d+1)/10.0;
//        cout << "freq = " << freq << endl;
//        cout << "dev = " << dev << endl;
//        for(int y = -size/2; y <= size/2; y++){
//            for(int x = -size/2; x <= size/2; x++){
//                float sigmoidVal = cos((1.0/freq) *2*M_PI * sqrt(x*x + y*y));
//                float gausVal = pow(M_E, -((x*x + y*y)/(2*dev*dev)))/(2*M_PI*dev*dev);
//                float result = sigmoidVal*gausVal;
//                kernel.at<float>(y+(size/2), x+(size/2)) = result;
//            }
//        }
//        cv::normalize(kernel, kernelShow, 0, 255, cv::NORM_MINMAX, CV_8U);
//        cv::imshow("kernel", kernelShow);
//        char key = cv::waitKey(1);
//        if(key == 'q')
//            exit(1);
//    }

    //              sqrt(2 ln(e))/2pi        (2^b + 1) / (2^b - 1)
    float sigLamb = 0.1873906251292776 * ((pow(2,b)+1)/(pow(2,b)-1));
    kernelMaxSize = std::numeric_limits<int>::min();
    for(int i = 0; i < waveLengths.size(); i++){
        int size = waveLengths[i]*3;
        size += size%2==0 ? 1 : 0;
        float sigma = sigLamb * waveLengths[i];
        Mat kernel = Mat(size,size, CV_32FC1);
        for(int y = -size/2; y <= size/2; y++){
            for(int x = -size/2; x <= size/2; x++){
                float sigmoidVal = cos((1.0/waveLengths[i]) * 2*M_PI * sqrt(x*x + y*y));
                float gausVal = pow(M_E, -((x*x + y*y)/(2*sigma*sigma)))/(2*M_PI*sigma*sigma);
                float result = sigmoidVal*gausVal;
                kernel.at<float>(y+(size/2), x+(size/2)) = result;
            }
        }
        Mat gausKernel = cv::getGaussianKernel(sigma*10+1, sigma*2, CV_32F);
        gausKernel = gausKernel * gausKernel.t();
        kernels.push_back(kernel);
        smoothKernels.push_back(gausKernel);
        kernelMaxSize = std::max(kernelMaxSize, kernel.rows);
    }

    if(DEBUG)
        saveKernels(debugKernelPath);
}

GaborTexture::GaborTexture(const GaborTexture &gabor)
    : lambdaMethod(gabor.lambdaMethod),
      lambdaSize(gabor.lambdaSize),
      directions(gabor.directions),
      thetas(gabor.thetas),
      lambdas(gabor.lambdas),
      kernels(gabor.kernels),
      smoothKernels(gabor.smoothKernels)
{

}

GaborTexture::~GaborTexture()
{

}

int GaborTexture::kernelsSize() const
{
    return kernels.size();
}

void GaborTexture::filterKernelListByDirection(vector<int>& directionsToKeep)
{
    vector<Mat> newKernelsList;
    newKernelsList.reserve(kernels.size());

    for(int i = 0; i < kernels.size(); i++){
        int mod = i % directions;
        for(int& d : directionsToKeep){
            if(mod == d){
                newKernelsList.push_back(kernels[i]);
                break;
            }
        }
    }

    directions = directionsToKeep.size();
    kernels = std::move(newKernelsList);
}

void GaborTexture::saveKernels(string path)
{
    for(int k = 0; k < kernelsSize(); k++){
        Mat kchar = Mat(kernels[k].size(), CV_8UC1);
        cv::normalize(kernels[k], kchar, 0, 255, cv::NORM_MINMAX, CV_8U);
//        for(int i = 0; i < kernels[k].rows; i++){
//            for(int j = 0; j < kernels[k].cols; j++){
//                double val = kernels[k].at<float>(i, j);
//                assert(val >= -1 && val <= 1);
//                kchar.at<uchar>(i, j) = (val -(-1))/(1-(-1))*255;
//            }
//        }
        imwrite(path+"/gabor"+to_string(k)+".png", kchar);
    }
    for(int k = 0; k < kernelsSize(); k++){
        Mat kchar = Mat(smoothKernels[k].size(), CV_8UC1);
//        double max;
//        double min;
//        cv::minMaxLoc(smoothKernels[k], &min, &max);
//        double range = max-min;
//        smoothKernels[k].convertTo(kchar, CV_8UC1, 255.0/range, -(255.0*min/range));
        cv::normalize(smoothKernels[k], kchar, 0, 255, cv::NORM_MINMAX, CV_8U);
        imwrite(path+"/gaus"+to_string(k)+".png", kchar);
    }
}

void GaborTexture::computeLambdaJZhang(unsigned int imageWidth){

    lambdaSize = log2(imageWidth/8);
    lambdaSize=(lambdaSize+1)*2;

    cout << "lambda_size = " << lambdaSize << endl;

    for(int i = 0; i < lambdaSize/2; i++){
        lambdas.push_back((pow(2.0,(double)i) - 0.5)/imageWidth);
    }

    for(int i = lambdaSize/2; i < lambdaSize; i++){
        lambdas.push_back(0.25 + lambdas[i-lambdaSize/2]);
    }

    for(int i = 0; i < lambdaSize/2; i++){
        lambdas[i] = 0.25-lambdas[i];
    }

    for(int i = 0; i < lambdaSize; i++)
    {
        for(int j = 0; j < lambdaSize; j++)
        {
            float swap=0;
            if(lambdas[j] > lambdas[i])
            {
                swap = lambdas[j];
                lambdas[j] = lambdas[i];
                lambdas[i] = swap;
            }
        }
    }

    for(int i = 0; i < lambdaSize; i++)
        lambdas[i] = 1/lambdas[i];
}

void GaborTexture::computeLambdaJain(unsigned int imageWidth)
{
//    Nc./((2.^(2:log2(Nc/4))).*sqrt(2));

    lambdaSize = log2(imageWidth/4)+1;
    int excluding = 2;
//    int excluding = 3;
//    int excluding = 4;
    lambdaSize -= excluding;
    double sqrt2 = sqrt(2.0);

    cout << "lambda_size = " << lambdaSize << endl;


    for(int i = 0; i < lambdaSize; i++){
        lambdas.push_back(imageWidth/(pow(2.0, i+excluding)*sqrt2));
    }

    for(int i = 0; i < lambdaSize; i++){
        for(int j = 0; j < lambdaSize; j++){
            float swap=0;
            if(lambdas[j] > lambdas[i]){
                swap = lambdas[j];
                lambdas[j] = lambdas[i];
                lambdas[i] = swap;
            }
        }
    }
}

Mat GaborTexture::createGaborKernel(float lambda, float theta, float gamma, float phi, float b)
{
    //sigma = (1 / pi) * sqrt(log(2)/2) * (2^b+1) / (2^b-1) * lambda;
    float sigma = (1/M_PI) * 0.588705011257737 *(pow(2.0,(double)b)+1) / (pow(2.0,(double)b)-1) * lambda;
//    float sigma = 25;
//    float sigma = 50;
    float Sy = sigma * gamma;

    int m_W=sigma;
//    int m_H=Sy;
//    int m_W = sigma < 25 ? sigma : 25;
//    int m_W = sigma < 50.0 ? sigma : 50;
//    int m_W = 25;
    int m_H = m_W* gamma;



//    Mat opencvGabor = cv::getGaborKernel(Size(50, 50), sigma, theta, lambda, gamma, phi, CV_64F);
//    return opencvGabor;

    Mat kernel = Mat((m_H*2+1), (m_H*2+1), CV_32FC1);

    //calculating kernel
    float xp, yp;
    int yy, xx;

    for(int x=(m_W*-1); x<=m_W; x++)
    {
        for(int y=(m_H*-1); y<=m_H; y++)
        {
            /*
            xp = x * cos(theta) + y * sin(theta);
            yp = y * cos(theta) - x * sin(theta);
            yy = fix(Sy)+y+1;
            xx = fix(sigma)+x+1;
            GF(yy,xx) = exp(-.5*(xp^2+gamma^2*yp^2)/sigma^2) * cos(2*pi*xp/lambda+phi);
            */

            xp = x * cos(theta) + y * sin(theta);
            yp = y * cos(theta) - x * sin(theta);
//            yy = ((int)(Sy))+y;
//            xx = ((int)(sigma))+x;
            yy = m_H+y;
            xx = m_W+x;

            //m_kernel[(y+m_H)*m_W+(x+m_W)] = exp(-0.5*(xp*xp+gamma*gamma*yp*yp)/(sigma*sigma)) * cos(2*PI*xp/lambda+phi);
            float v = exp(-0.5*(xp*xp+gamma*gamma*yp*yp)/(sigma*sigma)) * cos(2*M_PI*xp/lambda+phi);
            int index = yy*(m_W*2+1)+xx;

            ((float*)(kernel.data))[index] = v;
        }
    }
    m_W = m_W*2+1;
    m_H = m_H*2+1;

    return kernel;
}

Mat GaborTexture::createSmoothKernel(float beta, float lambda)
{
    float sigma = (1/M_PI) * 0.588705011257737 *(pow(2.0,beta)+1) / (pow(2.0,beta)-1) * lambda;

    int wsize = (int)(sigma)*2+1;
    float Sigma[4] = {sigma*sigma, 0, 0, sigma*sigma};

    float* mask = (float*) calloc(wsize*wsize, sizeof(float));

    float* dx = (float*) calloc(wsize*wsize, sizeof(float));
    float* dy = (float*) calloc(wsize*wsize, sizeof(float));

    int ws=wsize/2;
    for(int i=-ws; i<=ws; i++)
    {
        for(int j=-ws; j<=ws; j++)
        {
            dx[(i+ws)*wsize+(j+ws)]=j;
            dy[(i+ws)*wsize+(j+ws)]=i;
        }
    }

    float cofat=1/(Sigma[0]*Sigma[3] - Sigma[1]*Sigma[2]);
    float invSigma[4] = {Sigma[3]*cofat, -Sigma[1]*cofat, -Sigma[2]*cofat, Sigma[0]*cofat};
    //float invSigma[4] = {Sigma[3]*cofat, 0, 0, Sigma[0]*cofat};

    for(int i=0; i<wsize*wsize; i++)
    {
        float zy = (dy[i]*invSigma[0]+dx[i]*invSigma[2]) * dy[i];
        float zx = (dy[i]*invSigma[1]+dx[i]*invSigma[3]) * dx[i];

        mask[i]=exp(-0.5 * (zy+zx));
        //printf("\n%.4f", Z[i]);
    }
    free(dx);
    free(dy);

    Mat kernel = Mat(wsize, wsize, CV_32FC1);
    memcpy(kernel.data, mask, wsize*wsize*sizeof(float));

    return kernel;

}

void GaborTexture::computeNonlinearity(Mat &image) const
{
    assert(image.depth() == CV_32F);
//    static const float alpha = 0.01;
    static const float alpha = 0.10;
//    static const float alpha = 0.25;
//    static const float alpha = 0.50;
    for(int i = 0; i < image.rows; i++){
        float* data = image.ptr<float>(i);
        for(int j = 0; j < image.cols*image.channels(); j++){
            data[j] = tanh(data[j]*alpha);
        }
    }
}

Mat GaborTexture::convolution(const Mat& inImage, int kernelIndex, Point2i p) const
{
    assert(inImage.data);
    assert(inImage.depth() == CV_8U || inImage.depth() == CV_32F);
    assert(inImage.channels() == 1 || inImage.channels() == 3);
//    assert(p.x >= 0 && p.x < inImage.cols*inimage.channels());
    assert(p.x >= 0 && p.x < inImage.cols);
    assert(p.y >= 0 && p.y < inImage.rows);

    const Mat& kernel = kernels[kernelIndex];
    const Mat& smoothKernel = smoothKernels[kernelIndex];

    p = Point2i((p.x-kernel.cols/2), (p.y-kernel.rows/2));

    Mat inImageWindow(inImage, Rect2i(p, kernel.size()));
    Mat outPointMat = Mat(kernel.size(), CV_MAKETYPE(CV_32F, inImage.channels()));
    cv::filter2D(inImageWindow, outPointMat, CV_32F, kernel);

//    computeNonlinearity(outPointMat);
//    cv::filter2D(outPointMat, outPointMat, CV_32F, smoothKernel);

//    if(DEBUG){
//        double max, min;
//        cv::minMaxLoc(Mat(outPointMat, Rect(kernel.cols/2, kernel.rows/2, 1, 1)), &min, &max);
//        assert(max <= 1 && min >= -1);
//    }

    return Mat(outPointMat, Rect(kernel.cols/2, kernel.rows/2, 1, 1));

//    Mat outPointMat = Mat(1, 1, CV_MAKETYPE(CV_32F, inImage.channels()));

//    int dx = kernel.cols/2;
//    int dy = kernel.rows/2;

//    for(int c = 0; c < inImage.channels(); c++){

//        float sum = 0;

//        for(int m = 0; m < kernel.rows; m++){
//            for(int n = 0; n < kernel.cols; n++){

//                int x = p.x + (n - dx)*inImage.channels() + c;
//                int y = p.y + (m - dy);

//                if((x >= 0 && x < inImage.cols*inImage.channels()) &&
//                   (y >= 0 && y < inImage.rows))
//                {
//                    sum += inImage.at<uchar>(y, x) * kernel.at<float>(m, n);
//                }
//            }
//        }

//        outPointMat.at<float>(0, c) = sum;

//    }

//    return outPointMat;
}

void GaborTexture::convolution(const Mat& inImage, int kernelIndex, Mat& outImage) const
{
    assert(inImage.data);
    assert(inImage.depth() == CV_8U || inImage.depth() == CV_32F);
//    assert(inImage.channels() == outImage.channels());

    const Mat& kernel = kernels[kernelIndex];
    const Mat& smoothKernel = smoothKernels[kernelIndex];

    cv::filter2D(inImage, outImage, CV_32F, kernel, Point(-1,-1), 0, BORDER_REFLECT);
    computeNonlinearity(outImage);
    cv::filter2D(outImage, outImage, CV_32F, smoothKernel, Point(-1,-1), 0, BORDER_REFLECT);

//    if(DEBUG){
//        double max, min;
//        cv::minMaxLoc(outImage, &min, &max);
//        assert(max <= 1 && min >= -1);
//    }
}

int GaborTexture::getKernelMaxSize() const
{
    return kernelMaxSize;
}


















