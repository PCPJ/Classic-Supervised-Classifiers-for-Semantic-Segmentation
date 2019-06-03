# Classic-Supervised-Classifiers-for-Semantic-Segmentation


# Classic-Supervised-Classifiers-for-Semantic-Segmentation

# Publications

This repository contains the software used for the experiments published in this paper.
This software was build aiming the utilization and comparison of classical machine learning supervised classifiers in the semantic segmentation of agricultural aerial images. The code was written in c++ utilizing the OpenCV library.

# Usage

There are three executable tools in this repository: imgSeg, precisionCal, and decisionBoundaries.

## imgSeg

With this tool you can segment a list of images, specifying the wanted classifier, feature vector, and train data. There are some examples of yml files that can be used to configure the tool. The next images show an example of color texture image being segmented with several classifiers with two different image features.

![inputs](https://user-images.githubusercontent.com/8596365/58821575-c89ceb00-860b-11e9-869d-469fd4aac0b6.png)
![results](https://user-images.githubusercontent.com/8596365/58821615-d7839d80-860b-11e9-8804-c6ac874da926.png)

## precisionCal

This tool computes, for a list of results and their respective ground truths, two precision measures (F1 score and Jaccard index), and saves this data in a yml file.

## decisionBoundaries

This is a useful tool to analyze the behavior of each classifier. The user can draw a bi-dimensional train data, and the software computes de decision boundaries for several classifiers. The next image shows some examples.

![bordas](https://user-images.githubusercontent.com/8596365/58821715-0bf75980-860c-11e9-8e7b-4e89a3b623e7.png)
