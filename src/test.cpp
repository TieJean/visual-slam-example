#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main() {
    Mat img1 = imread("../../data/gray/1311877812.989574.png", 0);
    Mat img2 = imread("../../data/gray/1311877812.989574.png", 0);
    if (img1.data == NULL || img2.data == NULL) {
        perror("cannot open image files");
        exit(1);
    }

    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // std::vector<int> vints;
    // vints.push_back(1);
    // vints.push_back(2);
    // vints.push_back(3);
    // ofstream fp_out;
    // fp_out.open("tmp.txt");
    // fp_out.write((char*)&vints, sizeof(vints));
    // fp_out.close();

    // std::vector<int> vints2;
    // ifstream fp_in;
    // fp_in.open("tmp.txt");
    // fp_in.read((char*)&vints2, sizeof(vints));
    // fp_in.close();
    // for (int i : vints2) {
    //     cout << i << endl;
    // }
    // cout << "here" << endl;


    // std::sort(good_matches.begin(), good_matches.end());
    // std::vector<DMatch> best_n_matches;
    // std::copy_n(good_matches.begin(), 10, best_n_matches);

    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imshow("Good Matches", img_matches );
    waitKey();
    return 0;
}