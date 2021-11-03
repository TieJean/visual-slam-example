#include <string>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "feature_tracker.h"

using namespace std;
using namespace cv;

using std::vector;
using std::cout;
using std::endl;
using cv::DMatch;
using namespace cv::xfeatures2d;

namespace feature_tracker {

FeatureTracker::FeatureTracker() :
    kp_dir("../data/keypoints"),
    des_dir("../data/descriptors"),
    img_dir("../data/gray/"),
    minHessian(400),
    ratio_thresh(0.2f) {
    }

void FeatureTracker::getKpAndDes(const string& t, vector<KeyPoint>* kp_ptr, Mat* des_ptr) {
    vector<KeyPoint>& kp = *kp_ptr;
    Mat& des = *des_ptr;

    Mat img = imread(img_dir + t + ".png", 0);
    Ptr<SURF> detector = SURF::create( minHessian );
    detector->detectAndCompute( img, noArray(), kp, des);
}

void FeatureTracker::match(const vector<KeyPoint>& kp1, const Mat& des1, 
                           const vector<KeyPoint>& kp2, const Mat& des2, 
                           vector<DMatch>* matches_ptr) {
    // return
    vector<DMatch>& matches = *matches_ptr;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector< vector<DMatch> > knn_matches;
    matcher->knnMatch( des1, des2, knn_matches, 2);
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            matches.push_back(knn_matches[i][0]);
        }
    }
}

void FeatureTracker::match(const string& t1, const string& t2, vector<DMatch>* matches_ptr) {
    // return
    vector<DMatch>& matches = *matches_ptr;

    vector<KeyPoint> kp1;
    Mat des1;
    vector<KeyPoint> kp2;
    Mat des2;
    Mat img1 = imread(img_dir + t1 + ".png", 0);
    Mat img2 = imread(img_dir + t2 + ".png", 0);
    if (img1.data == NULL || img2.data == NULL) {
        perror("cannot open image files");
        exit(1);
    }

    Ptr<SURF> detector = SURF::create( minHessian );
    detector->detectAndCompute(img1, noArray(), kp1, des1);
    detector->detectAndCompute(img2, noArray(), kp2, des2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector< vector<DMatch> > knn_matches;
    matcher->knnMatch( des1, des2, knn_matches, 2);
    
    // vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            matches.push_back(knn_matches[i][0]);
        }
    }
}

}


// int main() {
//     feature_tracker::FeatureTracker tracker;
//     const string t1 = "1311877813.025527";
//     const string t2 = "1311877817.026998";

//     Mat img1 = imread(tracker.img_dir + t1 + ".png", 0);
//     Mat img2 = imread(tracker.img_dir + t2 + ".png", 0);
//     if (img1.data == NULL || img2.data == NULL) {
//         cout << tracker.img_dir + t1 + ".png" << endl;
//         cout << tracker.img_dir + t2 + ".png" << endl;
//         perror("cannot open image files");
//         exit(1);
//     }
//     vector<KeyPoint> kp1;
//     Mat des1;
//     vector<KeyPoint> kp2;
//     Mat des2;
//     tracker.getKpAndDes(t1, &kp1, &des1);
//     tracker.getKpAndDes(t2, &kp2, &des2);

//     vector<DMatch> matches;
//     // tracker.match(t1, t2, &matches);
//     tracker.match(kp1, des1, kp1, des2, &matches);
//     cout << matches.size() << endl;

//     Mat img_matches;
//     drawMatches( img1, kp1, img2, kp2, matches, img_matches, Scalar::all(-1),
//                  Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//     imshow("Good Matches", img_matches );
//     waitKey();
//     return 0;
// }