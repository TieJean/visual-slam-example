#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/types.hpp"

using namespace std;
using namespace cv;

using std::vector;
using cv::KeyPoint;

#ifndef FEATURE_TRACKER_H_
#define FEATURE_TRACKER_H_

namespace feature_tracker {

class FeatureTracker {

public:
    FeatureTracker();

    // Get keypoints and descriptors by timestamp
    // If having been calculated, just read from file 
    void getKpAndDes(const Mat& img, vector<KeyPoint>* kp_ptr, Mat* des_ptr);
    void getKpAndDes(const string& t, vector<KeyPoint>* kp_ptr, Mat* des_ptr);
    void match(const Mat& des1, const Mat& des2, vector<DMatch>* matches_ptr);
    void match(const string& t1, const string& t2, vector<DMatch>* matches_ptr);

// private:
    string kp_dir;
    string des_dir;
    string img_dir;
    int minHessian;
    float ratio_thresh;
};

}

#endif   // FEATURE_TRACKER_H_