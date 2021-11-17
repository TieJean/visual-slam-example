#include <vector>

#include "feature_tracker.h"
#include "slam.h"

using namespace std;
using namespace slam;

int main() {
    
    // ReprojectionError reprojection(cx, cy);
    // double camera1[] = {0.0, 0, 1.0, 0.0, 0.0, 0.0, 0.0};
    // double camera2[] = {0.707, 0, 0.707, 0.0, -1.0, 0.0, 1.0};
    // double camera3[] = {-0.707, 0, 0.707, 0.0, 1.0, 0.0, 1.0};
    // double point_test[]  = {0, 0, 1};
    // double residual[2];
    // // reprojection(camera1, point_test, residual);
    // // reprojection(camera2, point_test, residual);
    // // reprojection(camera3, point_test, residual);
    // Slam slam;
    // float X, Y, Z;
    // float x, y;
    // slam.imgToWorld_(camera3, cx, cy, 1.0*5000, &X, &Y, &Z);
    // slam.wolrdToImg_(camera3, 0, 0, 1.0, &x, &y);
    // printf("world coordinate: %.2f, %.2f, %.2f \n", X, Y, Z);
    // printf("img coordinate:   %.2f, %.2f\n", x, y);

    vector<pair<string, string>> timestamps;
    vector<pair<Vector3f, Quaternionf>> odometries;
    timestamps.emplace_back("1311877812.989574", "1311877812.987032");
    timestamps.emplace_back("1311877815.025449", "1311877815.025457");
    timestamps.emplace_back("1311877817.026998", "1311877817.027055");
    odometries.emplace_back(Vector3f(-2.2581, -2.3799, 0.5899), Quaternionf(-0.1617, 0.1914, 0.7173, -0.6502)); 
    odometries.emplace_back(Vector3f(-2.2640, -2.3775, 0.5895), Quaternionf(-0.1929, 0.2276, 0.7059, -0.6424)); 
    odometries.emplace_back(Vector3f(-2.2869, -2.3742, 0.5884), Quaternionf(-0.2905, 0.3374, 0.6641, -0.6005));  
    vector<Mat> imgs;

    Slam slam;
    slam.init();
    for (size_t i = 0; i < timestamps.size(); ++i) {
        slam.observeOdometry(odometries[i].first, odometries[i].second);
        Mat img = imread("../data/rgb/" + timestamps[i].first + ".png", 0);
        imgs.push_back(img);
        Mat depth = imread("../data/depth/" + timestamps[i].second + ".png", IMREAD_UNCHANGED);
        if (img.data == NULL || depth.data == NULL) {
            cout << "../data/rgb/" + timestamps[i].first + ".png" << endl;
            cout << "../data/depth/" + timestamps[i].second + ".png" << endl;
            perror("cannot open image files");
            exit(1);
        }
        slam.observeImage(img, depth);
    }

    // FeatureTracker tracker;
    // for (size_t i = 0; i < imgs.size(); ++i) {
    //     for (size_t j = i+1; j < imgs.size(); ++j) {
    //         vector<KeyPoint> kp1, kp2;
    //         Mat des1, des2;
    //         tracker.getKpAndDes(imgs[i], &kp1, &des1);
    //         tracker.getKpAndDes(imgs[j], &kp2, &des2);
    //         vector<DMatch> matches;
    //         tracker.match(des1, des2, &matches);
    //         Mat img_matches;
    //         drawMatches( imgs[i], kp1, imgs[j], kp2, matches, img_matches, Scalar::all(-1),
    //                     Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //         printf("i: %ld, j: %ld\n", i, j);
    //         imshow("Good Matches", img_matches );
    //         waitKey();
    //     }
    // }


    slam.displayCLMS();
    slam.displayPosesAndLandmarkcs();
    if (slam.optimize(false, true, false)) {
        slam.displayPosesAndLandmarkcs();
    } else {
        printf("optimization failed\n");
    }
    
}