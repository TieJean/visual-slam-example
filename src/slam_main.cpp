#include <vector>

#include "feature_tracker.h"
#include "slam.h"

using namespace std;
using namespace slam;
using namespace Eigen;

int main() {
    // Slam slam;
    // slam.init();
    // // pose: img to world
    // // camera: world to img
    // double pose1[] = {-0.1628, 0.1909, 0.7162, -0.6513, -2.2591, -2.3801, 0.5901};
    // double pose2[] = {-0.1929, 0.2276, 0.7059, -0.6424, -2.2581, -2.3799, 0.5899};
    // Mat depth1 = imread("../data/depth/1311877814.023140.png", IMREAD_UNCHANGED);
    // Mat depth2 = imread("../data/depth/1311877815.025457.png", IMREAD_UNCHANGED);
    // Affine3f odom_to_world, world_to_odom;
    
    // odom_to_world = Affine3f::Identity();
    // odom_to_world.translate(Vector3f(-2.2591, -2.3801, 0.5901));
    // odom_to_world.rotate(Quaternionf(-0.1628, 0.1909, 0.7162, -0.6513));
    // world_to_odom = odom_to_world.inverse();
    // Quaternionf angle1(world_to_odom.rotation());
    // Vector3f loc1(world_to_odom.translation());
    // double* camera1 = new double[] {angle1.w(), angle1.x(), angle1.y(), angle1.z(),
    //                                 loc1.x(), loc1.y(), loc1.z()};
    // odom_to_world = Affine3f::Identity();
    // odom_to_world.translate(Vector3f(-2.2581, -2.3799, 0.5899));
    // odom_to_world.rotate(Quaternionf(-0.1929, 0.2276, 0.7059, -0.6424));
    // world_to_odom = odom_to_world.inverse();
    // Quaternionf angle2(world_to_odom.rotation());
    // Vector3f loc2(world_to_odom.translation());
    // double* camera2 = new double[] {angle2.w(), angle2.x(), angle2.y(), angle2.z(),
    //                                 loc2.x(), loc2.y(), loc2.z()};
    // float X, Y, Z;
    // float x_prime, y_prime, z_prime;
    // float x, y;
    // x_prime = 541.87;
    // y_prime = 52.05;
    // z_prime = depth2.at<ushort>(52, 542);
    // cout << z_prime / 5000.0 << endl;
    // slam.imgToWorld_(camera2, x_prime, y_prime, 4*5000, &X, &Y, &Z);
    // printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
    // slam.worldToImg_(camera1, X, Y, Z, &x, &y);
    // printf("measure_pred: %.2f|%.2f\n", x, y);

    // x_prime = 438.84;
    // y_prime = 55.31;
    // z_prime = depth1.at<ushort>(55, 439);
    // slam.imgToWorld_(camera1, x_prime, y_prime, 4*5000, &X, &Y, &Z);
    // cout << z_prime / 5000.0 << endl;
    // printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
    // slam.worldToImg_(camera2, X, Y, Z, &x, &y);
    // printf("measure_pred: %.2f|%.2f\n", x, y);

    // cout << endl;
    // X = (-5.99 -5.04) / 2;
    // Y = (-4.82 -5.56) / 2;
    // Z = 1.67;
    // slam.worldToImg_(camera1, X, Y, Z, &x, &y);
    // printf("measure_pred: %.2f|%.2f\n", x, y);
    // slam.worldToImg_(camera2, X, Y, Z, &x, &y);
    // printf("measure_pred: %.2f|%.2f\n", x, y);



    vector<pair<string, string>> timestamps;
    vector<pair<Vector3f, Quaternionf>> odometries;
    timestamps.emplace_back("1311877812.989574", "1311877812.987032");
    timestamps.emplace_back("1311877814.025472", "1311877814.023140");
    timestamps.emplace_back("1311877815.025449", "1311877815.025457");
    // timestamps.emplace_back("1311877816.031231", "1311877816.031251");
    // timestamps.emplace_back("1311877817.026998", "1311877817.027055");
    odometries.emplace_back(Vector3f(-2.2581, -2.3799, 0.5899), Quaternionf(-0.1617, 0.1914, 0.7173, -0.6502)); 
    odometries.emplace_back(Vector3f(-2.2591, -2.3801, 0.5901), Quaternionf(-0.1628, 0.1909, 0.7162, -0.6513)); 
    odometries.emplace_back(Vector3f(-2.2640, -2.3775, 0.5895), Quaternionf(-0.1929, 0.2276, 0.7059, -0.6424)); 
    // odometries.emplace_back(Vector3f(-2.2732, -2.3759, 0.5888), Quaternionf(-0.2360, 0.2777, 0.6918, -0.6233)); 
    // odometries.emplace_back(Vector3f(-2.2869, -2.3742, 0.5884), Quaternionf(-0.2905, 0.3374, 0.6641, -0.6005));  
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
    // slam.displayPosesAndLandmarkcs();
    if (slam.optimize(false, true, false)) {
        // slam.displayPosesAndLandmarkcs();
    } else {
        printf("optimization failed\n");
    }
    
}