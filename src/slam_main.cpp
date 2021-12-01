#include <vector>

#include "feature_tracker.h"
#include "slam.h"

using namespace std;
using namespace slam;
using namespace Eigen;


int main() {
    if (1) {
        Slam slam;
        slam.init();
        // odom x --> camera z
        // odom y --> camera x
        // odom z --> camera y
        // double pose1[] = {1, 0, 0, 0, 0, 0, 0.13};
        // double pose2[] = {1, 0, 0, 0, 0.1524, 0, 0.13};
        double pose1[] = {1, 0, 0, 0, 0.000000, 0.000000, -20.000000};
        double pose2[] = {1, 0, 0, 0, 0.000000, 0.000000, -18.500000};

        Affine3f odom_to_world, world_to_odom;
        odom_to_world = Affine3f::Identity();
        // odom_to_world.translate(Vector3f(0, 0, 0.13));
        odom_to_world.translate(Vector3f(0.000000, 0.000000, -20.000000));
        odom_to_world.rotate(Quaternionf(1, 0, 0, 0));
        world_to_odom = odom_to_world.inverse();
        Quaternionf angle1(world_to_odom.rotation());
        Vector3f loc1(world_to_odom.translation());
        double* camera1 = new double[] {angle1.w(), angle1.x(), angle1.y(), angle1.z(),
                                        loc1.x(), loc1.y(), loc1.z()};
        odom_to_world = Affine3f::Identity();
        // odom_to_world.translate(Vector3f(0.1524, 0, 0.13));
        odom_to_world.translate(Vector3f(0.000000, 0.000000, -18.500000));
        odom_to_world.rotate(Quaternionf(1, 0, 0, 0));
        world_to_odom = odom_to_world.inverse();
        Quaternionf angle2(world_to_odom.rotation());
        Vector3f loc2(world_to_odom.translation());
        double* camera2 = new double[] {angle2.w(), angle2.x(), angle2.y(), angle2.z(),
                                        loc2.x(), loc2.y(), loc2.z()};
        
        float X, Y, Z;
        float x_prime, y_prime, z_prime;
        float x, y;
        x_prime = 536.877545; 
        y_prime = 209.116121;
        z_prime = 1 * 5000;
        slam.imgToWorld_(pose2, x_prime, y_prime, z_prime, &X, &Y, &Z);
        printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
        slam.worldToImg_(camera1, X, Y, Z, &x, &y);
        printf("measure_pred: %.2f|%.2f\n", x, y);

        x_prime = 528.511092; 
        y_prime = 210.307524;
        z_prime = (1 + 1.5) * 5000;
        slam.imgToWorld_(pose1, x_prime, y_prime, z_prime, &X, &Y, &Z);
        printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
        slam.worldToImg_(camera2, X, Y, Z, &x, &y);
        printf("measure_pred: %.2f|%.2f\n", x, y);
    }

    if (0) {
        Slam slam;
        slam.init();
        // pose: img to world
        // camera: world to img
        // odom x --> camera z
        // odom y --> camera x
        // odom z --> camera y
        double pose1[] = {-0.1617, -0.6502, 0.1914, 0.7173, 0.5899, -2.2581, -2.3799};
        double pose2[] = {-0.1628, -0.6513, 0.1909, 0.7162, 0.5901, -2.2591, -2.3801};
        double pose3[] = {-0.1929, -0.6424, 0.2276, 0.7059, 0.5899, -2.2581, -2.3799};
        Mat depth1 = imread("../data/depth/1311877812.987032.png", IMREAD_UNCHANGED);
        Mat depth2 = imread("../data/depth/1311877815.025457.png", IMREAD_UNCHANGED);
        Mat depth3 = imread("../data/depth/1311877815.025457.png", IMREAD_UNCHANGED);
        Affine3f odom_to_world, world_to_odom;
        
        odom_to_world = Affine3f::Identity();
        odom_to_world.translate(Vector3f(0.5899, -2.2581, -2.3799));
        odom_to_world.rotate(Quaternionf(-0.1617, -0.6502, 0.1914, 0.7173));
        world_to_odom = odom_to_world.inverse();
        Quaternionf angle1(world_to_odom.rotation());
        Vector3f loc1(world_to_odom.translation());
        double* camera1 = new double[] {angle1.w(), angle1.x(), angle1.y(), angle1.z(),
                                        loc1.x(), loc1.y(), loc1.z()};
        printf("camera1:         %.4f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f\n", camera1[0], camera1[1], camera1[2], camera1[3], camera1[4], camera1[5], camera1[6]);

        odom_to_world = Affine3f::Identity();
        odom_to_world.translate(Vector3f(0.5901, -2.2591, -2.3801));
        odom_to_world.rotate(Quaternionf(-0.1628, -0.6513, 0.1909, 0.7162));
        world_to_odom = odom_to_world.inverse();
        Quaternionf angle2(world_to_odom.rotation());
        Vector3f loc2(world_to_odom.translation());
        double* camera2 = new double[] {angle2.w(), angle2.x(), angle2.y(), angle2.z(),
                                        loc2.x(), loc2.y(), loc2.z()};
        printf("camera2:         %.4f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f\n", camera2[0], camera2[1], camera2[2], camera2[3], camera2[4], camera2[5], camera2[6]);

        odom_to_world = Affine3f::Identity();
        odom_to_world.translate(Vector3f(0.5899, -2.2581, -2.3799));
        odom_to_world.rotate(Quaternionf(-0.1929, -0.6424, 0.2276, 0.7059));
        world_to_odom = odom_to_world.inverse();
        Quaternionf angle3(world_to_odom.rotation());
        Vector3f loc3(world_to_odom.translation());
        double* camera3 = new double[] {angle3.w(), angle3.x(), angle3.y(), angle3.z(),
                                        loc3.x(), loc3.y(), loc3.z()};
        printf("camera3:         %.4f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f\n", camera3[0], camera3[1], camera3[2], camera3[3], camera3[4], camera3[5], camera3[6]);

        float X, Y, Z;
        float x_prime, y_prime, z_prime;
        float x, y;
        // x_prime = 541.87;
        // y_prime = 52.05;
        x_prime = 324.74;
        y_prime = 192.00;
        z_prime = depth2.at<ushort>((int)y_prime, (int)x_prime);
        cout << z_prime / 5000.0 << endl;
        slam.imgToWorld_(pose2, x_prime, y_prime, z_prime, &X, &Y, &Z);
        printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
        slam.worldToImg_(camera1, X, Y, Z, &x, &y);
        printf("measure_pred: %.2f|%.2f\n", x, y);
        slam.imgToWorld_(camera2, x_prime, y_prime, depth2, &X, &Y, &Z);
        printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
        slam.worldToImg_(camera1, X, Y, Z, &x, &y);
        printf("measure_pred: %.2f|%.2f\n", x, y);

        x_prime = 346.08;
        y_prime = 125.79;
        z_prime = depth1.at<ushort>((int)y_prime, (int)x_prime);
        slam.imgToWorld_(pose1, x_prime, y_prime, z_prime, &X, &Y, &Z);
        cout << z_prime / 5000.0 << endl;
        printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
        slam.worldToImg_(camera2, X, Y, Z, &x, &y);
        printf("measure_pred: %.2f|%.2f\n", x, y);

        cout << endl;
        X = (-1.35 -0.41) / 2;
        Y = (-5.76 -3.89) / 2;
        Z = -2.23;
        cout << X << ", " << Y << ", " << Z << endl;
        slam.worldToImg_(camera1, X, Y, Z, &x, &y);
        printf("measure_pred: %.2f|%.2f\n", x, y);
        slam.worldToImg_(camera2, X, Y, Z, &x, &y);
        printf("measure_pred: %.2f|%.2f\n", x, y);
    }
    

    if (0) {
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


        // slam.displayCLMS();
        // slam.displayPosesAndLandmarkcs();
        if (slam.optimize(false, true, false)) {
            slam.displayPoses();
        } else {
            printf("optimization failed\n");
        }
    }

    
    
}