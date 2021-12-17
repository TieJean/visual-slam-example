#include <vector>
#include <fstream>
#include <string>

#include "feature_tracker.h"
#include "slam.h"

using namespace std;
using namespace slam;
using namespace Eigen;

/**
 * TODO: change these back after finishing debugging
 * 1) free memory
 * 2) hashing 
 * 3) storage efficiency
 */

int main(int argc, char** argv) {
    if (0) {
        // pose: camera_to_world
        // camera: world_to_camera
        Slam slam;
        slam.init(0, 0);
        // camera z --> odom x
        // camera x --> odom -y
        // camera y --> odom -z
        // double pose1[] = {-20.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000};
        // double pose2[] = {-18.500000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000};

        //                0  1    2   3   4         5         6
        //                qw -qy -qz  qx -y        -z         x
        double pose1[] = {1, 0,   0,  0,  0.000000, 0.000000, -20.000000};
        double pose2[] = {1, 0,   0,  0,  0.000000, 0.000000, -18.500000};
        // double pose1[] = {1, 0, 0, 0, -20.000000, 0.000000, 0};
        // double pose2[] = {1, 0, 0, 0, -18.500000, 0.000000, 0};

        Affine3f camera_to_world, world_to_camera;
        camera_to_world = Affine3f::Identity();
        camera_to_world.translate(Vector3f(pose1[4], pose1[5], pose1[6]));
        camera_to_world.rotate(Quaternionf(pose1[0], pose1[1], pose1[2], pose1[3]));
        world_to_camera = camera_to_world.inverse();
        Quaternionf angle1(world_to_camera.rotation());
        Vector3f loc1(world_to_camera.translation());
        double* camera1 = new double[] {angle1.w(), angle1.x(), angle1.y(), angle1.z(),
                                        loc1.x(), loc1.y(), loc1.z()};
        camera_to_world = Affine3f::Identity();
        camera_to_world.translate(Vector3f(pose2[4], pose2[5], pose2[6]));
        camera_to_world.rotate(Quaternionf(pose2[0], pose2[1], pose2[2], pose2[3]));
        world_to_camera = camera_to_world.inverse();
        Quaternionf angle2(world_to_camera.rotation());
        Vector3f loc2(world_to_camera.translation());
        double* camera2 = new double[] {angle2.w(), angle2.x(), angle2.y(), angle2.z(),
                                        loc2.x(), loc2.y(), loc2.z()};
        
        float X, Y, Z;
        float x, y, z;
        float x_pred, y_pred;
        x = 534.015111; 
        y = 209.523739;
        z = (18.883421 + 18.5) * 5000;
        slam.imgToWorld_(camera2, x, y, z, &X, &Y, &Z);
        printf("------------\n");
        printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
        slam.worldToImg_(camera2, X, Y, Z, &x_pred, &y_pred);
        printf("camera2 measure_pred (estimate landmark): %.2f|%.2f\n", x_pred, y_pred); // 528.511092 210.307524
        slam.worldToImg_(camera1, X, Y, Z, &x_pred, &y_pred);
        printf("camera1 measure_pred (estimate landmark): %.2f|%.2f\n", x_pred, y_pred); // 528.511092 210.307524
        
        printf("------------\n");
        X = 18.883421; // world coordinate
        Y = -20.269062;
        Z = 2.886363;

        X = 20.269062; // camera coordinate
        Y = -2.886363;
        Z = 18.883421;
        slam.worldToImg_(camera2, X, Y, Z, &x_pred, &y_pred);
        printf("camera2 measure_pred (groundtruth landmark): %.2f|%.2f\n", x_pred, y_pred); // 528.511092 210.307524
        slam.worldToImg_(camera1, X, Y, Z, &x_pred, &y_pred);
        printf("camera1 measure_pred (groundtruth landmark): %.2f|%.2f\n", x_pred, y_pred); // 528.511092 210.307524


    }

    if (1) {
        // vector<pair<>>
        // const string DATA_DIR = "../data/vslam_set2/";
        // const string FEATURE_DIR = DATA_DIR + "features/";
        // size_t N_POSE = stoi(argv[1]);
        // size_t N_LANDMARK = 2 + stoi(argv[2]);

        // vector<Vector3f> landmarks; // store landmark positions in world coordinate
        // ifstream fp;
        // size_t line_num;
        // string line;
        // Slam slam;
        // vector<Measurement> measurements;

        // fp.open(FEATURE_DIR + "features.txt");
        // if (!fp.is_open()) {
        //     printf("error in opening file\n");
        //     exit(1);
        // }
        // while ( getline(fp, line) ) {
        //     float tmp[4];
        //     stringstream tokens(line);
        //     tokens >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
        //     landmarks.emplace_back(tmp[1], tmp[2], tmp[3]);
        // }
        // fp.close();

        
        // slam.init(N_POSE, 99);
        // for (size_t t = 1; t <= N_POSE; ++t) {
            
        //     if (t < 10) {
        //         fp.open(DATA_DIR + "00000" + to_string(t) + ".txt");
        //     } else {
        //         fp.open(DATA_DIR + "0000"  + to_string(t) + ".txt");
        //     }
        //     if (!fp.is_open()) {
        //         printf("error in opening file\n");
        //         exit(1);
        //     }
        //     line_num = 0;
        //     measurements.clear();

        //     double pose[7]; 
        //     while ( getline(fp, line) ) {
        //         ++line_num;
        //         if (line_num == 1) {continue;}
                
        //         if (line_num == 2) { // pose
        //             stringstream tokens(line);
        //             tokens >> pose[6] >> pose[4] >> pose[5] >> pose[3] >> pose[1] >> pose[2] >> pose[0];
        //             pose[1] = -pose[1];
        //             pose[2] = -pose[2];
        //             pose[4] = -pose[4];
        //             pose[5] = -pose[5];
        //             slam.observeOdometry(Vector3f(pose[4], pose[5], pose[6]), Quaternionf(pose[0], pose[1], pose[2], pose[3]));
        //             // printf("tokens: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]);
        //             continue;
        //         }
        //         if (line_num > N_LANDMARK) {break;}
        //         size_t feature_idx;
        //         float measurement_x, measurement_y;
        //         stringstream tokens(line);
        //         tokens >> feature_idx >> measurement_x >> measurement_y;
        //         measurements.emplace_back(feature_idx, measurement_x, measurement_y, (landmarks[feature_idx-1].x() - pose[6]) * 5000);
        //         // printf("%ld, %.2f, %.2f, %.2f\n", feature_idx, measurement_x, measurement_y, landmarks[feature_idx-1].x() - pose[6]);
        //     }
        //     slam.observeImage(measurements);
        //     fp.close();
        // }
        // slam.dumpLandmarksToCSV("../data/results/vslam-set2-landmarks-initEstimate.csv");
        // slam.optimize();
        // // slam.displayLandmarks();
        // // slam.displayPoses();
        // slam.dumpLandmarksToCSV("../data/results/vslam-set2-landmarks.csv");

    }
    
}