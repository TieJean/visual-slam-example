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
        // 0.250000 0.247404 0.000000 0.000000 0.000000 0.375379 0.926872
        // 0.750000 0.681639 0.000000 0.000000 0.000000 0.310614 0.950536
        //                0  1    2   3   4         5         6
        //                qw -qy -qz  qx -y        -z         x
        double pose1[] = {0.926872, 0.000000, -0.375379, 0.000000,  -0.247404, 0.000000, 0.250000}; // 284.517127 427.718032
        double pose2[] = {0.950536, 0.000000, -0.310614, 0.000000,  -0.681639, 0.000000, 0.750000}; // 224.271992 443.602705
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

        printf("camera2: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", camera2[0], camera2[1], camera2[2], camera2[3], camera2[4], camera2[5], camera2[6]);
        printf("camera1: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", camera1[0], camera1[1], camera1[2], camera1[3], camera1[4], camera1[5], camera1[6]);

        float X, Y, Z;
        float x, y, z;
        float x_pred, y_pred;
        x = 224.271992; 
        y = 443.602705; // observation made by camera2
        z = 11.07 * 5000;
        slam.imgToWorld_(camera2, x, y, z, &X, &Y, &Z);
        printf("------------\n");
        printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
        slam.worldToImg_(camera2, X, Y, Z, &x_pred, &y_pred);
        printf("camera2 measure_pred (predicted by estimate landmark): %.2f|%.2f\n", x_pred, y_pred);  // 224.271992 443.602705
        slam.worldToImg_(camera1, X, Y, Z, &x_pred, &y_pred);
        printf("camera1 measure_pred (predicted by estimate landmark): %.2f|%.2f\n", x_pred, y_pred); // 284.517127 427.718032
        
        printf("------------\n");
        X = 18.883421; // world coordinate
        Y = -20.269062;
        Z = 2.886363;

        X = -9.147064; // camera coordinate
        Y = 5.498642;
        Z = 7.941555;
        slam.worldToImg_(camera2, X, Y, Z, &x_pred, &y_pred);
        printf("camera2 measure_pred (predicted by groundtruth landmark): %.2f|%.2f\n", x_pred, y_pred); 
        slam.worldToImg_(camera1, X, Y, Z, &x_pred, &y_pred);
        printf("camera1 measure_pred (predicted by groundtruth landmark): %.2f|%.2f\n", x_pred, y_pred); 


    }

    if (1) {
        const string DATA_DIR = "../data/vslam_superset1/low_density/groundtruth/";
        const string FEATURE_DIR = DATA_DIR + "features/";
        size_t N_POSE = stoi(argv[1]);
        size_t N_LANDMARK = 2 + stoi(argv[2]);

        vector<Vector3f> landmarks; // store landmark positions in world coordinate
        ifstream fp;
        size_t line_num;
        string line;
        Slam slam;
        vector<Measurement> measurements;

        Affine3f extrinsicCamera = Affine3f::Identity();
        extrinsicCamera.translate(Vector3f(0,0,0));
        extrinsicCamera.rotate(Quaternionf(0.5, 0.5, -0.5, 0.5));

        fp.open(FEATURE_DIR + "features.txt");
        if (!fp.is_open()) {
            printf("error in opening file %s\n", (FEATURE_DIR + "features.txt").c_str());
            exit(1);
        }
        while ( getline(fp, line) ) {
            float tmp[4];
            stringstream tokens(line);
            tokens >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
            landmarks.emplace_back(tmp[1], tmp[2], tmp[3]);
        }
        fp.close();

        
        slam.init(N_POSE, 1000);
        for (size_t t = 1; t <= N_POSE; ++t) {
            
            if (t < 10) {
                fp.open(DATA_DIR + "00000" + to_string(t) + ".txt");
                if (!fp.is_open()) {
                    printf("error in opening file %s\n", (DATA_DIR + "00000" + to_string(t) + ".txt").c_str());
                    printf("error in opening file\n");
                    exit(1);
                }
            } else {
                fp.open(DATA_DIR + "0000"  + to_string(t) + ".txt");
                if (!fp.is_open()) {
                    printf("error in opening file %s\n", (DATA_DIR + "0000"  + to_string(t) + ".txt").c_str());
                    printf("error in opening file\n");
                    exit(1);
                }
            }
            
            line_num = 0;
            measurements.clear();

            double pose[7]; 
            Vector3f loc;
            Quaternionf angle;
            while ( getline(fp, line) ) {
                ++line_num;
                if (line_num == 1) {continue;}
                
                if (line_num == 2) { // pose
                    stringstream tokens(line);
                    // pose: qw  qx  qy qz  x  y z (camera coordinate)
                    // pose: qw -qy -qz qx -y -z x (world coordinate) 
                    for (size_t i = 0; i < poseDim; ++i) { tokens >> pose[i]; }
                    loc   = Vector3f(extrinsicCamera * Vector3f(pose[0], pose[1], pose[2]));
                    angle = Quaternionf((extrinsicCamera * Quaternionf(pose[6], pose[3], pose[4], pose[5])).rotation());
                    slam.observeOdometry(loc, angle);
                    // printf("cameras: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", loc.x(), loc.y(), loc.z(), angle.x(), angle.y(), angle.z(), angle.w());
                    // printf("pose:    %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]);
                    continue;
                }
                size_t feature_idx;
                float measurement_x, measurement_y;
                stringstream tokens(line);
                tokens >> feature_idx >> measurement_x >> measurement_y;
                if (feature_idx > N_LANDMARK) {break;}
                
                // camera z --> odom x
                // camera x --> odom -y
                // camera y --> odom -z
                // |(|z2 - z1| - |(x2 - x1)*tan(theta)|) * cos(theta)| - not sure about signs and angles; need to check
                // landmarks are in world coordinate
                Affine3f landmark_to_world = Affine3f::Identity();
                landmark_to_world.translate(landmarks[feature_idx-1]);
                landmark_to_world = extrinsicCamera * landmark_to_world;
                Affine3f camera_to_world = Affine3f::Identity();
                camera_to_world.translate(Vector3f(pose[0], pose[1], pose[2]));
                camera_to_world.rotate(Quaternionf(pose[6], pose[3], pose[4], pose[5]));
                camera_to_world = extrinsicCamera * camera_to_world;
                Affine3f landmark_to_camera = landmark_to_world * camera_to_world.inverse();
                printf("landmark (world coordinate)  %.2f, %.2f, %.2f\n", landmarks[feature_idx-1].x(), landmarks[feature_idx-1].y(), landmarks[feature_idx-1].z());
                printf("landmark (camera coordinate) %.2f, %.2f, %.2f\n", landmark_to_world.translation().x(), landmark_to_world.translation().y(), landmark_to_world.translation().z());
                printf("camera (world coordinate)    %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]);
                cout << "camera (camera coordinate)" << endl;
                cout << camera_to_world.translation() << endl;
                cout << camera_to_world.rotation() << endl;
                cout << "final result (camera coordinate)" << endl;
                cout << landmark_to_camera.translation() << endl;
                return 0;

                float depth = 1.0; // TODO: FIXME
                measurements.emplace_back(feature_idx, measurement_x, measurement_y, depth * 5000);
                printf("%ld, %ld, %.2f, %.2f, %.2f\n", t, feature_idx, measurement_x, measurement_y, depth);
                printf("landmark: %.2f, %.2f, %.2f\n", -landmarks[feature_idx-1].y(), -landmarks[feature_idx-1].z(), landmarks[feature_idx-1].x());
                printf("pose:     %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]);
                cout << endl;
            }
            slam.observeImage(measurements);
            fp.close();
        }
        slam.dumpLandmarksToCSV("../data/results/vslam-superset-landmarks-initEstimate.csv");
        slam.optimize();
        // slam.displayLandmarks();
        // slam.displayPoses();
        slam.dumpLandmarksToCSV("../data/results/vslam-superset-landmarks.csv");

    }
    
}