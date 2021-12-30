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

        double pose1[] = {0.00,0.00,0.00,0.00,0.00,0.38,0.92}; // 284.517127 427.718032
        double pose2[] = {0.25,0.25,0.00,0.00,0.00,0.38,0.93}; // 224.271992 443.602705
        Affine3f extrinsicCamera = Affine3f::Identity();
        extrinsicCamera.translate(Vector3f(0,0,0));
        extrinsicCamera.rotate(Quaternionf(0.5, 0.5, -0.5, 0.5));

        AngleAxisf angle_a1(Quaternionf(pose1[6], pose1[3], pose1[4], pose1[5]).normalized());
        // cout << "before" << endl;
        // cout << angle_a1.angle() << endl << angle_a1.axis() << endl;
        // cout << angle_a1.toRotationMatrix() << endl;
        angle_a1 = AngleAxisf(angle_a1.angle(), extrinsicCamera * angle_a1.axis());
        // cout << "after" << endl;
        // cout << angle_a1.angle() << endl << angle_a1.axis() << endl;
        // cout << angle_a1.toRotationMatrix() << endl;

        Affine3f camera_to_world, world_to_camera;
        camera_to_world = Affine3f::Identity();
        camera_to_world.translate(extrinsicCamera * Vector3f(pose1[0], pose1[1], pose1[2]));
        camera_to_world.rotate(angle_a1);
        world_to_camera = camera_to_world.inverse();
        Quaternionf angle1(world_to_camera.rotation());
        Vector3f loc1(world_to_camera.translation());
        double* camera1 = new double[] {angle1.w(), angle1.x(), angle1.y(), angle1.z(),
                                        loc1.x(), loc1.y(), loc1.z()};
        AngleAxisf angle_a2(Quaternionf(pose2[6], pose2[3], pose2[4], pose2[5]).normalized());
        angle_a2 = AngleAxisf(angle_a2.angle(), extrinsicCamera * angle_a2.axis());
        camera_to_world = Affine3f::Identity();
        camera_to_world.translate(Vector3f(pose2[0], pose2[1], pose2[2]));
        camera_to_world.rotate(angle_a2);
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
        x = 284.517127; 
        y = 427.718032; // observation made by camera2
        z = 11.07 * 5000;
        slam.imgToWorld_(camera2, x, y, z, &X, &Y, &Z);
        printf("------------\n");
        printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
        slam.worldToImg_(camera2, X, Y, Z, &x_pred, &y_pred);
        printf("camera2 measure_pred (predicted by estimate landmark): %.2f|%.2f\n", x_pred, y_pred);  // 284.517127 427.718032
        slam.worldToImg_(camera1, X, Y, Z, &x_pred, &y_pred);
        printf("camera1 measure_pred (predicted by estimate landmark): %.2f|%.2f\n", x_pred, y_pred);  // 291.782165 422.021836
        
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

    if (0) {
        Affine3f A_to_B = Affine3f::Identity();
        A_to_B.translate(Vector3f(0, 0, 1));
        A_to_B.rotate(Quaternionf(0, 0, -1, 0));
        cout << "A_to_B (matrix)" << endl;
        cout << A_to_B.matrix() << endl;
        Affine3f B_to_A = A_to_B.inverse();
        cout << "B_to_A (matrix)" << endl;
        cout << B_to_A.matrix() << endl;
        cout << B_to_A.translation() << endl;
        cout << Quaternionf(B_to_A.rotation()) << endl;

        Affine3f test1 = Affine3f::Identity();
        test1.translate(B_to_A.translation());
        test1.rotate(B_to_A.rotation());
        test1 = test1.inverse();
        cout << "test1" << endl;
        cout << test1.translation() << endl;
        cout << Quaternionf(test1.rotation()) << endl;
        cout << endl;
    }

    if (1) {
        string DATA_DIR = "../data/unittest1/";
        const string FEATURE_DIR = DATA_DIR + "features/";
        size_t N_POSE = stoi(argv[1]);
        size_t N_LANDMARK = 2 + stoi(argv[2]);
        if (argc >= 4) { DATA_DIR = DATA_DIR + string(argv[3]) + "/"; }

        vector<Vector3f> landmarks; // store landmark positions in world coordinate
        ifstream fp;
        size_t line_num;
        string line;
        Slam slam;

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

        
        slam.init(N_POSE, 5);
        for (size_t t = 1; t <= N_POSE; ++t) {
            
            if (t < 10) {
                fp.open(DATA_DIR + "00000" + to_string(t) + ".txt");
                if (!fp.is_open()) {
                    printf("error in opening file %s\n", (DATA_DIR + "00000" + to_string(t) + ".txt").c_str());
                    exit(1);
                }
            } else {
                fp.open(DATA_DIR + "0000"  + to_string(t) + ".txt");
                if (!fp.is_open()) {
                    printf("error in opening file %s\n", (DATA_DIR + "0000"  + to_string(t) + ".txt").c_str());
                    exit(1);
                }
            }
            
            line_num = 0;
            vector<Measurement> measurements; 

            double pose[7]; 
            Vector3f loc;
            Quaternionf angle;
            AngleAxisf angle_a;
            while ( getline(fp, line) ) {
                ++line_num;
                if (line_num == 1) {continue;}
                
                if (line_num == 2) { // pose
                    stringstream tokens(line);
                    // pose: qw  qx  qy qz  x  y z (camera coordinate)
                    // pose: qw -qy -qz qx -y -z x (world coordinate) 
                    for (size_t i = 0; i < poseDim; ++i) { tokens >> pose[i]; }
                    loc   = Vector3f(extrinsicCamera * Vector3f(pose[0], pose[1], pose[2]));
                    angle_a = AngleAxisf(Quaternionf(pose[6], pose[3], pose[4], pose[5]));
                    // cout << "before: " << endl;
                    // cout << angle_a.angle() << endl;
                    // cout << angle_a.axis() << endl;
                    // cout << angle_a.toRotationMatrix() << endl;
                    // angle_a = (extrinsicCamera * angle_a).rotation();
                    angle_a = AngleAxisf(angle_a.angle(), extrinsicCamera * angle_a.axis());
                    // cout << "after: " << endl;
                    // cout << angle_a.angle() << endl;
                    // cout << angle_a.axis() << endl;
                    // cout << angle_a.toRotationMatrix() << endl;
                    angle = Quaternionf(angle_a);

                    // cout << endl;
                    // cout << "pose: " << t << endl;
                    // cout << "angle: " << endl;
                    // cout << angle_a.angle() << endl;
                    // cout << "axis: " << endl;
                    // cout << angle_a.axis() << endl;
                    // cout << "rotation matrix" << endl;
                    // cout << angle_a.toRotationMatrix() << endl;
                    // printf("cameras: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", loc.x(), loc.y(), loc.z(), angle.x(), angle.y(), angle.z(), angle.w());
                    // printf("pose:    %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]);

                    slam.observeOdometry(loc, angle);
                    // slam.observeOdometry(Vector3f(pose[0], pose[1], pose[2]), Quaternionf(pose[6], pose[3], pose[4], pose[5]));
                    // cout << loc << endl;
                    // cout << angle.toRotationMatrix() << endl;
                    // cout << endl;
                    continue;
                }
                size_t feature_idx;
                float measurement_x, measurement_y;
                stringstream tokens(line);
                tokens >> feature_idx >> measurement_x >> measurement_y;
                if (feature_idx <= N_LANDMARK) {
                    // camera z --> odom x
                    // camera x --> odom -y
                    // camera y --> odom -z
                    Vector3f landmark_in_world(landmarks[feature_idx-1]);
                    Affine3f camera_to_world = Affine3f::Identity();
                    camera_to_world.translate(Vector3f(pose[0], pose[1], pose[2]));
                    camera_to_world.rotate(Quaternionf(pose[6], pose[3], pose[4], pose[5]));
                    Vector3f landmark_in_camera =  camera_to_world.inverse() * landmark_in_world;
                    float depth = landmark_in_camera.x(); // TODO: FIXME
                    measurements.emplace_back(feature_idx, measurement_x, measurement_y, depth * 5000);
                    // printf("add measurement: %ld, %ld, %.2f, %.2f, %.2f\n", t, feature_idx, measurement_x, measurement_y, depth);
                    // printf("landmark: %.2f, %.2f, %.2f\n", -landmarks[feature_idx-1].y(), -landmarks[feature_idx-1].z(), landmarks[feature_idx-1].x());
                    // printf("pose:     %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]);
                    // cout << endl;
                }
            }
            slam.observeImage(measurements);
            fp.close();
        }

        // slam.initLandmarks(landmarks, N_LANDMARK); // manually init landmark
        
        slam.evaluate();
        slam.dumpLandmarksToCSV("../data/results/unittest1-landmarks-initEstimate.csv");
        slam.dumpPosesToCSV("../data/results/unittest1-poses-initEstimate.csv");
        slam.displayLandmarks();
        slam.displayPoses();
        slam.optimize();
        slam.displayLandmarks();
        slam.displayPoses();
        slam.dumpLandmarksToCSV("../data/results/unittest1-landmarks.csv");
        slam.dumpPosesToCSV("../data/results/unittest1-poses.csv");
        slam.evaluate();
    }
    
}