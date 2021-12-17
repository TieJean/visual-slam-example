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

class Odometry {
public:
    Vector3f loc;
    Quaternionf angle;

    Odometry() {}
    Odometry(Vector3f loc, Quaternionf angle) {
        this->loc = loc;
        this->angle = angle;
    }

};

class Input {
public:
    string odomTimestamp;
    string imgTimestamp;
    string depthTimestamp;
    Odometry odom;
    Mat img;
    Mat depth;

    Input() {}
    Input(string odomTimestamp, Odometry odom, string imgTimestamp, Mat img, string depthTimestamp, Mat depth) {
        this->odomTimestamp = odomTimestamp;
        this->imgTimestamp = imgTimestamp;
        this->depthTimestamp = depthTimestamp;
        this->odom = odom;
        this->img = img;
        this->depth = depth;
    }

    void print() {
        cout << "timestamps: " << odomTimestamp << ", " << imgTimestamp << ", " << depthTimestamp << endl;
    }
};

int main(int argc, char** argv) {
    if (0) {
        // pose: camera_to_world
        // camera: world_to_camera
        Slam slam;
        slam.init();
        // camera z --> odom x
        // camera x --> odom -y
        // camera y --> odom -z
        //                0  1    2   3   4         5         6
        //                qw -qy -qz  qx -y        -z         x
        double pose1[] = {-0.1614, -0.7167, 0.6511, 0.1906, 2.3802, -0.5899, -2.2583};
        double pose2[] = {-0.1929, -0.7059, 0.6424, 0.2276, 2.3775, 0.5895, -2.2640};
        // -2.2583 -2.3802 0.5899 0.1906 0.7167 -0.6511 -0.1614
        // -2.2640 -2.3775 0.5895 0.2276 0.7059 -0.6424 -0.1929

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
        const string DATA_DIR   = "../data/";
        const string ODOM_PATH  = DATA_DIR + "groundtruth.txt";
        const string DEPTH_PATH = DATA_DIR + "depth.txt";
        const string IMG_PATH   = DATA_DIR + "rgb.txt";
        const string DEPTH_DIR  = DATA_DIR + "depth/";
        const string IMG_DIR    = DATA_DIR + "rgb/";
        const size_t ODOM_DOWNSAMPLE_RATE = 1;
        const size_t OBS_DOWNSAMPLE_RATE  = 15;
        const size_t INPUT_DOWNSAMPLE_RATE = 1;

        vector<pair<string, Odometry>> odometries;
        vector<pair<string, Mat>> images;
        vector<pair<string, Mat>> depths;
        ifstream fp;
        string line;
        string timestamp;
        size_t lineNum;

        fp.open(ODOM_PATH);
        if (!fp.is_open()) {
            printf("error in opening file %s\n", ODOM_PATH.c_str());
            exit(1);
        }

        // read odometry
        lineNum = 0;
        while ( getline(fp, line) ) {
            stringstream tokens(line);
            tokens >> timestamp;
            if (timestamp.compare("#") == 0) { continue; } // ignore comments
            ++lineNum;
            if (lineNum % ODOM_DOWNSAMPLE_RATE != 0) { continue; }
            float tmp[7];
            for (size_t i = 0; i < poseDim; ++i) { tokens >> tmp[i]; }
            Affine3f extrinsicCamera = Affine3f::Identity();
            extrinsicCamera.translate(Vector3f(0,0,0));
            extrinsicCamera.rotate(Quaternionf(0.5, 0.5, -0.5, 0.5));
            Vector3f translation(extrinsicCamera * Vector3f(tmp[0], tmp[1], tmp[2]));
            Quaternionf rotation((extrinsicCamera * Quaternionf(tmp[6], tmp[3], tmp[4], tmp[5])).rotation());
            Odometry odom(translation, rotation);
            odometries.emplace_back(timestamp, odom);
        }
        cout << "number of odometries: " <<  odometries.size() << endl;
        fp.close();

        // read image
        lineNum = 0;
        fp.open(IMG_PATH);
        if (!fp.is_open()) {
            printf("error in opening file %s\n", IMG_PATH.c_str());
            exit(1);
        }
        while( getline(fp, line) ) {
            stringstream tokens(line);
            tokens >> timestamp;
            if (timestamp.compare("#") == 0) { continue; } // ignore comments
            ++lineNum;
            if (lineNum % OBS_DOWNSAMPLE_RATE != 0) { continue; }
            string path;
            tokens >> path;
            Mat img = imread(DATA_DIR + path, IMREAD_UNCHANGED);
            if (img.data == NULL) {
                perror("cannot open image files");
                exit(1);
            }
            images.emplace_back(timestamp, img);
        }
        cout << "number of images: " <<  images.size() << endl;
        fp.close();

        // read depth
        lineNum = 0;
        fp.open(IMG_PATH);
        if (!fp.is_open()) {
            printf("error in opening file %s\n", IMG_PATH.c_str());
            exit(1);
        }
        while( getline(fp, line) ) {
            stringstream tokens(line);
            tokens >> timestamp;
            if (timestamp.compare("#") == 0) { continue; } // ignore comments
            ++lineNum;
            if (lineNum % OBS_DOWNSAMPLE_RATE != 0) { continue; }
            string path;
            tokens >> path;
            Mat depth = imread(DATA_DIR + path, IMREAD_UNCHANGED);
            if (depth.data == NULL) {
                perror("cannot open image files");
                exit(1);
            }
            depths.emplace_back(timestamp, depth);
        }
        cout << "number of depths: " <<  depths.size() << endl;
        fp.close();

        vector<Input> inputs;
        size_t odomIdx, depthIdx;
        odomIdx  = 0;
        depthIdx = 0;
        for (size_t imgIdx = 0; imgIdx < images.size(); ++imgIdx) {
            while (odomIdx  < odometries.size() && odometries[odomIdx].first < images[imgIdx].first) { ++odomIdx; }
            while (depthIdx < depths.size()     && depths[depthIdx].first    < images[imgIdx].first) { ++depthIdx; }
            inputs.emplace_back(odometries[odomIdx].first, odometries[odomIdx].second, 
                                images[imgIdx].first,        images[imgIdx].second,
                                depths[depthIdx].first,    depths[depthIdx].second);
        }

        if (0) {
            for (auto input : inputs) { input.print(); }
        }
        

        Slam slam;
        size_t n_pose = argc > 1 ? stoi(argv[1]) : 5;
        for (size_t t = 0; t < n_pose; ++t) {
            inputs[t].print();
            slam.observeOdometry(inputs[t].odom.loc, inputs[t].odom.angle);
            cout << "before observe Image" << endl;
            slam.observeImage(inputs[t].img, inputs[t].depth);
            cout << "after observe Image" << endl;
        }
        slam.displayCLMS();
        slam.dumpLandmarksToCSV("../data/results/tum-landmarks-initEstimate.csv");
        slam.optimize();
        slam.dumpLandmarksToCSV("../data/results/tum-landmarks.csv");
        slam.displayPoses();


        // slam.dumpLandmarksToCSV("../data/results/vslam-set2-landmarks-initEstimate.csv");
        // slam.optimize();
        // // slam.displayLandmarks();
        // // slam.displayPoses();
        // slam.dumpLandmarksToCSV("../data/results/vslam-set2-landmarks.csv");

    }
    
}