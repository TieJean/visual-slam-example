#include <fstream>
#include <cmath>

#include "feature_tracker.h"
#include "slam.h"


using namespace std;
using namespace gtsam;
using namespace slam;
using feature_tracker::FeatureTracker;

namespace slam {

Slam::Slam() :
    // isam(ISAM2Params(1)),
    K(new Cal3_S2(525.0, 525.0, 0.0, 319.5, 239.5)),
    img_dir("../data/gray/"),
    depth_dir("../data/depth/"),
    gt_ts_path("../data/groundtruth.txt"),
    img_ts_path("../data/rgb.txt"),
    depth_ts_path("../data/depth.tx"),
    pose_initialized(false),
    landmark_initialized(false),
    has_new_pose(false),
    prev_pose(Quaternion(0.0, 0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0)),
    tracker() {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        isam = ISAM2(parameters);
        // measurementNoise = noiseModel::Isotropic::Sigma(2, 1.0);
        // kPointPrior = noiseModel::Isotropic::Sigma(3, 0.1);
        // kPosePrior = noiseModel::Diagonal::Sigmas(
        //     (Vector(6) << Vector3::Constant(0.05), Vector3::Constant(0.1))
        //     .finished());
        
    }

void Slam::init() {
    // make sure we start with empty vectors
    poses.clear();
    landmarks.clear();
    img_features.clear();
    graph.resize(0);
    initialEstimate.clear();

    // ISAM2Params parameters;
    // parameters.relinearizeThreshold = 0.01;
    // parameters.relinearizeSkip = 1;
}

void Slam::imageToWorld(const int& v, const int& u, const Mat& depthImg, float* X_ptr, float* Y_ptr, float* Z_ptr) {
    float &X = *X_ptr;
    float &Y = *Y_ptr;
    float &Z = *Z_ptr;

    const float fx = 5.25;
    const float fy = 5.25;
    const float cx = 3.195;
    const float cy = 2.395;
    const float factor = 5000.0;

    cout << "depth: " << depthImg.at<float>(v, u) << endl;
    Z = depthImg.at<float>(v, u) / factor; // TODO: FIXME; unbelievablly small
    X = (u-cx) * Z / fx;
    Y = (v-cy) * Z / fy;
}
 
void Slam::observeImage(const string& t) {
    if (!has_new_pose) {return;}
    
    vector<KeyPoint> kp;
    Mat des;
    tracker.getKpAndDes(t, &kp, &des);
    pair<vector<KeyPoint>, Mat> img_feature(kp, des);
    
    if (img_features.size() < 1) {
        img_features[t] = img_feature;
        return;
    }

    for (auto feature : img_features) {
        string prev_t   = feature.first;
        vector<KeyPoint> prev_kp  = feature.second.first;
        Mat prev_des = feature.second.second;
        vector<DMatch> matches;
        // correspondance matching
        tracker.match(prev_kp, prev_des, kp, des, &matches);

        Mat depthImg = imread(depth_dir + "1311877813.024161" + ".png", 0);
        if (depthImg.data == NULL) {
            perror("cannot open depth files");
            exit(1);
        }
        for (auto match : matches) {
            // in opencv, keypoints are not in image coordinate
            int x = prev_kp[match.queryIdx].pt.x;
            int y = prev_kp[match.queryIdx].pt.y;

            struct TimeKp kp2add_prev;
            kp2add_prev.t = prev_t;
            kp2add_prev.kp = prev_kp[match.queryIdx];
            struct TimeKp kp2add;
            kp2add.t = t;
            kp2add.kp = kp[match.trainIdx];

            PinholeCamera<Cal3_S2> camera(poses.back(), *K);
            Point2 measurement(y, x);
            auto measurementNoise = noiseModel::Isotropic::Sigma(2, 1.0);
            printf("measurement: %2f, %2f\n", measurement.x(),measurement.y());
            // check if this landmark has already been added to the graph
            if (added_keypoints.contains(kp2add_prev)) { 
                // if so, retrieve previously-added landmark and add emplace_shared
                added_keypoints[kp2add] = added_keypoints[kp2add_prev];
                graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(measurement, measurementNoise, 
                                    Symbol('x', poses.size()-1), Symbol('l', added_keypoints[kp2add_prev]), K);
                continue; 
            }
            added_keypoints[kp2add] = landmarks.size();
            added_keypoints[kp2add_prev] = landmarks.size();
            graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(measurement, measurementNoise, Symbol('x', poses.size()-1), Symbol('l', landmarks.size()), K);
            
            float X, Y, Z;
            imageToWorld(x, y, depthImg, &X, &Y, &Z);
            printf("world: %2f, %2f, %2f\n", X,Y,Z);
            initialEstimate.insert<Point3>(Symbol('l', landmarks.size()), Point3(X,Y,Z));
            if (landmarks.size() == 1) {
                // dim = 3, sigma = 0.1
                static auto kPointPrior = noiseModel::Isotropic::Sigma(3, 0.1); // TODO: FIXME
                graph.addPrior(Symbol('l', 0), Point3(X,Y,Z), kPointPrior);
            }
            landmarks.push_back(Point3(X,Y,Z));
        }
    }
    img_features[t] = img_feature;
    has_new_pose = false;

}

float Slam::quaternion2Radian(const Quaternion& quaternion) {
    return 2 * acos(quaternion.w());
}

Quaternion Slam::getQuaternionDelta(const Quaternion& a1, const Quaternion& a2) {
    return a2 * a1.inverse();
}

Pose3 Slam::getPoseDelta(const Pose3& pose1, const Pose3& pose2) {
    return Pose3( Rot3( getQuaternionDelta( pose1.rotation().toQuaternion(), pose2.rotation().toQuaternion() ) ), 
                        Point3(pose2.x()-pose1.x(), pose2.y()-pose1.y(), pose2.z()-pose1.z()) );
}

void Slam::observeOdometry(const Pose3& odom) {
    Pose3 odom_delta = poses.size() == 0? Pose3(Rot3::Quaternion(0.0,0.0,0.0,0.0), Point3(0.0,0.0,0.0)) : getPoseDelta(poses.back(), odom);
    if (poses.size() != 0 && 
        (odom_delta.translation().norm() < MIN_D_DELTA || quaternion2Radian(odom_delta.rotation().toQuaternion()) < MIN_A_DELTA) ) {
        return;
    } 
    initialEstimate.insert(Symbol('x', poses.size()), odom);
    if (poses.size() == 0) {
        static auto kPosePrior = noiseModel::Diagonal::Sigmas(
            (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.1))
            .finished()); // TODO: FIXME
        // cout << "x" << poses.size() << ": " << odom << endl
        graph.addPrior(Symbol('x', poses.size()), odom, kPosePrior);
    } else {
        static auto odomNoise = noiseModel::Diagonal::Sigmas(
            (Vector(6) << Vector3::Constant(0.05), Vector3::Constant(0.1))
            .finished()); // TODO: FIXME
        Symbol prev_pose('x', poses.size()-1), pose('x', poses.size());
        graph.emplace_shared<BetweenFactor<Pose3>>(prev_pose, pose, odom_delta, odomNoise);
    }
    poses.push_back(odom);
    has_new_pose = true;
}

void Slam::optimize() {
    graph.print("graph: ");
    initialEstimate.print("initialEstimate: ");
    cout << "isam_addr: " << &isam << endl;
    cout << "graph_addr: " << &graph << endl;
    cout << "initialEstimate_addr: " << &initialEstimate << endl;
    // isam.update(graph, initialEstimate);
    isam.update();
    // Values currentEstimate = isam.calculateEstimate();
    // cout << "****************************************************" << endl;
    // currentEstimate.print("Current estimate: ");
}

}

