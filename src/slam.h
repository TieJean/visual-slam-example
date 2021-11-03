#include <vector>
#include <utility>
#include <unordered_map>
#include <math.h>
#include <iomanip>
#include <functional>
#include <unordered_set>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/Values.h"
#include "gtsam/slam/ProjectionFactor.h"
#include "gtsam/nonlinear/NonlinearISAM.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/nonlinear/ISAM2.h"
#include "gtsam/nonlinear/Values.h"
#include "feature_tracker.h"

using namespace std;
using std::hash;
using gtsam::NonlinearFactorGraph;
using gtsam::Values;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Point3;
using gtsam::Cal3_S2;
using gtsam::NonlinearISAM;
using gtsam::Quaternion;
using gtsam::Vector;
using gtsam::Vector3;
using gtsam::ISAM2;
using gtsam::ISAM2Params;
using feature_tracker::FeatureTracker;

#ifndef SLAM_H_
#define SLAM_H_

namespace slam {

struct TimeKp {
    string t;
    KeyPoint kp; 
    bool operator==(const TimeKp &other) const
    { return (t.compare(other.t) == 0
                && kp.pt.x == other.kp.pt.x
                && kp.pt.y == other.kp.pt.y);
    }
};

struct TimeKpHash
{
    std::size_t operator()(const TimeKp& p) const noexcept
    {
        size_t h0 = hash<string>{}(p.t); 
        size_t h1 = hash<int>{}(p.kp.pt.x);
        size_t h2 = hash<int>{}(p.kp.pt.y);
        return h0 ^ (h1 << 1) ^ (h2 >> 1);
    }
};


class Slam {
    
public: 
    NonlinearFactorGraph graph;
    Values initialEstimate;

    Slam();

    void init();
    void observeImage(const string& t);
    void observeOdometry(const Pose3& odom);
    void optimize();


private:
    // NonlinearFactorGraph graph;
    // Values initialEstimate;
    // NonlinearISAM isam;
    ISAM2 isam;
    Cal3_S2::shared_ptr K;
    vector<Pose3> poses;
    vector<Point3> landmarks;
    string img_dir;
    string depth_dir;
    string gt_ts_path;
    string img_ts_path;
    string depth_ts_path;

    bool pose_initialized;
    bool landmark_initialized;
    bool has_new_pose;
    Pose3 prev_pose;
    FeatureTracker tracker;
    std::unordered_map< string, pair<vector<KeyPoint>, Mat> > img_features;
    // pair<string, KeyPoint> 
    // static constexpr auto h = [](const pair<string, KeyPoint>& p) {
    //     size_t h0 = hash<string>{}(p.first); 
    //     size_t h1 = hash<int>{}(p.second.pt.x);
    //     size_t h2 = hash<int>{}(p.second.pt.y);
    //     return h0 ^ (h1 << 1) ^ (h2 << 2);
    // };
    // {<img_timestamp, keypoint> : landmark_idx }
    std::unordered_map< TimeKp, int, TimeKpHash > added_keypoints;

    size_t n_pose;
    size_t n_landmark;

    // auto measurementNoise = noiseModel::Isotropic::Sigma(2, 1.0);
    // static auto kPosePrior = = noiseModel::Isotropic::Sigma(3, 0.1);
    // static auto kPointPrior = noiseModel::Diagonal::Sigmas(
    //         (Vector(6) << Vector3::Constant(0.05), Vector3::Constant(0.1))
    //         .finished());

    const float MIN_D_DELTA = 0.15;
    const float MIN_A_DELTA = M_PI / 180 * 15;
    

    float quaternion2Radian(const Quaternion& quaternion);
    Quaternion getQuaternionDelta(const Quaternion& a1, const Quaternion& a2);
    Pose3 getPoseDelta(const Pose3& pose1, const Pose3& pose2);
    void imageToWorld(const int& x, const int& y, const Mat& depthImg, float* X, float* Y, float* Z);
};

}

#endif
