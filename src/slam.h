#include <vector>
#include <utility>
#include <unordered_map>
#include <cmath>
#include <array>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"

#include "feature_tracker.h"

using namespace std;
using namespace ceres;
using namespace Eigen;
using namespace cv;
using namespace feature_tracker;

#ifndef SLAM_H_
#define SLAM_H_

namespace slam {

struct TimeKp {
    size_t t;
    KeyPoint kp; 

    TimeKp() {}

    TimeKp(int t, KeyPoint kp) : t(t), kp(kp) {}

    bool operator==(const TimeKp &other) const
    { return (t == other.t
           && kp.pt.x == other.kp.pt.x
           && kp.pt.y == other.kp.pt.y);
    }
};

struct TimeKpHash
{
    std::size_t operator()(const TimeKp& p) const noexcept
    {
        size_t h0 = hash<size_t>{}(p.t); 
        size_t h1 = hash<int>{}(p.kp.pt.x);
        size_t h2 = hash<int>{}(p.kp.pt.y);
        return h0 ^ (h1 << 1) ^ (h2 >> 1);
    }
};

struct LP {
    size_t landmark;
    size_t pose;
    LP() {}
    LP(size_t landmark, size_t pose) : landmark(landmark), pose(pose) {}
    bool operator==(const LP &other) const
    { return (landmark == other.landmark
           && pose     == other.pose);
    }
};

struct LPHash
{
    std::size_t operator()(const pair<int, int>& p) const noexcept
    {
        size_t h0 = hash<int>{}(p.first);
        size_t h1 = hash<int>{}(p.second);
        return h0 ^ (h1 << 1);
    }
};

const float fx = 520.9;
const float fy = 521.0;
const float cx = 325.1;
const float cy = 249.7;
const float d0 = 0.2312;
const float d1 = -0.7849;
const float d2 = -0.0033;
const float d3 = -0.0001;
const float d4 = 0.9172;
const float ds = 1.031;

struct ReprojectionError {
    float observed_x;
    float observed_y;

    ReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}
    
    // camera - 0,1,2,3 angle-axis rotation
    //        - 4,5,6 translation
    template <typename T>
    bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
        T p[3];
        ceres::QuaternionRotatePoint(camera, point, p); // rotation
        p[0] += camera[4]; p[1] += camera[5]; p[2] += camera[6]; // translation
        
        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];
        T predicted_x = - xp * T(fx) + T(cx);
        T predicted_y = - yp * T(fy) + T(cy);
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x,
                                      const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 7, 3>(
                    new ReprojectionError(observed_x, observed_y)));
    }

};

class Slam {

public: 
    Slam();

    void init();
    void observeImage(const Mat& img, const Mat& depth);
    void observeOdometry(const Vector3f& odom_loc ,const Quaternionf& odom_angle);
    void optimize(bool minimizer_progress_to_stdout, bool briefReport, bool fullReport);
    vector<pair<Vector3f, Quaternionf>>& getPoses();
    vector<Vector3f>& getLandmarks();

private:
    FeatureTracker feature_tracker;
    Vector3f prev_odom_loc_;
    Quaternionf prev_odom_angle_;
    bool odom_initialized_;
    bool pose_initialized_;
    bool landmark_initialized_;
    bool has_new_pose_;
    vector<pair<Vector3f, Quaternionf>> poses;
    vector<Vector3f> landmarks;
    // vector<Vector2f> observations;
    // unordered_map<pair<int, int>, Vector2f, LPHash> observations;
    vector<Vector2f> measurements;
    // lp_map[i]: for ith landmarks, all cameras it has appearred on 
    vector<vector<int>> lp_map;
    vector<pair<vector<KeyPoint>, Mat>> features;
    // TimeKp : landmark
    unordered_map<TimeKp, int, TimeKpHash> added_keypoints;

    float MIN_DELTA_D = 0.5;
    float MIN_DELTA_A = 0.8;

    Quaternionf getQuaternionDelta_(const Quaternionf& a1, const Quaternionf& a2);
    float getDist_(const Vector3f& odom1, const Vector3f& odom2);
    void imgToWorld_(const size_t& u, const size_t& v, const Mat& depth,
                     float* X_ptr, float* Y_ptr, float* Z_ptr);

};

} 

#endif