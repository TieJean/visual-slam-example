#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"

#include "slam.h"


using namespace std;
using namespace ceres;

namespace slam {

Slam::Slam() :
    feature_tracker(),
    prev_odom_loc_(0.0,0.0,0.0),
    prev_odom_angle_(0.0,0.0,0.0,0.0),
    has_new_pose_(false) {
        // poses = new vector<pair<Vector3f, Quaternionf>>();
        // landmarks = new vector<Vector3f>();
        // observations = new vector<Vector2f>();
        // lp_map = new vector<vector<int>>();
        // vector<pair<vector<KeyPoint>, Mat>> features;
    }

void Slam::init() {
    poses.clear();
    landmarks.clear();
    // observations.clear();
    measurements.clear();
    lp_map.clear();
    features.clear();
    added_keypoints.clear();
}

void Slam::observeImage(const Mat& img, const Mat& depth) {
    if (!has_new_pose_) { return; }
    has_new_pose_ = false;

    vector<KeyPoint> kp_curr;
    Mat des_curr;
    feature_tracker.getKpAndDes(img, &kp_curr, &des_curr);
    features.emplace_back(kp_curr, des_curr);

    if (features.size() < 2) { return; }

    vector<KeyPoint> kp;
    Mat des;
    vector<DMatch> matches;
    struct TimeKp timeKp, timeKp_curr;
    size_t T = features.size() - 1;
    timeKp_curr.t = T;

    for (size_t t = 0; t < T; ++t) {
        kp = features[t].first;
        des = features[t].second;
        // des_curr: query; des: train
        feature_tracker.match(des_curr, des, &matches);
        for (auto match : matches) {
            timeKp.t = t;
            timeKp.kp = kp[match.trainIdx];
            timeKp_curr.kp = kp_curr[match.queryIdx];
            int x = kp[match.trainIdx].pt.x;
            int y = kp[match.trainIdx].pt.y;
            if (added_keypoints.contains(timeKp)) {
                added_keypoints[timeKp_curr] = added_keypoints[timeKp];
                // added_keypoints[timeKp] - landmark; t - new pose
                lp_map[added_keypoints[timeKp]].push_back(t);
                // observations[pair<int, int>(added_keypoints[timeKp], T)] = Vector2f(x, y);
                continue;
            }
            
            // in opencv, keypoints are not in image coordinate
            
            // observations[pair<int, int>(landmarks.size()-1, t)] = Vector2f(x, y);
            // observations[pair<int, int>(landmarks.size()-1, T)] = Vector2f(x, y);
            measurements.emplace_back(x, y);
            float X, Y, Z;
            imgToWorld_(x, y, depth, &X, &Y, &Z);
            landmarks.emplace_back(X, Y, Z);
            added_keypoints[timeKp]      = landmarks.size() - 1;
            added_keypoints[timeKp_curr] = landmarks.size() - 1;
            // TODO: do i need to initialize vector??
            if (lp_map.size() < landmarks.size()) { lp_map.resize(landmarks.size()); }
            lp_map[landmarks.size() - 1].push_back(t);
            lp_map[landmarks.size() - 1].push_back(T);
            // if (landmarks.size() == 5) {
            //     cout << added_keypoints[timeKp] << endl;
            //     cout << added_keypoints[timeKp] << endl;
            // }
        }
    }

}

void Slam::observeOdometry(const Vector3f& odom_loc ,const Quaternionf& odom_angle) {
    if ( poses.size() != 0 || getDist_(prev_odom_loc_, odom_loc) > MIN_DELTA_D || getQuaternionDelta_(prev_odom_angle_, odom_angle).norm() < MIN_DELTA_A ) { 
        prev_odom_loc_ = odom_loc;
        prev_odom_angle_ = odom_angle;
        poses.emplace_back(prev_odom_loc_, prev_odom_angle_);
        has_new_pose_ = true; 
    }
}

vector<pair<Vector3f, Quaternionf>>& Slam::getPoses() { return poses; }

vector<Vector3f>& Slam::getLandmarks() {return landmarks; }

void Slam::optimize(bool minimizer_progress_to_stdout, bool briefReport, bool fullReport) {
    // for (size_t i = 0; i < lp_map.size(); ++i) {
    //     cout << i << ": ";
    //     for (size_t j = 0; j < lp_map[i].size(); ++j) {
    //         cout << lp_map[i][j] << ", ";
    //     }
    //     cout << endl;
    // }

    Problem problem;
    // double** opt_poses = new double*[poses.size()];
    // double** opt_landmarks = new double*[landmarks.size()];
    // for (size_t i = 0; i < poses.size(); ++i) {
    //     opt_poses[i] = new double[7];
    // }
    // for (size_t i = 0; i < landmarks.size(); ++i) {
    //     opt_landmarks[i] = new double[3];
    // }

    // for (size_t i = 0; i < poses.size(); ++i) {
    //     opt_poses[i][0]     = poses[i].first.x();
    //     opt_poses[i][1]     = poses[i].first.y();
    //     opt_poses[i][2]     = poses[i].first.z();
    //     opt_poses[i][3]     = poses[i].second.w();
    //     opt_poses[i][4]     = poses[i].second.x();
    //     opt_poses[i][5]     = poses[i].second.y();
    //     opt_poses[i][6]     = poses[i].second.z();
    // }
    // for (size_t i = 0; i < landmarks.size(); ++i) {
    //     opt_landmarks[i][0] = landmarks[i].x();
    //     opt_landmarks[i][1] = landmarks[i].y();
    //     opt_landmarks[i][2] = landmarks[i].z();
    // }
    // double* observations_x = new double[observations.size()];
    // double* observations_y = new double[observations.size()];
    // for (size_t i = 0; i < observations.size(); ++i) {
    //     observations_x[i] = observations[i].x();
    //     observations_y[i] = observations[i].y();
    // }
/*
    for (size_t i = 0; i < landmarks.size(); ++i) {
        if (i >= 27) {continue;}
        for (size_t j = 0; j < lp_map[i].size(); ++j) {
            if (!observations.contains(pair<int,int>(i,j))) { continue;}
            Vector2f obs = observations[pair<int,int>(i,j)];
            CostFunction* cost_function = ReprojectionError::Create(obs.x(), obs.y());
            if (lp_map[i][j] < 0 || lp_map[i][j] >= poses.size() || j >= 1) { 
                continue;
            }
            // cout << lp_map[i].size() << endl;
            // problem.AddResidualBlock(cost_function, NULL, opt_poses[lp_map[i][j]], opt_landmarks[i]);
            cout << landmarks.size() << ": " << i << endl;
        }
        // double camera[] = {poses[i].first.x(),  poses[i].first.y(),  poses[i].first.z(),
        //                    poses[i].second.w(), poses[i].second.x(), poses[i].second.y(), poses[i].second.z()};
        // double point[] = {landmarks[i].x(), landmarks[i].y(), landmarks[i].z()};
        // NULL - least square loss
        
        // poses[i] = pair<Vector3f, Quaternionf>(Vector3f(camera[0], camera[1], camera[2]),
        //                                       Quaternionf(camera[3], camera[4], camera[5], camera[6])); 
        // landmarks[i] = Vector3f(point[0], point[1], point[2]);
    }
    */
    for (size_t i = 0; i < landmarks.size(); ++i) {
        CostFunction* cost_function = ReprojectionError::Create(measurements[i].x(), measurements[i].y());
        double camera[] = {poses[i].first.x(),  poses[i].first.y(),  poses[i].first.z(),
                           poses[i].second.w(), poses[i].second.x(), poses[i].second.y(), poses[i].second.z()};
        double point[] = {landmarks[i].x(), landmarks[i].y(), landmarks[i].z()};
        problem.AddResidualBlock(cost_function, NULL, camera, point);
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = minimizer_progress_to_stdout;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (briefReport) std::cout << summary.FullReport() << "\n";
    if (fullReport)  std::cout << summary.BriefReport() << "\n";
    // for (size_t i = 0; i < landmarks.size(); ++i) {
    //     poses[i] = pair<Vector3f, Quaternionf>(Vector3f(opt_poses[i][0],    opt_poses[i][1], opt_poses[i][2]),
    //                                            Quaternionf(opt_poses[i][3], opt_poses[i][4], opt_poses[i][5], opt_poses[i][6])); 
    //     landmarks[i] = Vector3f(opt_landmarks[i][0], opt_landmarks[i][1], opt_landmarks[i][2]);
    // }
    // for (size_t i = 0; i < poses.size(); ++i) {
    //     delete[] opt_poses[i];
    // }
    // for (size_t i = 0; i < landmarks.size(); ++i) {
    //     delete[] opt_landmarks[i];
    // }
    // delete opt_poses;
    // delete opt_landmarks;
    // delete observations_x;
    // delete observations_y;
    
}

Quaternionf Slam::getQuaternionDelta_(const Quaternionf& a1, const Quaternionf& a2) {
    return a2 * a1;
}

float Slam::getDist_(const Vector3f& odom1, const Vector3f& odom2) {
    float x_pow = pow(odom1.x()-odom2.x(), 2);
    float y_pow = pow(odom1.y()-odom2.y(), 2);
    float z_pow = pow(odom1.z()-odom2.z(), 2);
    return sqrt( x_pow + y_pow + z_pow);
}

void Slam::imgToWorld_(const size_t& x, const size_t& y, const Mat& depth,
                 float* X_ptr, float* Y_ptr, float* Z_ptr) {
    float &X = *X_ptr;
    float &Y = *Y_ptr;
    float &Z = *Z_ptr;

    float factor = 5000;

    Z = depth.at<float>(y, x) / factor;
    X = (x - cx) * Z / fx;
    Y = (Y - cy) * Z / fy;
}

}
