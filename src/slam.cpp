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
    prev_odom_loc_(0.0,0.0,0.0),
    prev_odom_angle_(0.0,0.0,0.0,0.0),
    has_new_pose_(false) {
        extrinsicCamera = Affine3f::Identity();
        extrinsicCamera.translate(Vector3f(0,0,0));
        extrinsicCamera.rotate(Quaternionf(0.5, 0.5, -0.5, 0.5));
    }

void Slam::init(size_t N_POSE, size_t N_LANDMARK) {
    // TODO: clear all vector
    points.resize(N_LANDMARK);
    point_cnts.resize(N_LANDMARK);
    for (size_t i = 0; i < N_LANDMARK; ++i) {
        point_cnts[i] = 0;
    }
}

void Slam::initLandmarks(const vector<Vector3f>& landmarks, const size_t n_landmark) {
    cout << "points: " << points.size() << endl;
    for (size_t i = 1; i <= n_landmark; ++i) {
        Vector3f landmarkExtrinsic = extrinsicCamera * landmarks[i-1];
        points[i] = new double[3] {landmarkExtrinsic.x(), landmarkExtrinsic.y(), landmarkExtrinsic.z()};
    }
}

void Slam::observeImage(const vector<Measurement>& observation) {
    vector<Measurement> prev_observation; 
    float prev_pred[3];
    float pred[3];
    const size_t T = cameras.size() - 1; // last pose idx
    // cout << "observeImage: " << T << endl;

    if (!has_new_pose_) { return; }
    has_new_pose_ = false;
    observations.push_back(observation);
    if (cameras.size() < 2) { return; }
    for (size_t t = 0; t < cameras.size() - 1; ++t) {
        prev_observation = observations[t];
        if (prev_observation.size() == 0) {continue;}
        // iterate through all current measurements
        size_t idx_prev = 0;
        for (size_t idx = 0; idx < observation.size(); ++idx) { 
            while (idx_prev < prev_observation.size() && prev_observation[idx_prev].landmarkIdx < observation[idx].landmarkIdx) {
                ++idx_prev;
            }
            if (prev_observation[idx_prev].landmarkIdx == observation[idx].landmarkIdx) { // correspondance matching
                size_t landmarkIdx = prev_observation[idx_prev].landmarkIdx;
                imgToWorld_(cameras[t], prev_observation[idx_prev].measurementX, 
                                        prev_observation[idx_prev].measurementY, 
                                        prev_observation[idx_prev].depth, 
                                        &prev_pred[0], &prev_pred[1], &prev_pred[2]);
                imgToWorld_(cameras[T], observation[idx].measurementX,
                                        observation[idx].measurementY,
                                        observation[idx].depth,
                                        &pred[0], &pred[1], &pred[2]);
                // cout << idx_prev << "," << idx << ": " << observation[idx].landmarkIdx << endl;
                // printf("%ld: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f | %.2f,%.2f,%.2f | %.2f,%.2f,%.2f\n", t,
                //         cameras[t][0], cameras[t][1], cameras[t][2], cameras[t][3], cameras[t][4], cameras[t][5], cameras[t][6], 
                //         prev_observation[idx_prev].measurementX, prev_observation[idx_prev].measurementY, prev_observation[idx_prev].depth,
                //         prev_pred[0], prev_pred[1], prev_pred[2] );
                // printf("%ld: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f | %.2f,%.2f,%.2f | %.2f,%.2f,%.2f\n", T,
                //         cameras[T][0], cameras[T][1], cameras[T][2], cameras[T][3], cameras[T][4], cameras[T][5], cameras[T][6], 
                //         observation[idx].measurementX, observation[idx].measurementY, observation[idx].depth,
                //         pred[0], pred[1], pred[2] );
                // cout << endl;
                float thresh = 0.15;
                if ( point_cnts[landmarkIdx]  == 0 ) {
                    if ( abs(pred[0] - prev_pred[0]) > thresh
                      || abs(pred[1] - prev_pred[1]) > thresh
                      || abs(pred[2] - prev_pred[2]) > thresh) {
                          printf("inconsistent! %ld\n", observation[idx].landmarkIdx);
                          printf("landmark prev: %.2f, %.2f, %.2f\n", prev_pred[0], prev_pred[1], prev_pred[2]);
                          printf("landmark curr:  %.2f, %.2f, %.2f\n", pred[0], pred[1], pred[2]);
                    }
                    double* point = new double[]{ (prev_pred[0] + pred[0]) / 2, 
                                                  (prev_pred[1] + pred[1]) / 2,
                                                  (prev_pred[2] + pred[2]) / 2 };
                    points[landmarkIdx] = point;
                    point_cnts[landmarkIdx] = 2;
                    // cout << "measurement (case1)" << endl;
                    // printf("%ld, %ld: %.2f, %.2f\n", t+1, landmarkIdx, prev_observation[idx_prev].measurementX,  prev_observation[idx_prev].measurementY);
                    // printf("%ld, %ld: %.2f, %.2f\n", T+1, landmarkIdx,      observation[idx].measurementX,       observation[idx].measurementY);
                    clms.emplace_back(t, landmarkIdx, prev_observation[idx_prev].measurementX,  prev_observation[idx_prev].measurementY);
                    clms.emplace_back(T, landmarkIdx,      observation[idx].measurementX,       observation[idx].measurementY);
                } else {
                    if ( abs(pred[0] - points[landmarkIdx][0]) > thresh
                      || abs(pred[1] - points[landmarkIdx][1]) > thresh
                      || abs(pred[2] - points[landmarkIdx][2]) > thresh) {
                          printf("outlier! %ld\n", observation[idx].landmarkIdx);
                          printf("landmark prev: %.2f, %.2f, %.2f\n", points[landmarkIdx][0], points[landmarkIdx][1], points[landmarkIdx][2]);
                          printf("landmark curr:  %.2f, %.2f, %.2f\n", pred[0], pred[1], pred[2]);
                    }
                    // cout << "measurement (case2)" << endl;
                    // printf("%ld, %ld: %.2f, %.2f\n", T+1, landmarkIdx,      observation[idx].measurementX,       observation[idx].measurementY);
                    points[landmarkIdx][0] = (points[landmarkIdx][0] * point_cnts[landmarkIdx] + pred[0]) / (point_cnts[landmarkIdx] + 1);
                    points[landmarkIdx][1] = (points[landmarkIdx][1] * point_cnts[landmarkIdx] + pred[1]) / (point_cnts[landmarkIdx] + 1);
                    points[landmarkIdx][2] = (points[landmarkIdx][2] * point_cnts[landmarkIdx] + pred[2]) / (point_cnts[landmarkIdx] + 1);
                    ++point_cnts[landmarkIdx];
                    clms.emplace_back(T, landmarkIdx,      observation[idx].measurementX,       observation[idx].measurementY);
                }
                ++idx_prev;
            } else { // prev_observation[idx_prev].landmarkIdx > observation[idx].landmarkIdx)
                continue;
            }
        }
    }
}

int Slam::clmsFind_(const CLM& clm) {
    for (int i = 0; i < clms.size(); ++i) {
        if (clm == clms[i]) {return i;}
    }
    return -1;
}

void Slam::observeOdometry(const Vector3f& odom_loc ,const Quaternionf& odom_angle) {
    // if ( poses.size() != 0 || getDist_(prev_odom_loc_, odom_loc) > MIN_DELTA_D || getQuaternionDelta_(prev_odom_angle_, odom_angle).norm() < MIN_DELTA_A ) { 
        prev_odom_loc_ = odom_loc;
        prev_odom_angle_ = odom_angle;

        // assume "pose" and "camera" are transformations in camera coordinate
        Affine3f odom_to_world = Affine3f::Identity();
        odom_to_world.translate(odom_loc);
        odom_to_world.rotate(odom_angle);
        // odom_to_world = extrinsicCamera * odom_to_world;
        // cout << "quat: "  << Quaternionf(odom_to_world.rotation()) << endl;
        // cout << "angle: " << AngleAxisf(odom_to_world.rotation()).angle() << endl;
        // cout << "axis: "  << AngleAxisf(odom_to_world.rotation()).axis() << endl;
        // cout << "loc: "   << odom_to_world.translation() << endl;
        Affine3f world_to_odom = odom_to_world.inverse();
        Quaternionf angle(world_to_odom.rotation());
        Vector3f loc(world_to_odom.translation());
        double* camera = new double[] {angle.w(), angle.x(), angle.y(), angle.z(),
                                       loc.x(), loc.y(), loc.z()};
        cameras.push_back(camera);

        // printf("\n------observeOdometry------------\n");
        // printf("pose: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]);
        // printf("camera: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", camera[0], camera[1], camera[2], camera[3], camera[4], camera[5], camera[6]);

        has_new_pose_ = true; 
    // }
}

void Slam::evaluate() {
    for (size_t i = 0; i < clms.size(); ++i) {
        Problem problem;
        CostFunction* cost_function = ReprojectionError::Create(clms[i].measurement[0], clms[i].measurement[1]);
        problem.AddResidualBlock(cost_function, NULL, cameras[clms[i].poseIdx], points[clms[i].landmarkIdx]);
        double cost;
        problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
        printf("measurement: %.2f, %.2f\n", clms[i].measurement[0], clms[i].measurement[1]);
        printf("camera: %ld - %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", clms[i].poseIdx+1, cameras[clms[i].poseIdx][0], cameras[clms[i].poseIdx][1], cameras[clms[i].poseIdx][2], cameras[clms[i].poseIdx][3], cameras[clms[i].poseIdx][4], cameras[clms[i].poseIdx][5], cameras[clms[i].poseIdx][6]);
        printf("points: %ld - %.2f,%.2f,%.2f\n", clms[i].landmarkIdx, points[clms[i].landmarkIdx][0], points[clms[i].landmarkIdx][1], points[clms[i].landmarkIdx][2]);
        printf("cost:   %.2f\n", cost);
        cout << endl;
    }
}

bool Slam::optimize() {
    cout << "optimize" << endl;
    Problem problem;
    if (observations.size() < 1) {
        printf("unexpected error: observations size shouldn't be 0\n");
        exit(1);
    }

    // debug start
    ofstream fp;
    fp.open("../data/results/clms.csv", ios::trunc);
    if (!fp.is_open()) {
        printf("error in opening file\n");
        exit(1);
    }
    for (size_t i = 0; i < clms.size(); ++i) {
        // cout << i << endl;
        // printf("measurement: %.2f, %.2f\n", clms[i].measurement[0], clms[i].measurement[1]);
        // printf("camera: %ld - %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", clms[i].poseIdx+1, cameras[clms[i].poseIdx][0], cameras[clms[i].poseIdx][1], cameras[clms[i].poseIdx][2], cameras[clms[i].poseIdx][3], cameras[clms[i].poseIdx][4], cameras[clms[i].poseIdx][5], cameras[clms[i].poseIdx][6]);
        // printf("points: %ld - %.2f,%.2f,%.2f\n", clms[i].landmarkIdx, points[clms[i].landmarkIdx][0], points[clms[i].landmarkIdx][1], points[clms[i].landmarkIdx][2]);
        // cout << endl;
        fp << clms[i].poseIdx+1 << "," << clms[i].landmarkIdx << "," << clms[i].measurement[0] << "," << clms[i].measurement[1] << endl;
    }
    // cout << "start optimize" << endl;
    // end debug

    for (size_t i = 0; i < clms.size(); ++i) {
        CostFunction* cost_function = ReprojectionError::Create(clms[i].measurement[0], clms[i].measurement[1]);
        problem.AddResidualBlock(cost_function, new HuberLoss(huberLossScale), cameras[clms[i].poseIdx], points[clms[i].landmarkIdx]);
    }
    // problem.SetParameterBlockConstant(cameras[0]); // TODO: delete me after debugging

    Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    return (summary.termination_type == ceres::CONVERGENCE 
         || summary.termination_type == ceres::USER_SUCCESS);
    // return false;
}

// void Slam::displayCLMS() {
//     printf("-------display inputs---------\n");
//     for (size_t i = 0; i < clms.size(); ++i) {
//         printf("%ld:         %ld | %ld\n", i, clms[i].poseIdx, clms[i].landmarkIdx);
//         printf("camera:      %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
//             poses[clms[i].poseIdx][0], poses[clms[i].poseIdx][1], poses[clms[i].poseIdx][2], poses[clms[i].poseIdx][3],
//             poses[clms[i].poseIdx][4], poses[clms[i].poseIdx][5], poses[clms[i].poseIdx][6]);
//         printf("landmark:    %.2f | %.2f | %.2f \n", points[clms[i].landmarkIdx][0], points[clms[i].landmarkIdx][1], points[clms[i].landmarkIdx][2]);
//         printf("measurement: %.2f | %.2f \n\n", clms[i].measurement[0], clms[i].measurement[1]);
//     }
// }

bool vectorContains_(const vector<double*>& vec, double* elt, size_t size) {
    for (auto v : vec) {
        bool ret = true;
        for (size_t i = 0; i < size; ++i) {
            ret &= (v[i] == elt[i]);
        }
        if (ret) {return true;}
    }
    return false;
}

void Slam::displayPoses() {
    // printf("----------cameras----------\n");
    // for (size_t i = 0; i < cameras.size(); ++i) {
    //     printf("%ld: %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
    //         i, cameras[i][0], cameras[i][1], cameras[i][2], cameras[i][3], cameras[i][4], cameras[i][5], cameras[i][6]);
    // }
    printf("----------poses----------\n");
    for (size_t i = 0; i < cameras.size(); ++i) {
        Affine3f world_to_camera = Affine3f::Identity();
        world_to_camera.translate(Vector3f(cameras[i][4], cameras[i][5], cameras[i][6]));
        world_to_camera.rotate(Quaternionf(cameras[i][0], cameras[i][1], cameras[i][2], cameras[i][3]));
        Affine3f camera_to_world = world_to_camera.inverse();
        Vector3f loc = extrinsicCamera.inverse() * camera_to_world.translation();
        AngleAxisf angle_a(camera_to_world.rotation());
        angle_a = AngleAxisf(angle_a.angle(), extrinsicCamera.inverse() * angle_a.axis());
        Quaternionf angle(angle_a);
        printf("%ld: %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
            i, loc.x(), loc.y(), loc.z(), angle.x(), angle.y(), angle.z(), angle.w());
    }
}

void Slam::displayLandmarks() {
    printf("----------landmarks----------\n");
    for (size_t i = 0; i < points.size(); ++i) {
        if (point_cnts[i] != 0) {
            printf("%ld: %.2f | %.2f | %.2f \n", i, points[i][2], -points[i][0], -points[i][1]);
        }
    }
}

void Slam::dumpLandmarksToCSV(string path) {
    ofstream fp;
    fp.open(path, ios::trunc);
    if (!fp.is_open()) {
        printf("error in opening file\n");
        exit(1);
    }
    for (size_t i = 0; i < points.size(); ++i) {
        if (point_cnts[i] != 0) {
            fp << i << "," << points[i][2] << "," << -points[i][0] << "," << -points[i][1] << endl;
        }
    }
    fp.close();
}

void Slam::dumpPosesToCSV(string path) {
    ofstream fp;
    fp.open(path, ios::trunc);
    if (!fp.is_open()) {
        printf("error in opening file\n");
        exit(1);
    }
    for (size_t i = 0; i < cameras.size(); ++i) {
        Affine3f world_to_camera = Affine3f::Identity();
        world_to_camera.translate(Vector3f(cameras[i][4], cameras[i][5], cameras[i][6]));
        world_to_camera.rotate(Quaternionf(cameras[i][0], cameras[i][1], cameras[i][2], cameras[i][3]));
        Affine3f camera_to_world = world_to_camera.inverse();
        Vector3f loc = extrinsicCamera.inverse() * camera_to_world.translation();
        AngleAxisf angle_a(camera_to_world.rotation());
        angle_a = AngleAxisf(angle_a.angle(), extrinsicCamera.inverse() * angle_a.axis());
        Quaternionf angle(angle_a);
        fp << loc.x() << "," << loc.y() << "," << loc.z() << "," <<  angle.x() << "," << angle.y() << "," << angle.z() << "," << angle.w() << endl; 
    }
}

void Slam::imgToWorld_(double* camera, const int& x, const int& y, const int& z,
                 float* X_ptr, float* Y_ptr, float* Z_ptr) {
    float &X = *X_ptr;
    float &Y = *Y_ptr;
    float &Z = *Z_ptr;

    const float factor = 5000.0;

    // X, Y, Z are in image coordinate, odom frame
    Z = (float) z / factor;
    X = (x - cx) * Z / fx;
    Y = (y - cy) * Z / fy;

    // camera is in image coordinate
    Affine3f world_to_camera = Affine3f::Identity();
    world_to_camera.translate(Vector3f(camera[4], camera[5], camera[6]));
    world_to_camera.rotate(Quaternionf(camera[0], camera[1], camera[2], camera[3]));
    Vector3f world = world_to_camera.inverse() * Vector3f(X, Y, Z);
    
    X = world.x();
    Y = world.y();
    Z = world.z();
}

bool Slam::worldToImg_(double* camera, const float& X, const float& Y, const float& Z,
                     float* x_ptr, float* y_ptr) {
    float &x = *x_ptr;
    float &y = *y_ptr;
    
    // X, Y, Z are in world coordinate
    // camera is in image coordinate
    Affine3f world_to_camera = Affine3f::Identity();
    world_to_camera.translate(Vector3f(camera[4], camera[5], camera[6]));
    world_to_camera.rotate(Quaternionf(camera[0], camera[1], camera[2], camera[3]));
    Vector3f point = world_to_camera * Vector3f(X, Y, Z);
    float xp = point.x() / point.z();
    float yp = point.y() / point.z();
    x = xp * fx + cx;
    y = yp * fy + cy;

    return (x < (float)(imgWidth + imgEdge)) && (x >= (float)(-imgEdge)) && (y <= (float)(imgHeight + imgEdge)) && (y > (float)(-imgEdge));
}



}
