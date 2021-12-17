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
    }

void Slam::init() {
    
}

void Slam::observeImage(const Mat& img, const Mat& depth) {
    float prev_pred[3];
    float curr_pred[3];

    vector<KeyPoint> kp_curr, kp_prev;
    Mat des_curr, des_prev;
    Mat depth_prev;
    vector<DMatch> matches;
    CLM clm_curr, clm_prev;
    int idx;
    size_t landmarkIdx;
    float predicted_x, predicted_y;

    if (!has_new_pose_) { return; }
    has_new_pose_ = false;
    feature_tracker.getKpAndDes(img, &kp_curr, &des_curr);
    observations.emplace_back(kp_curr, des_curr, depth);
    if (observations.size() < 2) { return; }

    const size_t T = cameras.size() - 1; // last pose idx

    for (size_t t = 0; t < cameras.size() - 1; ++t) {
        kp_prev    = observations[t].kps;
        des_prev   = observations[t].descriptors;
        depth_prev = observations[t].depth;
        // prev - train, curr - query
        feature_tracker.match(des_curr, des_prev, &matches);

        for (auto match : matches) {
            clm_prev.setPoseIdx(t);
            clm_prev.setMeasurement(kp_prev[match.trainIdx]);
            clm_curr.setPoseIdx(T);
            clm_curr.setMeasurement(kp_curr[match.queryIdx]);
            idx = clmsFind_(clm_prev); // TODO: change me back to hashing after debugging
            imgToWorld_(cameras[t], kp_prev[match.trainIdx].pt.x, 
                                    kp_prev[match.trainIdx].pt.y, 
                                    observations[t].depth, 
                                    &prev_pred[0], &prev_pred[1], &prev_pred[2]);
            imgToWorld_(cameras[T], kp_curr[match.queryIdx].pt.x, 
                                    kp_curr[match.queryIdx].pt.y, 
                                    observations[T].depth, 
                                    &curr_pred[0], &curr_pred[1], &curr_pred[2]);
            if (idx != -1) { 
                landmarkIdx = clms[idx].landmarkIdx;
                if (!worldToImg_(cameras[T], 
                                points[landmarkIdx].first[0], points[landmarkIdx].first[1], points[landmarkIdx].first[2],
                                &predicted_x, &predicted_y)) { continue; } // if landmark projection onto camera T is invalid
                clm_curr.setLandmarkIdx(landmarkIdx);
                clms.push_back(clm_curr);
                for (size_t i = 0; i < landmarkDim; ++i) {
                    points[landmarkIdx].first[i] = (points[landmarkIdx].first[i] * points[landmarkIdx].second + curr_pred[i]) / (++points[landmarkIdx].second);
                }
            } else {
                // if prev predicted landmark projection onto the current camera is invalid OR
                // if curr predicted landmark porjection onto the previous camera is invalid   --> outliers
                if (!worldToImg_(cameras[T], prev_pred[0], prev_pred[1], prev_pred[2], &predicted_x, &predicted_y) ||
                    !worldToImg_(cameras[t], curr_pred[0], curr_pred[1], curr_pred[2], &predicted_x, &predicted_y) ) { continue; }
                clm_curr.setLandmarkIdx(points.size());
                clm_prev.setLandmarkIdx(points.size());
                clms.push_back(clm_curr);
                clms.push_back(clm_prev);
                double* landmark = new double[3];
                for (size_t i = 0; i < landmarkDim; ++i) {
                    landmark[i] = (prev_pred[i] + curr_pred[i]) / 2; // TODO: this may not be valid
                }
                points.emplace_back(landmark, 2);
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

        // assume "camera" is already transformation in camera coordinate
        Affine3f odom_to_world = Affine3f::Identity();
        odom_to_world.translate(Vector3f(odom_loc.x(), odom_loc.y(), odom_loc.z()));
        odom_to_world.rotate(Quaternionf(odom_angle.w(), odom_angle.x(), odom_angle.y(), odom_angle.z()));
        Affine3f world_to_odom = odom_to_world.inverse();
        Quaternionf angle(world_to_odom.rotation());
        Vector3f loc(world_to_odom.translation());
        double* camera = new double[] {angle.w(), angle.x(), angle.y(), angle.z(),
                                       loc.x(), loc.y(), loc.z()}; // TODO: free me
        cameras.push_back(camera);

        // printf("------observeOdometry------------\n");
        // printf("pose: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", odom_angle.w(), odom_angle.x(), odom_angle.y(), odom_angle.z(), odom_loc.x(), odom_loc.y(), odom_loc.z());
        // printf("camera: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n\n", camera[0], camera[1], camera[2], camera[3], camera[4], camera[5], camera[6]);

        has_new_pose_ = true; 
    // }
}

bool Slam::optimize() {
    cout << "optimize" << endl;
    Problem problem;
    if (observations.size() < 1) {
        printf("unexpected error: observations size shouldn't be 0\n");
        exit(1);
    }

    // debug start
    // for (size_t i = 0; i < clms.size(); ++i) {
    //     printf("measurement: %.2f, %.2f\n", clms[i].measurement[0], clms[i].measurement[1]);
    //     printf("camera: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", cameras[clms[i].poseIdx][0], cameras[clms[i].poseIdx][1], cameras[clms[i].poseIdx][2], cameras[clms[i].poseIdx][3], cameras[clms[i].poseIdx][4], cameras[clms[i].poseIdx][5], cameras[clms[i].poseIdx][6]);
    //     printf("points: %ld - %.2f,%.2f,%.2f\n", clms[i].landmarkIdx, points[clms[i].landmarkIdx][0], points[clms[i].landmarkIdx][1], points[clms[i].landmarkIdx][2]);
    //     cout << endl;
    // }
    // vector<vector<pair<double, double>>> debug;
    // debug.resize(6);
    // for (size_t i = 0; i < 6; ++i) { debug[i].resize(99); }
    // for (size_t i = 0; i < 6; ++i) {
    //     for (size_t j = 0; j < 99; ++j) {
    //         debug[i][j] = pair<double, double>(0,0);
    //     }
    // }
    // for (size_t i = 0; i < clms.size(); ++i) {
    //     debug[clms[i].poseIdx][clms[i].landmarkIdx] = pair<double, double>(clms[i].measurement[0], clms[i].measurement[1]);
    // }
    // ofstream fp;
    // for (size_t i = 0; i < 6; ++i) {
    //     fp.open("../data/results/" + to_string(i+1) + ".csv", ios::trunc);
    //     if (!fp.is_open()) {
    //         printf("error in opening file\n");
    //         exit(1);
    //     }
    //     for (size_t j = 0; j < 99; ++j) {
    //         if (debug[i][j].first == 0 && debug[i][j].second == 0) { continue; }
    //         fp << j << "," << debug[i][j].first << "," << debug[i][j].second << endl;
    //     }
    //     fp.close();
    // }
    cout << "start optimize" << endl;
    // end debug

    for (size_t i = 0; i < clms.size(); ++i) {
        CostFunction* cost_function = ReprojectionError::Create(clms[i].measurement[0], clms[i].measurement[1]);
        problem.AddResidualBlock(cost_function, new HuberLoss(huberLossScale), cameras[clms[i].poseIdx], points[clms[i].landmarkIdx].first);
    }
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

void Slam::displayCLMS() {
    printf("-------display inputs---------\n");
    for (size_t i = 0; i < clms.size(); ++i) {
        printf("%ld:         %ld | %ld\n", i, clms[i].poseIdx, clms[i].landmarkIdx);
        printf("camera:      %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
            cameras[clms[i].poseIdx][0], cameras[clms[i].poseIdx][1], cameras[clms[i].poseIdx][2], cameras[clms[i].poseIdx][3],
            cameras[clms[i].poseIdx][4], cameras[clms[i].poseIdx][5], cameras[clms[i].poseIdx][6]);
        printf("landmark:    %.2f | %.2f | %.2f \n", points[clms[i].landmarkIdx].first[0], points[clms[i].landmarkIdx].first[1], points[clms[i].landmarkIdx].first[2]);
        printf("measurement: %.2f | %.2f \n\n", clms[i].measurement[0], clms[i].measurement[1]);
    }
}

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
    printf("----------poses----------\n");
    Affine3f extrinsicCamera = Affine3f::Identity();
    extrinsicCamera.translate(Vector3f(0,0,0));
    extrinsicCamera.rotate(Quaternionf(0.5, 0.5, -0.5, 0.5));
    for (size_t i = 0; i < cameras.size(); ++i) {
        Affine3f world_to_odom = Affine3f::Identity();
        world_to_odom.translate(Vector3f(cameras[i][4], cameras[i][5], cameras[i][6]));
        world_to_odom.rotate(Quaternionf(cameras[i][0], cameras[i][1], cameras[i][2], cameras[i][3]));
        Affine3f odom_to_world = extrinsicCamera.inverse() * world_to_odom.inverse();
        Vector3f loc(odom_to_world.translation());
        Quaternionf angle(odom_to_world.rotation());
        // printf("camera %ld: %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
        //     i, cameras[i][0], cameras[i][1], cameras[i][2], cameras[i][3], cameras[i][4], cameras[i][5], cameras[i][6]);
        printf("pose %ld: %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
            i, loc.x(), loc.y(), loc.z(), angle.x(), angle.y(), angle.z(), angle.w());
    }
}

void Slam::displayLandmarks() {
    printf("----------landmarks----------\n");
    for (size_t i = 0; i < points.size(); ++i) {
        printf("%ld: %.2f | %.2f | %.2f \n", i, points[i].first[2], -points[i].first[0], -points[i].first[1]);
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
        fp << i << "," << points[i].first[2] << "," << -points[i].first[0] << "," << -points[i].first[1] << endl;
    }
    fp.close();
}

void Slam::dumpPosesToCSV(string path) {

}

void Slam::imgToWorld_(double* camera, const int x, const int y, const Mat& depth,
                       float* X_ptr, float* Y_ptr, float* Z_ptr) {
    float z = (float)depth.at<ushort>(y, x);
    imgToWorld_(camera, x, y, z, X_ptr, Y_ptr, Z_ptr);
}

void Slam::imgToWorld_(double* camera, const int x, const int y, const int z,
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
    if (0) {
        cout << endl;
        cout << Z << endl;
        cout << x << ", " << y << endl;
        printf("camera: %.4f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f\n", camera[0], camera[1], camera[2], camera[3], camera[4], camera[5], camera[6]);
        cout << world_to_camera.rotation() << endl;
        cout << world_to_camera.translation() << endl;
        cout << endl;
    }
    
    X = world.x();
    Y = world.y();
    Z = world.z();
}

bool Slam::worldToImg_(double* camera, const float X, const float Y, const float Z,
                       float* x_ptr, float* y_ptr) {
    float &x = *x_ptr;
    float &y = *y_ptr;
    
    // X, Y, Z are in world coordinate
    // camera is in image coordinate
    Affine3f world_to_camera = Affine3f::Identity();
    world_to_camera.translate(Vector3f(camera[4], camera[5], camera[6]));
    world_to_camera.rotate(Quaternionf(camera[0], camera[1], camera[2], camera[3]));
    Vector3f point = world_to_camera * Vector3f(X, Y, Z);

    // TODO: danger - fix dividing by small number
    float xp = point.x() / point.z();
    float yp = point.y() / point.z();
    x = xp * fx + cx;
    y = yp * fy + cy;
    // Vector3f image = cameraIntrinsic * Vector3f(xp, yp, 1);
    // x = image.x();
    // y = image.y();

    if (0) {
        cout << endl;
        printf("camera: %.4f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f\n", camera[0], camera[1], camera[2], camera[3], camera[4], camera[5], camera[6]);
        cout << world_to_camera.rotation() << endl;
        cout << world_to_camera.translation() << endl;
        cout << endl;
    }

    return (x < (float)(imgWidth + imgEdge)) && (x >= (float)(-imgEdge)) && (y <= (float)(imgHeight + imgEdge)) && (y > (float)(-imgEdge));
}



}
