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
    }

void Slam::init() {
    poses.clear();
    landmarks.clear();
    clms.clear();
    features.clear();
}

void Slam::observeImage(const Mat& img, const Mat& depth) {
    if (!has_new_pose_) { return; }
    has_new_pose_ = false;
    depths.push_back(depth);

    // extract current image features
    vector<KeyPoint> kp_curr;
    Mat des_curr;
    feature_tracker.getKpAndDes(img, &kp_curr, &des_curr);
    features.emplace_back(kp_curr, des_curr);

    // if this is the first pose, nothing needs to be done
    if (features.size() < 2) { return; }

    // if we have multiple poses
    vector<KeyPoint> kp;
    Mat des;
    vector<DMatch> matches;

    float X_curr, Y_curr, Z_curr, X, Y, Z;
    int idx, x, y;
    float predicted_x, predicted_y;
    size_t T = features.size() - 1;
    size_t landmarkIdx;
    
    // perform correspondance matching across sequence of images
    for (size_t t = 0; t < T; ++t) {
        // retrieve prevously stored features of image t
        kp = features[t].first;
        des = features[t].second;
        // correspondance matching
        feature_tracker.match(des_curr, des, &matches);
        // for all correspondances btw img T (curr) and img t
        for (auto match : matches) {
            // CLM stores <poseIdx, landmarkIdx> : measurement
            CLM clm, clm_curr;
            clm.setPoseIdx(t);
            clm.setMeasurement(kp[match.trainIdx]);
            clm_curr.setPoseIdx(T);
            clm_curr.setMeasurement(kp[match.queryIdx]);
            // check if we've seen this landmark before
            idx = clmsFind_(clm); // TODO: don't use hash for debugging; need to change back to hash
            if (idx != -1) { 
                // if so, landmarks[clms[idx].landmarkIdx] stores init X,Y,Z world coordinate estimate
                // use wolrdToImg_ on X,Y,Z to get predicted measurement of idx-th landmark at pose T
                // wolrdToImg_ also checks if image is supposed to be in bound
                landmarkIdx = clms[idx].landmarkIdx;
                if (worldToImg_(cameras[T], 
                                points[landmarkIdx][0], points[landmarkIdx][1], points[landmarkIdx][2], 
                                &predicted_x, &predicted_y)) { // TODO: FIXME
                    // add predicted measurement of idx-th landmark at pose T to clms for optimization
                    clm_curr.setLandmarkIdx(clms[idx].landmarkIdx);
                    clms.push_back(clm_curr);
                    // update init estimate of landmarks
                    imgToWorld_(cameras[T], x, y, depth, &X_curr, &Y_curr, &Z_curr);
                    ++point_cnts[landmarkIdx];
                    float denominator = (float) point_cnts[landmarkIdx];
                    points[landmarkIdx][0] = (1.0 / denominator) * X_curr + (denominator - 1) / denominator * points[landmarkIdx][0];
                    points[landmarkIdx][1] = (1.0 / denominator) * X_curr + (denominator - 1) / denominator * points[landmarkIdx][1];
                    points[landmarkIdx][2] = (1.0 / denominator) * X_curr + (denominator - 1) / denominator * points[landmarkIdx][2];
                } else {
                    // printf("measure_pred: %.2f|%.2f\n", predicted_x, predicted_y);
                    // this shouldn't happen often
                    if (0) {
                        printf("---------why you're out of rclange?? (case1)------------\n");
                        printf("camera:       %.2f|%.2f|%.2f|%.2f|%.2f|%.2f|%.2f\n", cameras[t][0], cameras[t][1], cameras[t][2], cameras[t][3], cameras[t][4], cameras[t][5], cameras[t][6]);
                        printf("camera_cur:   %.2f|%.2f|%.2f|%.2f|%.2f|%.2f|%.2f\n", cameras[T][0], cameras[T][1], cameras[T][2], cameras[T][3], cameras[T][4], cameras[T][5], cameras[T][6]);
                        printf("landmark:     %.2f|%.2f|%.2f\n", points[clms[idx].landmarkIdx][0], points[clms[idx].landmarkIdx][1], points[clms[idx].landmarkIdx][2]);
                        printf("measure:      %.2f|%.2f\n", kp[match.trainIdx].pt.x, kp[match.trainIdx].pt.y);
                        printf("measure_cur:  %.2f|%.2f\n", kp[match.queryIdx].pt.x, kp[match.queryIdx].pt.y);
                        printf("measure_pred: %.2f|%.2f\n", predicted_x, predicted_y);
                    }
                }
            } else {
                // if we've never seen this landmark before, we want to add both to clms for optimization
                x = kp[match.queryIdx].pt.x;
                y = kp[match.queryIdx].pt.y;
                // get init X,Y,Z world coordinate estimate for new landmark
                imgToWorld_(cameras[T], x, y, depth, &X_curr, &Y_curr, &Z_curr);
                clm_curr.setLandmarkIdx(points.size());
                clms.push_back(clm_curr);
                x = kp[match.trainIdx].pt.x;
                y = kp[match.trainIdx].pt.y;
                imgToWorld_(cameras[t], x, y, depths[t], &X, &Y, &Z);
                X = (X + X_curr) / 2;
                Y = (Y + Y_curr) / 2;
                Z = (Z + Z_curr) / 2;
                if (worldToImg_(cameras[t], X, Y, Z, &predicted_x, &predicted_y)) { // TODO: FIXME
                    // if this landmark is in bound of the current camera
                    clm.setLandmarkIdx(points.size());
                    clms.push_back(clm);
                } else {
                    // TODO: this shouldn't happen often
                    if (0) {
                        printf("---------why you're out of range?? (case2)------------\n");
                        printf("camera:         %.4f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f\n", cameras[t][0], cameras[t][1], cameras[t][2], cameras[t][3], cameras[t][4], cameras[t][5], cameras[t][6]);
                        printf("camera_cur:     %.4f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f\n", cameras[T][0], cameras[T][1], cameras[T][2], cameras[T][3], cameras[T][4], cameras[T][5], cameras[T][6]);
                        printf("landmark:       %.2f|%.2f|%.2f\n", X, Y, Z);
                        printf("landmark_cur:   %.2f|%.2f|%.2f\n", X_curr, Y_curr, Z_curr);
                        printf("measure:        %.2f|%.2f\n", kp[match.trainIdx].pt.x, kp[match.trainIdx].pt.y);
                        printf("measure_cur:    %.2f|%.2f\n", kp[match.queryIdx].pt.x, kp[match.queryIdx].pt.y);
                        printf("measure_pred:   %.2f|%.2f\n", predicted_x, predicted_y);
                    }
                }
                double* landmark = new double[]{X, Y, Z};
                points.push_back(landmark);
                point_cnts.push_back(2);
            }
            // TODO: add me back to clean up codes - clms.push_back(clm_curr);
        }
    }
}

void Slam::observeOdometry(const Vector3f& odom_loc ,const Quaternionf& odom_angle) {
    // if ( poses.size() != 0 || getDist_(prev_odom_loc_, odom_loc) > MIN_DELTA_D || getQuaternionDelta_(prev_odom_angle_, odom_angle).norm() < MIN_DELTA_A ) { 
        prev_odom_loc_ = odom_loc;
        prev_odom_angle_ = odom_angle;
        // pose: transform baselink back to world coordinate
        double* pose = new double[] {odom_angle.w(), odom_angle.x(), odom_angle.y(), odom_angle.z(),
                                    odom_loc.x(), odom_loc.y(), odom_loc.z()};
        poses.push_back(pose);
        Affine3f odom_to_world = Affine3f::Identity();
        odom_to_world.translate(odom_loc);
        odom_to_world.rotate(odom_angle);
        Affine3f world_to_odom = odom_to_world.inverse();
        Quaternionf angle(world_to_odom.rotation());
        Vector3f loc(world_to_odom.translation());
        double* camera = new double[] {angle.w(), angle.x(), angle.y(), angle.z(),
                                       loc.x(), loc.y(), loc.z()};
        cameras.push_back(camera);

        has_new_pose_ = true; 
    // }
}

bool Slam::optimize(bool minimizer_progress_to_stdout, bool briefReport, bool fullReport) {
    Problem problem;
    for (size_t i = 0; i < clms.size(); ++i) {
        if (debug) {
            printf("--------------cmls--------------\n");
            printf("camera:          %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
                cameras[clms[i].poseIdx][0], cameras[clms[i].poseIdx][1], cameras[clms[i].poseIdx][2], cameras[clms[i].poseIdx][3],
                cameras[clms[i].poseIdx][4], cameras[clms[i].poseIdx][5], cameras[clms[i].poseIdx][6]);
            printf("landmark:        %.2f | %.2f | %.2f \n", points[clms[i].landmarkIdx][0], points[clms[i].landmarkIdx][1], points[clms[i].landmarkIdx][2]);
            printf("measurement:     %.2f | %.2f \n", clms[i].measurement[0], clms[i].measurement[1]);
            float predicted_x, predicted_y;
            worldToImg_(cameras[clms[i].poseIdx], points[clms[i].landmarkIdx][0], points[clms[i].landmarkIdx][1], points[clms[i].landmarkIdx][2], &predicted_x, &predicted_y);
            printf("measurment_pred: %.2f | %.2f \n", predicted_x, predicted_y);
        }
        CostFunction* cost_function = ReprojectionError::Create(clms[i].measurement[0], clms[i].measurement[1]);
        problem.AddResidualBlock(cost_function, NULL, cameras[clms[i].poseIdx], points[clms[i].landmarkIdx]);
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = minimizer_progress_to_stdout;
    Solver::Summary summary;
    double cost;
    // problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    ceres::Solve(options, &problem, &summary);
    if (briefReport) std::cout << summary.FullReport() << "\n";
    if (fullReport)  std::cout << summary.BriefReport() << "\n";
    return (summary.termination_type == ceres::CONVERGENCE 
         || summary.termination_type == ceres::USER_SUCCESS);
}

void Slam::displayCLMS() {
    printf("-------display inputs---------\n");
    for (size_t i = 0; i < clms.size(); ++i) {
        printf("%ld:         %ld | %ld\n", i, clms[i].poseIdx, clms[i].landmarkIdx);
        printf("camera:      %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
            poses[clms[i].poseIdx][0], poses[clms[i].poseIdx][1], poses[clms[i].poseIdx][2], poses[clms[i].poseIdx][3],
            poses[clms[i].poseIdx][4], poses[clms[i].poseIdx][5], poses[clms[i].poseIdx][6]);
        printf("landmark:    %.2f | %.2f | %.2f \n", points[clms[i].landmarkIdx][0], points[clms[i].landmarkIdx][1], points[clms[i].landmarkIdx][2]);
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

void Slam::displayPosesAndLandmarkcs() {
    printf("----------cameras----------\n");
    for (size_t i = 0; i < cameras.size(); ++i) {
        printf("%ld: %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
            i, cameras[i][0], cameras[i][1], cameras[i][2], cameras[i][3], cameras[i][4], cameras[i][5], cameras[i][6]);
    }
    printf("----------poses----------\n");
    for (size_t i = 0; i < cameras.size(); ++i) {
        Affine3f odom_to_world, world_to_odom;
        world_to_odom = Affine3f::Identity();
        world_to_odom.translate(Vector3f(cameras[i][4], cameras[i][5], cameras[i][6]));
        world_to_odom.rotate(Quaternionf(cameras[i][0], cameras[i][1], cameras[i][2], cameras[i][3]));
        odom_to_world = world_to_odom.inverse();
        Quaternionf angle(world_to_odom.rotation());
        Vector3f loc(world_to_odom.translation());
        double* pose = new double[] {angle.w(), angle.x(), angle.y(), angle.z(),
                                        loc.x(), loc.y(), loc.z()};
        printf("%ld: %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f\n", 
            i, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]);
    }
    printf("----------landmarks----------\n");
    for (size_t i = 0; i < points.size(); ++i) {
        printf("%ld: %.2f | %.2f | %.2f \n", i, points[i][0], points[i][1], points[i][2]);
    }
}

// void Slam::dumpToCSV(string path) {

// }


int Slam::clmsFind_(const CLM& clm) {
    for (int i = 0; i < clms.size(); ++i) {
        if (clm == clms[i]) {return i;}
    }
    return -1;
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

void Slam::imgToWorld_(double* camera, const int& x, const int& y, const Mat& depth,
                 float* X_ptr, float* Y_ptr, float* Z_ptr) {
    float &X = *X_ptr;
    float &Y = *Y_ptr;
    float &Z = *Z_ptr;

    const float factor = 5000.0;
    // cout << "imgToWorld_depth " << x << ", " << y << endl;

    Z = depth.at<ushort>(y, x) / factor;
    X = (x - cx) * Z / fx;
    Y = (y - cy) * Z / fy;

    Affine3f world_to_odom = Affine3f::Identity();
    world_to_odom.translate(Vector3f(camera[4], camera[5], camera[6]));
    world_to_odom.rotate(Quaternionf(camera[0], camera[1], camera[2], camera[3]));
    // Vector3f world = odom_to_world.inverse() * Vector3f(X, Y, Z);
    Vector3f world = world_to_odom.inverse() * Vector3f(X, Y, Z);

    // Quaternionf r(camera[0], camera[1], camera[2], camera[3]);
    // Vector3f v(camera[4], camera[5], camera[6]);
    // Vector3f world = r.inverse() * (Vector3f(X, Y, Z) - v);
    X = world.x();
    Y = world.y();
    Z = world.z();
}

void Slam::imgToWorld_(double* camera, const int& x, const int& y, const int& z,
                 float* X_ptr, float* Y_ptr, float* Z_ptr) {
    float &X = *X_ptr;
    float &Y = *Y_ptr;
    float &Z = *Z_ptr;

    const float factor = 5000.0;
    // cout << "imgToWorld_z " << x << ", " << y << endl;

    Z = z / factor;
    X = (x - cx) * Z / fx;
    Y = (y - cy) * Z / fy;

    // Quaternionf r(camera[0], camera[1], camera[2], camera[3]);
    // Vector3f v(camera[4], camera[5], camera[6]);
    // Vector3f world = r * Vector3f(X, Y, Z) + v;
    // Vector3f world = r.inverse() * (Vector3f(X, Y, Z) - v);

    Affine3f world_to_odom = Affine3f::Identity();
    world_to_odom.translate(Vector3f(camera[4], camera[5], camera[6]));
    world_to_odom.rotate(Quaternionf(camera[0], camera[1], camera[2], camera[3]));
    // Vector3f world = odom_to_world.inverse() * Vector3f(X, Y, Z);
    Vector3f world = world_to_odom.inverse() * Vector3f(X, Y, Z);

    X = world.x();
    Y = world.y();
    Z = world.z();
}

bool Slam::worldToImg_(double* camera, const float& X, const float& Y, const float& Z,
                     float* x_ptr, float* y_ptr) {
    float &x = *x_ptr;
    float &y = *y_ptr;

    // Quaternionf r(camera[0], camera[1], camera[2], camera[3]);
    // Vector3f v(camera[4], camera[5], camera[6]);
    // Vector3f point = r * Vector3f(X, Y, Z) + v;
    // Vector3f point = r.inverse() * (Vector3f(X, Y, Z) - v);

    Affine3f world_to_odom = Affine3f::Identity();
    world_to_odom.translate(Vector3f(camera[4], camera[5], camera[6]));
    world_to_odom.rotate(Quaternionf(camera[0], camera[1], camera[2], camera[3]));
    Vector3f point = world_to_odom * Vector3f(X, Y, Z);
    // Vector3f point = odom_to_world.inverse() * Vector3f(X, Y, Z);

    float xp = point.x() / point.z();
    float yp = point.y() / point.z();
    x = xp * fx + cx;
    y = yp * fy + cy;

    return (x < (float)(imgWidth + imgEdge)) && (x >= (float)(-imgEdge)) && (y <= (float)(imgHeight + imgEdge)) && (y > (float)(-imgEdge));
}



}
