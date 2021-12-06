#include <vector>

#include "feature_tracker.h"
#include "slam.h"

using namespace std;
using namespace slam;
using namespace Eigen;


int main() {
    if (1) {
        // pose: camera_to_world
        // camera: world_to_camera
        Slam slam;
        slam.init();
        // odom x --> camera z
        // odom y --> camera x
        // odom z --> camera y
        // double pose1[] = {-20.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000};
        // double pose2[] = {-18.500000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000};
        double pose1[] = {1, 0, 0, 0, 0.000000, 0.000000, -20.000000};
        double pose2[] = {1, 0, 0, 0, 0.000000, 0.000000, -18.500000};
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
        
        float X, Y, Z;
        float x, y, z;
        float x_pred, y_pred;
        x = 528.511092; 
        y = 210.307524;
        z = (18.883421 + 18.5) * 5000;
        slam.imgToWorld_(pose2, x, y, z, &X, &Y, &Z);
        printf("landmark:     %.2f|%.2f|%.2f\n", X, Y, Z);
        // X = 18.883421;
        // Y = -20.269062;
        // Z = 2.886363;
        slam.worldToImg_(camera1, X, Y, Z, &x_pred, &y_pred);
        printf("measure_pred: %.2f|%.2f\n", x_pred, y_pred); // 536.877545 209.116121


    }

    if (0) {
        Slam slam;
        slam.init();

    }

    
    
}