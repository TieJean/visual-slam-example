
CC = g++
CFLAGS = -std=c++2a -w -g -fsanitize=address
OPENCV_DIR = `pkg-config --cflags --libs opencv4`
INCLUDE_DIR = -I/usr/local/include/eigen3 -I./
GTSAM_DIR = -L ~/Documents/Research/amrl/gtsam-lib/gtsam/build/gtsam -lgtsam 

test: test.cpp
	$(CC) $(OPENCV_DIR) $(CFLAGS)  test.cpp -o test.out

feature: feature_tracker.cpp 
	$(CC) $(OPENCV_DIR) $(CFLAGS)  feature_tracker.cpp -o feature_tracker.out

ba: ba_solver.cpp 
	$(CC) $(INCLUDE_DIR) $(OPENCV_DIR) -std=c++14 ba_solver.cpp -o ba_solver.out

feature_tracker.o: feature_tracker.cpp
	$(CC) $(OPENCV_DIR) $(CFLAGS) -c feature_tracker.cpp -o feature_tracker.o

slam.o: slam.cpp feature_tracker.o
	$(CC) $(INCLUDE_DIR) $(OPENCV_DIR) $(CFLAGS) -c slam.cpp feature_tracker.o -o slam.o


slam: slam_main.cpp slam.o feature_tracker.o 
	$(CC) $(OPENCV_DIR) $(GTSAM_DIR) $(INCLUDE_DIR) $(CFLAGS) slam_main.cpp slam.o feature_tracker.o  -o slam.out

# slam: slam_main.cpp slam.c feature_tracker.cpp 
# 	$(CC) $(OPENCV_DIR) $(GTSAM_DIR) $(INCLUDE_DIR) $(CFLAGS) slam_main.cpp slam.o feature_tracker.o  -o slam.out

clean:
	rm -f *.o
	rm -f *.out