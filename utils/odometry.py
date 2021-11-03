import numpy as np
from numpy.random import normal

fd_write = open("../data/odometry.txt", "w")
with open("../data/groundtruth.txt", "r") as fd_read:
    for line in fd_read:
        if line[0] == "#":
            continue
        line = line.split()
        fd_write.write(line[0])
        odometry = np.around([float(i) for i in line[1:]] + normal(0, 0.06, 7), 4)
        for odom in odometry:
            fd_write.write(" " + str(odom))
        fd_write.write("\n")
    fd_read.close()