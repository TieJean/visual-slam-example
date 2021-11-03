import numpy as np
import cv2
from tqdm import tqdm
import pickle

DATA_DIR = '../data/'

stage = int(input())

if stage <= 0:
    fp_write = open(DATA_DIR + 'time.txt', 'w')
    with open(DATA_DIR + 'rgb.txt', 'r') as fp_read:
        print('---------start extracting timestamps----------')
        for line in tqdm(fp_read):
            if line[0] == '#':
                continue
            (timestamp, path) = line.split(" ", 1)
            fp_write.write(timestamp + '\n')
        print('---------ends----------')
    fp.close()


if stage <= 1:
    with open(DATA_DIR + 'rgb.txt', 'r') as fp:
        print('---------start converting rgb to grayscale----------')
        for line in tqdm(fp):
            if line[0] == '#':
                continue
            (timestamp, path) = line.split(" ", 1)
            img_rgb  = cv2.imread(DATA_DIR + path[:-1])
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) 
            cv2.imwrite(DATA_DIR + 'gray/' + timestamp + '.png', img_gray)
        print('---------ends----------')
    fp.close()

if stage <= 2:
    with open(DATA_DIR + 'depth.txt', 'r') as fp:
        print('---------start converting depth to grayscale----------')
        for line in tqdm(fp):
            if line[0] == '#':
                continue
            (timestamp, path) = line.split(" ", 1)
            img_depth  = cv2.imread(DATA_DIR + path[:-1])
            img_depth_gray = cv2.cvtColor(img_depth, cv2.COLOR_RGB2GRAY) 
            cv2.imwrite(DATA_DIR + 'depth_gray/' + timestamp + '.png', img_depth_gray)
        print('---------ends----------')
    fp.close()
            
# if stage <= 2:
#     with open(DATA_DIR + 'time.txt', 'r') as fp:
#         print('---------start detection and computation----------')
#         for line in tqdm(fp):
#             if line[0] == '#':
#                 continue
#             timestamp = line[:-1]
#             img = cv2.imread(DATA_DIR + 'gray/' + timestamp + '.png')
#             orb = cv2.ORB_create() 
#             kp, des = orb.detectAndCompute(img, None)
#             pickle.dump(kp,  DATA_DIR + 'keypoints/'   + timestamp)
#             pickle.dump(des, DATA_DIR + 'descriptors/' + timestamp)
#         print('---------ends----------')


