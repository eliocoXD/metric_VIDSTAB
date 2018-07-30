import cv2
from cv2 import xfeatures2d
import time
import os
import sys
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

sys.path.append('functs/')
from stabFuncts import *
from frameTransformation import getTrans, getMotion
from videoReconstruction import reconVideo
from analisys import *

start_time = time.time()

# video path
videoInPath = "C:/Users/Admin/PycharmProjects/Video_stabilizer/v2.mp4"
if len(sys.argv) > 1:
    try:
        videoInPath = sys.argv[1]
    except:
        print('Error at loading video')

        sys.exit()

videoInName, videoExt = os.path.splitext(videoInPath)
videoBaseName = os.path.basename(videoInName)

# detector and matcher
#detector = xfeatures2d.SIFT_create()
#detector = xfeatures2d.SURF_create()
detector = cv2.ORB_create()


#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher()

# parameters
MATCH_THRES = float(1000)
RANSAC_THRES = 0.2
BORDER_CUT = 10
# FILT = "square"
FILT = "gauss"
FILT_WIDTH = 7
FILT_SIGMA = 0.2
FAST = True
if FILT == "square":
    filt = (1.0 / FILT_WIDTH) * np.ones(FILT_WIDTH)
    suffix = "_MT_" + str(MATCH_THRES) + "_RT_" + str(RANSAC_THRES) + "_FILT_" + FILT + "_FW_" + str(
        FILT_WIDTH) + "_FAST_" + str(FAST)
elif FILT == "gauss":
    filtx = np.linspace(-3 * FILT_SIGMA, 3 * FILT_SIGMA, FILT_WIDTH)
    filt = np.exp(-np.square(filtx) / (2 * FILT_SIGMA))
    filt = 1 / (np.sum(filt)) * filt
    suffix = "_MT_" + str(MATCH_THRES) + "_RT_" + str(RANSAC_THRES) + "_FILT_" + FILT + "_FW_" + str(
        FILT_WIDTH) + "_SG_" + str(FILT_SIGMA) + "_FAST_" + str(FAST)
videoOutPath = videoInName + "_res" + suffix + videoExt

# get video array
videoArr = getVideoArray(videoInPath)

metricas(videoArr, detector, bf, MATCH_THRES, RANSAC_THRES, filt)
#trans = getTrans(videoArr, detector, bf, MATCH_THRES, RANSAC_THRES, filt, FAST)
#reconVideo (videoInPath, videoOutPath, trans, BORDER_CUT)