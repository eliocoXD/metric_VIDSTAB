import numpy as np
import cv2
from matplotlib import pyplot as plt
from stabFuncts import *
from frameTransformation import *

global Po,Co, P1
Po=0*[0]
P1=0*[0]
Co=0*[0]
def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0
    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.
    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.
    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.
    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    global Po, Co,P1
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    #print(Co[0])
    po=Po
    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1

        cv2.circle(out, (int(Po[0]+Co[0]), int(Po[1]+Co[1])), 10, (0, 255, 0), 4)
        cv2.circle(out, (int(Co[0]), int(Co[1])), 10, (0, 255, 255), 4)
        cv2.circle(out, (int(P1[0]+Co[0]), int(P1[1]+Co[1])), 10, (0, 0, 255), 4)
        #print(Po,'--',P1)
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    #plt.figure()
    #plt.imshow(out)
    #plt.show()
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out


def filterMatches (matches, MATCH_THRES):
    goodMatches = []
    for m in matches:
        if m.distance < MATCH_THRES:
            goodMatches.append(m)
    return goodMatches

def maskMatches (matches, mask):
    goodMatches = []
    for i in range(len(matches)):
        if mask[i] == 1:
            goodMatches.append(matches[i])
    return goodMatches

def Keypointprocces(frame1, frame2, detector, bf, MATCH_THRES, RANSAC_THRES):
    global Co



    # get keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(frame1, None)
    kp2, des2 = detector.detectAndCompute(frame2, None)

    # get matches
    matches = bf.match(des1, des2)

    matches = filterMatches(matches, MATCH_THRES)
    P1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    P2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    #P1, P2 = keypoints(kp1, kp2, matches)


    Co=[len(frame1[1])/2,len(frame1)/2]
    #dx, dy = distance(P1, P2, Po,Co)

    #M, mask = cv2.findHomography(P1, P2, cv2.RANSAC, RANSAC_THRES)
    M=cv2.estimateRigidTransform(np.array(P1), np.array(P2),  False)
    dy = M[1][2]
    dx = M[0][2]
    # plotkeypts(P1, P2, frame1, frame2)

    #drawMatches(frame1, kp1, frame2, kp2, matches)


    # plotMatches(frame1, kp1, frame2, kp2, matches, 0)

    return dx,dy


def Matrixprocces(frame1, frame2, detector, bf, MATCH_THRES, RANSAC_THRES,Dx,Dy,i,promx,promy):
    # get keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(frame1, None)
    kp2, des2 = detector.detectAndCompute(frame2, None)

    # get matches
    matches = bf.match(des1, des2)
    matches = filterMatches(matches, MATCH_THRES)
    P1, P2 = keypoints(kp1, kp2, matches)

    VID_WIDTH = len(frame1[1])
    VID_HEIGHT = len(frame1)
    videoInSize = (int(VID_WIDTH), int(VID_HEIGHT))

    #              X,Y

    M = np.identity(3)


    M [1][2]= -Dy[i]
    M [0][2]= -Dx[i]
    frameOut = cv2.warpPerspective(frame2,M, videoInSize, flags=cv2.INTER_NEAREST)
    #frameOut=cv2.warpAffine(frame1,M,videoInSize)
    r = 500.0 / frame1.shape[1]
    dim = (500, int(frame1.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)
    frameOut = cv2.resize(frameOut, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('oyut',frameOut)
    cv2.imshow('in',resized)
    cv2.waitKey(1)



    return M

def keypoints (kp1, kp2, matches):
    """

        This function takes in two images with their associated
        keypoints, as well as a list of DMatch data structure (matches)
        that contains which keypoints matched in which images.

        It will return two arrays with all matched keypoints from frame 2 a nd frame 1
        with X,Y format.

        img1,img2 - Grayscale images
        kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
                  detection algorithms
        matches - A list of matches of corresponding keypoints through any
                  OpenCV keypoint matching algorithm
        """

    #create an empty array to be fill with coodenates
    i = 0
    X1 = [0] * len(kp1)
    Y1 = [0] * len(kp1)
    X2 = [0] * len(kp2)
    Y2 = [0] * len(kp2)
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage

    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        X1[i] = x1
        Y1[i] = y1
        X2[i] = x2
        Y2[i] = y2


        i = i + 1
    X1 = np.resize(X1, (1, i))
    Y1 = np.resize(Y1, (1, i))
    X2 = np.resize(X2, (1, i))
    Y2 = np.resize(Y2, (1, i))

    points1 = X1, Y1
    points1 = np.transpose(points1)
    points2 = X2, Y2
    points2 = np.transpose(points2)

    #print(points1,points2)

    return points1,points2

def distance(points1,points2,po,co):

    global Po,P1
    long=len(points1)

    dx = [0] * long
    dy = [0] * long

    #P1=points2

    if len(po)== 0:
        po =0* points1
        for i in range(long):
            po[i] = [points1[i][0][0] - co[0], points1[i][0][1] - co[1]]
        tx=0
        ty=0
        for j in range(long):
            tx = tx + po[j][0][0]
            ty = ty + po[j][0][1]
        Po = [tx / long, ty / long]

        p1 =0* points2
        for i in range(long):
            p1[i] = [points2[i][0][0] - co[0], points2[i][0][1] - co[1]]
        tx = 0
        ty = 0
        for j in range(long):
            tx = tx + p1[j][0][0]
            ty = ty + p1[j][0][1]
        P1 = [tx / long, ty / long]
        #print(Po,P1)

        return P1[0]-Po[0],P1[1]-Po[1]
    else:

        p1 = 0 * points1
        for i in range(long):
            p1[i] = [points1[i][0][0] - co[0], points1[i][0][1] - co[1]]
        tx = 0
        ty = 0
        for j in range(long):
            tx = tx + p1[j][0][0]
            ty = ty + p1[j][0][1]
        P1 = [tx / long, ty / long]
        #print(Po, P1)

        return P1[0] - Po[0], P1[1] - Po[1]





def plotkeypts(points1,points2,img1,img2):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    out = np.zeros((max([rows1, rows2]), cols1, 3), dtype='uint8')


    for i in range (len(points1)):


        # x - columns
        # y - rows
        (x1,y1) = points1[i][0][0],points1[i][0][1]
        (x2,y2) = points2[i][0][0],points2[i][0][1]

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 10, (255, 0, 0), 1)
        cv2.circle(out, (int(x2),int(y2)), 10, (0, 255, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.imshow('out',out)

    cv2.waitKey()

def getLocalMotionMetric(videoArr, filt, detector, bf, MATCH_THRES, RANSAC_THRES):
    N_FRAMES = videoArr.shape[0]
    FILT_WIDTH = filt.size
    halfFilt = FILT_WIDTH // 2
    localMotion = np.zeros((N_FRAMES, FILT_WIDTH, 3, 3))
    print('local fast')
    Dx=[0]*N_FRAMES
    Dy=[0]*N_FRAMES
    M = np.zeros((N_FRAMES, FILT_WIDTH, 3, 3))

    # get data frame motion with ORB  and calculate distances between keypoints
    for i in range(N_FRAMES):
        print("frame " + str(i))
        try:
            Dx[i],Dy[i]= Keypointprocces(videoArr[i, :, :], videoArr[i + 1, :, :], detector, bf, MATCH_THRES, RANSAC_THRES)
            #print(Dy[i])
        except IndexError:
            Dx=Dx
            Dy=Dy


    t=0
    Out=[0]*len(Dy)
    for i in range(len(Dy)):
        t = Dy[i] + t
        Out[i] = t

    t = 0
    Outx = [0] * len(Dx)
    for i in range(len(Dx)):
        t = Dx[i] + t
        Outx[i] = t


    ref1=np.linspace(0,46,24)
    ref2 = np.linspace(45, 0, 24)
    ref3=np.concatenate([ref1,ref2])
    ref=np.concatenate([ref3,ref3])
    ref4=np.linspace(45,0,25)
    ref5=np.concatenate([ref1,ref4])
    ref=np.concatenate([ref,ref5])
    print(len(ref),len(Dy))

    fig = plt.figure()
    ax1 = fig.add_subplot(221)

    plt.title('My')
    ax1.plot(Dy, 'r-')
    plt.grid()

    ax2 = fig.add_subplot(222)
    plt.title('Mx')
    ax2.plot(Dx, 'k-')
    plt.grid()

    ax3 = fig.add_subplot(223)
    plt.title('Dy')
    ax3.plot(Out, 'b-')
    plt.grid()

    ax4 = fig.add_subplot(224)
    plt.title('Dx')
    ax4.plot(Outx, 'y-')
    plt.grid()

    ######Second figure
    fig2 = plt.figure()
    aX = fig2.add_subplot(414)
    aX.set_xlabel("N Frames")
    aY = fig2.add_subplot(412)
    aY.set_ylabel("Pixeles")
    ax5 = fig2.add_subplot(411)
    plt.title('Desplazamiento en Y')
    ax5.plot(ref, 'r-',label='referencia')
    ax5.legend()
    plt.grid()

    ax6 = fig2.add_subplot(412)
    ax6.plot(Out, 'b-',label='Dy')
    ax6.legend()
    plt.grid()

    ax7 = fig2.add_subplot(413)
    ax7.plot(Out, 'b-', label='Dy')
    ax7.plot(ref, 'r-', label='referencia')
    ax7.legend()
    plt.grid()

    ax8 = fig2.add_subplot(414)

    ax8.plot(abs(ref - Out), 'g-',label='error')
    ax8.legend()
    plt.grid()

    plt.show()
    promx=sum(Dx)/len(Dx)
    promy=sum(Dy)/len(Dy)
    M=0
    """""""""
        # get M homography
    for i in range(N_FRAMES):
        M[i, int(halfFilt), :, :] = np.identity(3)
        try:
            M[i, int(halfFilt) + 1, :, :] = Matrixprocces(videoArr[i, :, :], videoArr[i + 1, :, :], detector, bf, MATCH_THRES, RANSAC_THRES,Outx,Out,i,promx,promy)

        except IndexError:
            M[i, int(halfFilt) + 1, :, :] = np.identity(3)
"""""
    return M




def metricas(videoArr, detector, bf, MATCH_THRES, RANSAC_THRES, filt):
    N_FRAMES = videoArr.shape[0]
    trans = np.zeros((N_FRAMES, 3, 3))

    localMotion = getLocalMotionMetric(videoArr, filt, detector, bf, MATCH_THRES, RANSAC_THRES)
