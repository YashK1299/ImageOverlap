import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

#FLANN_INDEX_LSH = 6
image_name = ''		#Enter file path here
MIN_MATCH_COUNT = 15;
i = 0;
max_avg_count = 20;
frame_skip = max_avg_count;
a = np.zeros((max_avg_count,4,1,2));

#surf = cv2.xfeatures2d.SURF_create(1000);
surf = cv2.ORB_create();
#bf = cv2.BFMatcher();
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True);

def detectFeatures(kp1, des1, img2, j):
    i = j%max_avg_count;
    kp2, des2 = surf.detectAndCompute(img2, None);
    
    """FLANN_INDEX_KDTREE = 0;
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5);
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    search_params = dict(checks = 50);
    
    flann = cv2.FlannBasedMatcher(index_params, search_params);
    
    matches = flann.knnMatch(des1,des2,k=2);"""
    
    # store all the good matches as per Lowe's ratio test.
    #matches = bf.knnMatch(des1, des2, k=2);
    good = [];
    # Need to draw only good matches, so create a mask
    good = bf.match(des1, des2);
    """print(np.shape(matches), dir(matches));
    for (m, n) in matches:
        if m.distance < 0.7*n.distance:
            good.append(m);"""
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2);
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2);
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0);
        matchesMask = mask.ravel().tolist();
        
        h,w = img1.shape;
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2);
        dst = cv2.perspectiveTransform(pts,M);

        a[i] = dst;
        #print(a.shape, np.average(a, axis=0));
        #average = (avg[0] + avg[2] + avg[3] + avg[4] + avg[5])/5;

    else:
        #print("Not enough matches are found -", len(good)/MIN_MATCH_COUNT);
        matchesMask = None;
        
    avg = np.average(a, axis=0);
    img2 = cv2.polylines(img2,[np.int32(avg)],True,255,3, cv2.LINE_AA);
    #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   #singlePointColor = None,
                   #matchesMask = matchesMask, # draw only inliers
                   #flags = 2);

    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params);
    
    #plt.imshow(img3, 'gray'),plt.show()


cap = cv2.VideoCapture(image_name);		# Add video location
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('1.mp4');
ret, frame = cap.read();
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
img1 = gray;
kp1, des1 = surf.detectAndCompute(img1, None);

frame_ctr = 0;

while(True):
    ret, frame = cap.read();
    frame_ctr = frame_ctr+1;
    if(frame_ctr % 10 != 0):
        continue;
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    gray = cv2.resize(gray, None, fx = 0.5, fy = 0.5);
    img2 = gray;
    t = cv2.waitKey(1);
    if t == 27:
        break;
    elif (t == ord('s')):
        img1 = gray;
        kp1, des1 = surf.detectAndCompute(img1, None);
    """elif t == ord('d'):
        continue;"""
    detectFeatures(kp1, des1, img2, i%max_avg_count);
    i = i+1;
    cv2.imshow('frame', gray);

cap.release();
cv2.destroyAllWindows();
