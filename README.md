# ImageOverlap
Pipeline:
    1. Capture video
    2. Detect features using one of sift, surf and orb
    3. Find correspondances using valid matcher from brute-force or flann
    4. Find best set of matches
    5. Find homography using RANSAC or LEAST_MEDIAN algorithm
    6. Transform perspective
-- min_match_count used to determine if not enough matches found after bf matcher
-- surf object created with low parameter value for high accuracy (can use SIFT but keep parameter value - high for high accuracy), can use orb also much faster and robust
-- bfmatcher object created (can use other matches like Flann) Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation. And the closest one is returned.
-- detect and compute features
-- BFMatcher.knnMatch() =>returns k best matches where k is specified by the user. It may be useful when we need to do additional work on that.
--BFMatcher.match() => gives best match
-- Taking good matches using Lowe's ratio test: only for knnMatches in BF and for Flann
-- src_pts - coordinates of good keypoints in image1
-- dst_pts - coordinates of good keypoints in image2
-- cv2.findHomography() returns a mask which specifies the inlier and outlier points. There can be some possible errors while matching which may affect the result. To solve this problem, algorithm uses RANSAC or LEAST_MEDIAN (which can be decided by the flags). So good matches which provide correct estimation are called inliers and remaining are called outliers.
-- perspectiveTransform() finds the new coordinates of the points in the current frame.
-- Capture video and saving 1st frame as ref. frame
-- main loop for detecting features and homography for each frame
Further details: 
    • cv2.findHomography() - https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    • Feature – Matchers - https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
    • Paper: http://www.cs.toronto.edu/~kyros/courses/2530/papers/Lecture-14/Szeliski2006.pdf
Run Image-Overlap:
    • 1. Attach a webcam or change the image filename to your image name with directory.
    • 2. Run the script
