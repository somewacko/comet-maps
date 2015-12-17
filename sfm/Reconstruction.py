'''
Reconstruction.py

Functions to assist with geometric reconstruction of images.

'''

import sfm

import cv2
import numpy as np
import scipy

import Util


def format_points(kp1, kp2, matches):
    '''
    Format points according to matches for use with OpenCV's API.

    Args:
        kp1 (list<cv2.KeyPoint>): Key points from the first image.
        kp2 (list<cv2.KeyPoint>): "" from the second image.
        matches (list<cv2.DMatch>): Matches between key points.
    Returns:
        numpy.ndarray, matched key points from kp1.
        numpy.ndarray, "" from kp2.
    '''

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)
    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    return pts1, pts2


def find_matches(kp1, des1, kp2, des2):
    '''
    Finds matches between keypoints using RANSAC and projective
    transformations to get rid of outliers.

    Args:
        kp1 (list<cv2.KeyPoint>): Key points from the first image.
        des1 (list<numpy.ndarray>): Descriptors of key points.
        kp2 (list<cv2.KeyPoint>): "" from the second image.
        des2 (list<numpy.ndarray>): "" from the second image.
    Return:
        list<cv2.DMatch>, all found matches.
    '''

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = list()

    for m,n in matches:
        if m.distance < sfm.SIFT_MATCHING_RATIO*n.distance:
            good_matches.append(m)

    return good_matches


def find_inliers(kp1, kp2, matches):
    '''
    Find inliers between a set of key point matches using RANSAC and
    projective transforms.

    Args:
        kp1 (list<cv2.KeyPoint>): The key points in the first image.
        kp2 (list<cv2.KeyPoint>): "" from second image.
        matches (list<cv2.DMatch>): Matches from kp1 to kp2.
    Return:
        list<cv2.DMatch>, all inlier matches.
    '''

    pts1, pts2 = format_points(kp1, kp2, matches)

    # Use RANSAC with projective alignment to find outliers
    xform, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5)
    mask = np.hstack(mask)

    # Extract inliers and return
    return [m for i, m in enumerate(matches) if mask[i]==1], xform


def estimate_fundamental(kp1, kp2, matches):
    '''
    Finds the fundamental matrix between a set of key point matches.

    Args:
        kp1 (list<cv2.KeyPoint>): The key points in the first image.
        kp2 (list<cv2.KeyPoint>): "" from second image.
        matches (list<cv2.DMatch>): Matches from kp1 to kp2.
    Returns:
        numpy.ndarray, the found fundamental matrix.
        list<cv2.DMatch>, all inlier matches.
    '''

    pts1, pts2 = format_points(kp1, kp2, matches)

    # Find the fundamental and essential matrices
    fundamental, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
            0.1, 0.9)

    # Extract inliers
    matches = [m for i, m in enumerate(matches) if mask[i]==1]

    return fundamental, matches


def essential_from_fund(intrinsic1, intrinsic2, fundamental):
    '''
    Extracts the essential matrix from the fundamental using the intrinsic
    matrices of the two cameras.

    Args:
        intrinsic1 (numpy.ndarray): The intrinsic matrix of the first camera.
        intrinsic2 (numpy.ndarray): The intrinsic matrix of the second camera.
        fundamental (numpy.ndarray): The fundamental matrix between the two.
    Returns:
        numpy.ndarray, the extracted essential matrix.
    '''

    return intrinsic1.T.dot(fundamental).dot(intrinsic2)


def fundamental_from_esst(intrinsic1, intrinsic2, essential):
    '''
    Extracts the fundamental matrix from the essential using the intrinsic
    matrices of the two cameras.

    Args:
        intrinsic1 (numpy.ndarray): The intrinsic matrix of the first camera.
        intrinsic2 (numpy.ndarray): The intrinsic matrix of the second camera.
        essential (numpy.ndarray): The essential matrix between the two.
    Returns:
        numpy.ndarray, the extracted essential matrix.
    '''

    inv_intrinsic1 = np.linalg.inv(intrinsic1)
    inv_intrinsic2 = np.linalg.inv(intrinsic2)

    return inv_intrinsic1.T.dot(essential).dot(inv_intrinsic2)


def extract_relative_pos(essential):
    '''
    Extracts relative rotation and translation from the essential matrix.

    Args:
        essentrial (numpy.ndarray): The essential matrix.
    Returns:
        numpy.ndarray, the relative rotation matrix.
        numpy.ndarray, the relative translation matrix.
    '''

    U,S,V = np.linalg.svd(essential)

    rotation = U.dot(np.array([[0,-1,0],[1,0,0],[0,0,1]])).dot(V)
    translation = U[:,2]

    return rotation, translation


def extrinsic_matrix(rotation, translation):
    '''
    Finds the extrinsic matrix from the relative geometric information and
    the reference camera's extrinsic matrix (LATTER PART NOT IMPLEMENTED)

    Args:
        rotation (numpy.ndarray): The relative rotation matrix.
        translation (numpy.ndarray) The relative translation matrix.
    Returns:
        numpy.ndarray, the extrinsic matrix.
    '''

    return np.hstack((rotation, translation[np.newaxis].T))


def triangulate_points(pts1, pts2, P1, P2, intr):
    '''
    Triangulates points from the camera image to the world.

    Args:
        pts1 (numpy.ndarray): Points in the first image.
        pts2 (numpy.ndarray): "" second image.
        P1 (numpy.ndarray): Projection matrix of the first camera.
        P2 (numpy.ndarray): "" second camera.
    Returns:
        numpy.ndarray, real-world coordinates for each point pair.
    '''

    '''
    pts = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts = cv2.convertPointsFromHomogeneous(pts.T)
    return np.vstack([pt[0] for pt in pts])
    '''

    points = list()

    length = pts1.shape[0]

    intr_inv = np.linalg.inv(intr)
    intr_proj = intr.dot(P2)

    for idx in range(0, length):

        u1 = np.hstack((pts1[idx][0], [1.]))
        u2 = np.hstack((pts2[idx][0], [1.]))

        u1 = intr_inv.dot(u1)
        u2 = intr_inv.dot(u2)

        u1, v1 = u1[0], u1[1]
        u2, v2 = u2[0], u2[1]

        A = np.array([
            [u1*P1[2,0]-P1[0,0], u1*P1[2,1]-P1[0,1], u1*P1[2,2]-P1[0,2]],
            [v1*P1[2,0]-P1[1,0], v1*P1[2,1]-P1[1,1], v1*P1[2,2]-P1[1,2]],
            [u2*P2[2,0]-P2[0,0], u2*P2[2,1]-P2[0,1], u2*P2[2,2]-P2[0,2]],
            [v2*P2[2,0]-P2[1,0], v2*P2[2,1]-P2[1,1], v2*P2[2,2]-P2[1,2]],
        ])

        b = np.array([
            [ -(u1*P1[2,3]-P1[0,3]) ],
            [ -(v1*P1[2,3]-P1[1,3]) ],
            [ -(u2*P2[2,3]-P2[0,3]) ],
            [ -(v2*P2[2,3]-P2[1,3]) ],
        ])

        pt = np.linalg.lstsq(A, b)[0]

        points.append(pt.T)

    return np.vstack(points)


def find_correspondences(img1, img2, fundamental, xform):
    '''
    Finds correspondences for each point between images.

    Args:
        instance (ImageInstance): The other instance to find
            correspondences in.
    Returns:
        dict<tuple, tuple>, a map of correspondences between (x,y) points
            in this image and (x,y) points in the other
    '''

    rows, cols, _ = img1.shape

    # Transform to get better correspondences using cross-correlation.
    #
    # This isn't *really* rectifying the image to line up correspondences,
    # but we were unable to get it to work the proper way so here's the next
    # best thing.

    xform = np.linalg.inv(xform) # Reverse transform
    timg = cv2.warpPerspective(img2, xform, (1024, 1024))

    # Find mask of the comet's silhouette

    def mask_for_img(img):
        _, mask = cv2.threshold(img, 24, 255, cv2.THRESH_BINARY)
        kernel = np.ones((4,4))
        mask = cv2.erode(mask, kernel)
        mask = np.sum(mask, axis=2)
        return mask

    mask1 = mask_for_img(img1)
    mask2 = mask_for_img(img2)

    # Get points in image that lie on mask

    pts = [(x,y) for y in range(0, rows, sfm.CORRESPONDENCE_SKIP)
                 for x in range(0, cols, sfm.CORRESPONDENCE_SKIP)
                     if mask1[y,x] > 0]
    pts = np.float32(pts)

    # Find epilines

    lines = cv2.computeCorrespondEpilines(pts, 1, fundamental)

    # Scan along epilines and find correspondences

    correspondences = dict()

    def normalize(x):
        return (x-np.mean(x))/np.std(x)

    imgpad1 = np.pad(img1, sfm.WINDOW_RADIUS, 'edge')
    imgpad2 = np.pad(timg, sfm.WINDOW_RADIUS, 'edge')

    win_size = 2*sfm.WINDOW_RADIUS + 1

    for i in range(0, len(pts)):
        a,b,c = lines[i][0]
        x,y   = int(pts[i][0]), int(pts[i][1])

        win1 = normalize(imgpad1[y:y+win_size,x:x+win_size])

        best_pos = None
        best_corr = 0

        for xx in range(0, cols):
            yy = int(np.round((-a*xx - c)/b))

            if 0 <= yy and yy < cols and mask2[yy,xx] > 0:

                # Find corresponding points in transformed image
                t_x = int( (xform[0,0]*xx + xform[0,1]*yy + xform[0,2]) /
                           (xform[2,0]*xx + xform[2,1]*yy + xform[2,2]) )
                t_y = int( (xform[1,0]*xx + xform[1,1]*yy + xform[1,2]) /
                           (xform[2,0]*xx + xform[2,1]*yy + xform[2,2]) )

                win2 = normalize(imgpad2[t_y:t_y+win_size,t_x:t_x+win_size])
                corr = np.sum(win1*win2) / win1.size

                if corr > best_corr:
                    best_pos = (xx,yy)
                    best_corr = corr

        if best_pos is not None:

            if sfm.SHOW_CORRESPONDENCE:

                img = Util.combine_images(img1, img2)
                cv2.circle(img, (x,y), 8, Util.GREEN, 2)
                cv2.circle(img, (best_pos[0]+1024, best_pos[1]), 8, Util.GREEN, 2)
                y1 = int(-c/b)
                y2 = int((-a*cols-c)/b)
                cv2.line(img, (cols, y1), (2*cols, y2), Util.BLUE, 1)
                img = scipy.misc.imresize(img, 0.5)
                cv2.imshow('debug', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            correspondences[(x,y)] = best_pos

    return correspondences


