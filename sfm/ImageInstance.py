'''
ImageInstance.py

Class to hold information about an image and its geometric relationship
to other images.

'''

import sfm

import NavcamInfo
import Reconstruction
import Util

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy

import os, sys



class ImageInstance:
    ''' Class to hold images and related flight information. '''

    def __init__(self, id, img):
        '''
        Constructor for ImageData object.

        Args:
            id (str): A unique ID for this instance (e.g. the filename).
            img (numpy.ndarray): The image for this instance.
        '''

        self.id  = id
        self.img = img

        self.matches     = dict() # Keypoint matches
        self.calibration = dict() # Calibration information
        self.transforms  = dict() # Projective transforms

        # ^ We store these as dicts with information corresponding to another
        # instance for the pipe dream of actually making this thing go beyond
        # just two views...

        self.intrinsic = NavcamInfo.intrinsic


    def __hash__(self):
        return hash(self.id)


    def __eq__(self, other):
        return self.id == other.id


    def find_features(self):
        '''
        Finds features in the image.
        '''

        self.features = cv2.SIFT().detectAndCompute(self.img, None)


    def match_features(self, instance):
        '''
        Matches features to another image and computes relevant matrices.

        Args:
            instance (ImageInstance): The image to match to.
        '''

        kp1, des1 = self.features
        kp2, des2 = instance.features

        matches = Reconstruction.find_matches(kp1, des1, kp2, des2)
        matches, xform = Reconstruction.find_inliers(kp1, kp2, matches)

        self.matches[instance] = matches
        self.transforms[instance] = xform

        # Perform calibration as well
        self.calibrate(instance)


    def calibrate(self, instance):
        '''
        Calibrates the current instance with another by finding the relative
        fundamental, essential, and extrinsic matrix.

        Args:
            instance (ImageInstance): The other instance to calibrate with.
        '''

        kp1, _ = self.features
        kp2, _ = instance.features

        fundamental, self.matches[instance] = Reconstruction.\
                estimate_fundamental(kp1, kp2, self.matches[instance])

        essential = Reconstruction.essential_from_fund(self.intrinsic,
                instance.intrinsic, fundamental)

        rotation, translation = Reconstruction.extract_relative_pos(essential)
        extrinsic = Reconstruction.extrinsic_matrix(rotation, translation)

        # Store information in relative calibration dict entry

        calibration_info = dict()
        calibration_info['essential']   = essential
        calibration_info['fundamental'] = fundamental
        calibration_info['rotation']    = rotation
        calibration_info['translation'] = translation
        calibration_info['extrinsic']   = extrinsic

        self.calibration[instance] = calibration_info


    def triangulate_points(self, instance):
        '''
        Finds the depth map of the image using calibration information from
        another image.

        Args:
            instance (ImageInstance): The other instance to triangulate from.
        Returns:
            numpy.ndarray, triangulated points in the world.
        '''

        rows, cols, _ = self.img.shape

        # Find corresponding points between images

        correspondences = Reconstruction.find_correspondences(self.img,
                instance.img, self.calibration[instance]['fundamental'],
                self.transforms[instance])

        pts1 = np.float32(
            [key for key, val in correspondences.iteritems()]
        ).reshape(-1, 1, 2)
        pts2 = np.float32(
            [val for key, val in correspondences.iteritems()]
        ).reshape(-1, 1, 2)

        # Triangulate points from correspondences

        P1 = np.hstack( (np.eye(3), np.zeros((3,1))) )
        P2 = self.calibration[instance]['extrinsic']

        points = Reconstruction.triangulate_points(pts1, pts2, P1, P2,
                self.intrinsic)

        # Create depth map

        depth = np.empty((rows, cols))
        depth.fill(np.min(points[:,-1]))

        skip = sfm.CORRESPONDENCE_SKIP

        for idx, pos in enumerate(correspondences.iteritems()):
            x,y = pos[0]
            depth[y:y+skip,x:x+skip] = points[idx][-1]


        return points, depth





def load67PImages(id1=None, id2=None, directory='data/img/'):
    '''
    Loads images of comet 67P from the Rosetta spacecraft. If no ids are
    provided, the function will just load the first two image files in the
    given directory.

    Args (optional):
        directory (str): The directory where the data lives. Expects there
            to be img and lbl directories for appropriate data.
        id1 (str): The id of the first image to load.
        id2 (str): The id of the second image to load.
    Returns:
        ImageInstance, the first image instance.
        ImageInstance, the second image instance.
    '''

    if not directory.endswith('/'):
        directory = directory + '/'

    img1_filename = ''
    img2_filename = ''

    if id1 is None or id2 is None:

        files = [x for x in os.listdir(directory)
                    if x.endswith('.png') or x.endswith('.jpg')]

        if len(files) < 2:
            raise RuntimeError('Not enough images in data directory!')

        img1_filename = files[0]
        img2_filename = files[1]

    else:
        img1_filename = [x for x in os.listdir(directory) if id1 in x][0]
        img2_filename = [x for x in os.listdir(directory) if id2 in x][0]

        if not img1_filename or not img2_filename:
            raise RuntimeError('Unable to find specified images!')

    img1 = cv2.imread(directory+img1_filename)
    img2 = cv2.imread(directory+img2_filename)

    instance1 = ImageInstance(img1_filename[:-4], img1)
    instance2 = ImageInstance(img2_filename[:-4], img2)

    return instance1, instance2


