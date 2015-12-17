'''
TestSFM.py

Unit tests for structure from motion module.

'''

import ImageInstance
import NavcamInfo
import Reconstruction
import StructureFromMotion
import Util

import numpy as np

import unittest


epsilon = 1e-12 # Allowed error for floating point ops


class TestSFM(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.instances = ImageInstance.load67PData(max_instances=2)

        for instance in self.instances:
            instance.find_features()

        self.instances[0].match_features(self.instances[1])

    def checkArrays(self, a, b):
        ''' Check if two arrays are equal up to some epsilon. '''

        return np.all(np.abs(a-b) < epsilon)

    def testNavcamIntrinsic(self):
        ''' Check trig on computing the focal length. '''

        fov = np.arctan2(NavcamInfo.image_width/2, NavcamInfo.focal_length_px)
        fov = np.rad2deg(2*fov)

        self.assertEqual(fov, NavcamInfo.field_of_view)

    def testFundamentalEssential(self):
        ''' Check that the fundamental and essential matrix equations are
            equivalent. '''

        kp1, _ = self.instances[0].features
        kp2, _ = self.instances[1].features
        matches = self.instances[0].matches[self.instances[1]]

        fund, _ = Reconstruction.estimate_fundamental(kp1, kp2, matches)

        esst = Reconstruction.essential_from_fund(
                self.instances[0].intrinsic,
                self.instances[1].intrinsic,
                fund,
        )
        fund2 = Reconstruction.fundamental_from_esst(
                self.instances[0].intrinsic,
                self.instances[1].intrinsic,
                esst,
        )
        self.assertTrue( self.checkArrays(fund, fund2) )

    def testExtrinsicAndProjection(self):
        ''' Try a simple example for extrinsic parameters of a camera. '''

        intrinsic = np.array([
            [1000,    0, 500, 0],
            [   0, 1000, 500, 0],
            [   0,    0,   1, 0],
        ])

        rot1 = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1],
        ])
        trans1 = np.array([0,0,0])

        rot2 = np.array([
            [0,0,1],
            [0,1,0],
            [1,0,0],
        ])
        trans2 = np.array([1000,1000,0])

        ext1 = Reconstruction.extrinsic_matrix(rot1, trans1)
        ext2 = Reconstruction.extrinsic_matrix(rot2, trans2)

        P1 = Reconstruction.projection_matrix(ext1, None)
        P2 = Reconstruction.projection_matrix(ext2, ext1)


if __name__ == '__main__':
    unittest.main()

