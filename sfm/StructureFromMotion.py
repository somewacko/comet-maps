'''
StructureFromMotion.py

Primary class for performing SfM on comet images.

'''

import sfm

import ImageInstance
import Util

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pcl
import scipy.ndimage
import vtk_visualizer as vtk_vis

import os


class StructureFromMotion:

    def __init__(self, instances, output_dir='output/'):
        '''
        Constructor for StructureFromMotion object.

        Args:
            instances (list<ImageData>): ImageData objects to use for SfM.
            output (str): The output directory to save to.
        '''

        if len(instances) > 2:
            raise RuntimeError('Only two images for SfM are currently '
                               'supported!')

        if not output_dir.endswith('/'):
            output_dir = output_dir+'/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.instances = instances
        self.output_dir = output_dir


    def run(self):
        '''
        Runs SfM on the images.
        '''

        instance1, instance2 = self.instances


        # Initial report (spit back params)

        print ''
        print 'Using images {} and {}'.format(instance1.id, instance2.id)
        print ''
        print 'SIFT matching ratio   : {}'.format(sfm.SIFT_MATCHING_RATIO)
        print 'Window radius         : {}'.format(sfm.WINDOW_RADIUS)
        print 'Skip factor           : {}'.format(sfm.CORRESPONDENCE_SKIP)
        print 'Filtering point cloud : {}'.format(sfm.FILTER_POINT_CLOUD)
        print ''


        # Find features

        msg = 'Finding features...'

        Util.print_progress_bar(14, msg)
        instance1.find_features()

        Util.print_progress_bar(28, msg)
        instance2.find_features()

        if sfm.SHOW_FEATURES:
            self.DEBUG_show_features()


        # Match features

        msg = 'Matching...'

        Util.print_progress_bar(43, msg)
        instance1.match_features(instance2)

        if sfm.SHOW_MATCHES:
            self.DEBUG_show_matches()

        if sfm.SHOW_EPILINES:
            self.DEBUG_show_epilines()


        # Find correspondences

        msg = 'Triangulating...'

        Util.print_progress_bar(57, msg)
        points, depth = instance1.triangulate_points(instance2)


        # Names for output files

        name = '{}-{}-r{}s{:02}w{:02}'.format(instance1.id, instance2.id,
                sfm.SIFT_MATCHING_RATIO, sfm.CORRESPONDENCE_SKIP,
                sfm.WINDOW_RADIUS)

        grey_fig_name  = self.output_dir+name+'-grey.pdf'
        color_fig_name = self.output_dir+name+'-color.pdf'
        ptcld_name     = self.output_dir+name+'.ply'


        # Save depth map

        msg = 'Depth map...'

        Util.print_progress_bar(71, msg)

        plt.imshow(depth, cmap=plt.cm.bone)
        plt.savefig(grey_fig_name)

        plt.imshow(depth)
        plt.savefig(color_fig_name)

        if sfm.SHOW_DEPTH_MAP:
            self.DEBUG_show_depth_map(plt)
 

        # Create and save point cloud

        msg = 'Point cloud...'

        Util.print_progress_bar(86, msg)

        ptcld = pcl.PointCloud(np.float32(points))

        if sfm.FILTER_POINT_CLOUD:
            fil = ptcld.make_statistical_outlier_filter()
            fil.set_mean_k(50)
            fil.set_std_dev_mul_thresh(1.0)
            ptcld = fil.filter()

        pcl.save(ptcld, ptcld_name, format='ply')

        if sfm.SHOW_POINT_CLOUD:
            self.DEBUG_show_point_cloud()


        # Finish up

        Util.finish_progress_bar()

        print ''

        print 'Saved files:'
        print '\t'+grey_fig_name
        print '\t'+color_fig_name
        print '\t'+ptcld_name
        print ''


    # ---- Debug Methods

    def DEBUG_show_point_cloud(self, ptcld):

        print 'DEBUG: Showing point cloud'

        vtk_control = vtk_vis.VTKVisualizerControl()
        vtk_control.AddPointCloudActor(np.asarray(ptcld))
        vtk_control.SetActorColor(0, (1,1,1))
        vtk_control.Render()
        vtk_control.ResetCamera()

        raw_input()

    def DEBUG_show_depth_map(self, points):

        print 'DEBUG: Showing depth map'

        plt.show()

    def DEBUG_show_features(self):

        print 'DEBUG: Displaying features'

        for instance in self.instances:
            flag = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            img = cv2.drawKeypoints(instance.img,
                    instance.features[0], flags=flag)
            img = scipy.misc.imresize(img, 0.5)
            cv2.imshow('debug', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def DEBUG_show_matches(self):

        print 'DEBUG: Displaying matches'

        for idx, instance in enumerate(self.instances[0:-1]):

            other_instance = self.instances[idx+1]

            img = Util.draw_matches(
                instance.img,
                instance.features[0],
                other_instance.img,
                other_instance.features[0],
                instance.matches[other_instance],
            )
            img = scipy.misc.imresize(img, 0.5)

            cv2.imshow('debug', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def DEBUG_show_epilines(self):

        print 'DEBUG: Showing epilines'

        for idx, instance in enumerate(self.instances[0:-1]):

            other_instance = self.instances[idx+1]

            img = Util.draw_epilines(
                instance.img,
                instance.features[0],
                other_instance.img,
                other_instance.features[0],
                instance.matches[other_instance],
                instance.calibration[other_instance]['fundamental'],
            )
            if img is not None:
                img = scipy.misc.imresize(img, 0.5)

                cv2.imshow('debug-epilines', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


