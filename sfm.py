'''
sfm.py

Driver script to run Structure from Motion on comet images.

'''

import sfm

import argparse, sys


def main():

    parser = argparse.ArgumentParser(description='A program to apply structure '
            'from motion to create 3D reconstructions of comets and '
            'estimate digital elevation maps.')

    parser.add_argument(
        '-data_location',
        type=str,
        default='data/img/',
        help='Specifies the directory where the data lives. Data should be '
             'organized into img and lbl directories.',
    )
    parser.add_argument(
        '-ids',
        nargs=2,
        type=str,
        default=[None, None],
        help='The image ids to use.',
    )
    parser.add_argument(
        '-sift_ratio',
        type=float,
        default=0.8,
        help='The ratio to use when matching SIFT features.',
    )
    parser.add_argument(
        '-skip',
        type=int,
        default=16,
        help='The number of pixels to skip over when finding correspondences.',
    )
    parser.add_argument(
        '-window',
        type=int,
        default=5,
        help='The window size to use when finding correspondences.',
    )
    parser.add_argument(
        '-nofilter',
        action='store_true',
        help='Do not filter the resulting point cloud.',
    )

    parser.add_argument(
        '-show_features',
        action='store_true',
        help='Show SIFT features once found.',
    )
    parser.add_argument(
        '-show_matches',
        action='store_true',
        help='Show matches once found.',
    )
    parser.add_argument(
        '-show_epilines',
        action='store_true',
        help='Show epilines once found.',
    )
    parser.add_argument(
        '-show_correspondence',
        action='store_true',
        help='Show correspondences as they\'re found.',
    )
    parser.add_argument(
        '-show_point_cloud',
        action='store_true',
        help='Show point cloud in VTK viewer.',
    )
    parser.add_argument(
        '-show_depth_map',
        action='store_true',
        help='Show the resulting depth map.',
    )

    # Parse args and run

    args = parser.parse_args()

    # Set params on sfm module
    sfm.SHOW_FEATURES       = args.show_features
    sfm.SHOW_MATCHES        = args.show_matches
    sfm.SHOW_EPILINES       = args.show_epilines
    sfm.SHOW_CORRESPONDENCE = args.show_correspondence
    sfm.SHOW_POINT_CLOUD    = args.show_point_cloud
    sfm.SHOW_DEPTH_MAP      = args.show_depth_map

    sfm.SIFT_MATCHING_RATIO = args.sift_ratio
    sfm.CORRESPONDENCE_SKIP = args.skip
    sfm.WINDOW_RADIUS       = args.window
    sfm.FILTER_POINT_CLOUD  = not args.nofilter


    images = sfm.load67PImages(
        directory=args.data_location,
        id1=args.ids[0],
        id2=args.ids[1],
    )
    sfm.StructureFromMotion(images).run()


if __name__ == '__main__':
    main()

