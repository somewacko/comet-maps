'''
Util.py

'''

import cv2
import numpy as np

import math, random, sys


BLUE  = (255,   0,   0)
GREEN = (  0, 255,   0)
RED   = (  0,   0, 255)


# ---- Progress Bar Methods

def print_progress_bar(progress, message, length=40, fill_char='=', empty_char=' ',
        side_char='|'):
    '''
    Prints a simple progress bar using carriage returns.

    Args:
        progress (int): The progress done as a percentage 0-100.
        message (str): A message to show with the progress bar.
    Args (optional):
        length (int): How long the progress bar should be.
        fill_char (str): The character used to draw the bar.
        empty_char (str): The character used to draw the empty space.
        side_char (str): The character used to draw the boundaries of the bar.
    '''

    progress = int(progress)

    if progress < 0 or 100 < progress:
        raise RuntimeError('Progress may only be from 0-100.')

    not_prog = 100-progress

    done = fill_char  * int(  math.ceil( length*progress/100.0 ) )
    left = empty_char * int( math.floor( length*not_prog/100.0 ) )

    sys.stdout.write(
            '\r{0: <20}{1}{2}{3}{1} {4:3d}%'.format(message, side_char,
                done, left, progress)
    )
    sys.stdout.flush()


def finish_progress_bar():
    '''
    Fills the progress bar to 100% and prints a newline.
    '''

    print_progress_bar(100, 'Finished!')
    sys.stdout.write('\n')


# ---- Display Methods

def random_colors():
    '''
    Generates a list of randomly-sorted colors.

    Returns:
        list<tuple>, bgr color tuples.
    '''

    colors = list()

    for b in range(64, 196, 32):
        for g in range(64, 196, 32):
            for r in range(64, 196, 32):
                colors.append((b,g,r))
    random.shuffle(colors)

    return colors


def combine_images(img1, img2):
    '''
    Combines two images side-by-side.

    Args:
        img1, img2 (nunmpy.ndarray): Images to be displayed together.
    Returns:
        numpy.ndarray, combined image.
    '''

    rows1, cols1, d = img1.shape
    rows2, cols2, d = img2.shape

    # Pad shorter image with zeros if they aren't same size
    if rows1 != rows2:
        if rows1 < rows2:
            zeros = np.zeros((rows2-rows1, cols1, d))
            img1 = np.concatenate((img1, zeros), axis=0)
        else:
            zeros = np.zeros((rows1-rows2, cols2, d))
            img2 = np.concatenate((img2, zeros), axis=0)

    # Place images side by side
    return np.concatenate((img1, img2), axis=1)


def draw_matches(img1, kp1, img2, kp2, matches):
    '''
    Draws matches between features in two images.

    Args:
        img1, img2 (numpy.ndarray): Images to be displayed together.
        keypoints (list<tuple>): Tuples of keypoint pairs.
    Returns:
        numpy.ndarray, image with matched keypoints.
    '''

    _, cols, _ = img1.shape

    # Place images side by side
    img = combine_images(img1, img2)

    colors = random_colors()

    # Draw matches
    for idx, match in enumerate(matches):

        idx1 = match.queryIdx
        idx2 = match.trainIdx

        x1, y1 = int(kp1[idx1].pt[0]), int(kp1[idx1].pt[1])
        x2, y2 = int(kp2[idx2].pt[0]), int(kp2[idx2].pt[1])

        x2 += cols

        color = colors[idx%len(colors)]

        cv2.circle(img, (x1,y1), 8, color, 2)
        cv2.circle(img, (x2,y2), 8, color, 2)
        #cv2.line(img, (x1,y1), (x2,y2), color, 1)

    return img


def draw_epilines(img1, kp1, img2, kp2, matches, fund):
    '''
    Draws epilines between two images.

    Args:
        ...
    '''

    _, cols, _ = img1.shape

    # Place images side by side
    img = combine_images(img1, img2)

    colors = random_colors()

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)
    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    lines2 = cv2.computeCorrespondEpilines(pts1, 1, fund)
    lines1 = cv2.computeCorrespondEpilines(pts2, 2, fund)

    try:
        # Draw matches
        for i in range(0, len(pts1)):

            color = colors[i%len(colors)]

            x1, y1 = int(pts1[i][0][0]), int(pts1[i][0][1])
            x2, y2 = int(pts2[i][0][0]), int(pts2[i][0][1])

            x2 += cols

            a1, b1, c1 = lines1[i][0]
            a2, b2, c2 = lines2[i][0]

            cv2.circle(img, (x1,y1), 8, color, 2)
            cv2.circle(img, (x2,y2), 8, color, 2)

            y11 =          -c1  / b1 # At x = 0
            y12 = (-a1*cols-c1) / b1 # At x = col

            y21 =          -c2  / b2 # At x = 0
            y22 = (-a2*cols-c2) / b2 # At x = col

            cv2.line(img, (0,    int(y11)), (cols,   int(y12)), color, 1)
            cv2.line(img, (cols, int(y21)), (2*cols, int(y22)), color, 1)

        return img

    except:
        print 'Warning: Could not create epilines'
        return None





