'''
download_data.py

Script to download Rosetta P67 images and .lbl files from ESA.

'''

from bs4 import BeautifulSoup, SoupStrainer

import argparse
import httplib2
import math
import os
import sys
import urllib


def print_progress_bar(progress, length=40, fill_char='=', empty_char=' ',
        side_char='|'):
    '''
    Prints a simple progress bar using carriage returns.

    Args:
        progress (int): The progress done as a percentage 0-100
    Args (optional):
        length (int): How long the progress bar should be.
        fill_char (str): The character used to draw the bar.
        empty_char (str): The character used to draw the empty space.
        side_char (str): The character used to draw the boundaries of the bar.
    '''

    progress = int(progress) # Force int

    if progress < 0 or 100 < progress:
        raise RuntimeError('Progress may only be from 0-100.')

    not_prog = 100-progress

    done = fill_char  * int(  math.ceil( length*progress/100.0 ) )
    left = empty_char * int( math.floor( length*not_prog/100.0 ) )

    sys.stdout.write(
        '\r{0}{1}{2}{0} {3:3d}%'.format(side_char, done, left, progress)
    )
    sys.stdout.flush()


def url_for_item(category, item):
    '''
    Returns the appropriate URL for a given image in a category.

    Args:
        category (int): The category of the image to query.
        item (int): The item number to grab from the category.
    Returns:
        str, the URL to get the appropriate item.
    '''

    return url_for_item.base.format(item, category)

url_for_item.base = (
    'http://imagearchives.esac.esa.int/picture.php?/{}/category/{}'
)


def scrape_files(img_dir='data/img', lbl_dir='data/lbl', category=63,
        item_start=6900, num_items=1):
    '''
    Scrapes images of comet P67 from esac.

    Args (optional):
        img_dir (str): The location to store images.
        lbl_dir (str): The location to store .lbl files.
        category (int): The category to download images from on esac.
        item_start (int): The item number to start from in the category.
        num_items (int): The number of items to grab.
    '''

    print 'Fetching files for images {}-{}...'.format(
            item_start, item_start+num_items)

    # Ensure that there's a / appended to each path
    if not img_dir.endswith('/'):
        img_dir = img_dir + '/'
    if not lbl_dir.endswith('/'):
        lbl_dir = lbl_dir + '/'

    # Ensure that directories are created
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(lbl_dir):
        os.makedirs(lbl_dir)


    http = httplib2.Http()

    for item in range(item_start, item_start+num_items):

        print_progress_bar(100*(item-item_start)/float(num_items))

        try:

            # Grab HTML page

            status, response = http.request(url_for_item(category, item))

            # Find link for processed .png file

            png = BeautifulSoup(response, 'html.parser').find(id='theMainImage')

            # Fix link - hackiest way I could do this.
            #
            #   Basically, the image link has a redirect to the actual image
            #   file. Get around this by chopping the first and last bits off,
            #   and hoping that the link sticks.

            img_link = (
                'http://imagearchives.esac.esa.int/' +
                png['src'][8:-7] + '.png'
            )

            # Find link for .LBL file (buried within a bunch of other links)

            links = BeautifulSoup(
                response,
                'html.parser',
                parse_only=SoupStrainer('a')
            )
            lbls = [
                x for x in links
                    if x.has_attr('href')
                    and x['href'].endswith('.LBL')
            ]

            if len(lbls) > 1:
                raise RuntimeError('More than one .LBL link found!')

            # Fetch the files and save them

            urllib.urlretrieve(img_link, img_dir+'{:06}.png'.format(item))

            urllib.urlretrieve(
                lbls[0]['href'],
                lbl_dir+'{:06}.lbl'.format(item),
            )

        except Exception, error:
            print 'Warning: Unable to download file - {}'.format(error)

    # End progress bar
    print_progress_bar(100)
    print ''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Scrapes images of processed comet 67P and corresponding flight '
            'data from esa.int.'
        )
    )
    parser.add_argument(
        '-start',
        type=int,
        default=6900,
        help='The item # to start from.'
    )
    parser.add_argument(
        '-num_items',
        type=int,
        default=5,
        help='The number of images to grab.'
    )
    parser.add_argument(
        '-img_dir',
        type=str,
        default='data/img/',
        help='The directory to store image files in.'
    )
    parser.add_argument(
        '-lbl_dir',
        type=str,
        default='data/lbl/',
        help='The directory to store .lbl files in.'
    )

    args = parser.parse_args()

    scrape_files(
        item_start = args.start,
        num_items  = args.num_items,
        img_dir    = args.img_dir,
        lbl_dir    = args.lbl_dir,
    )

    print 'Done!'


