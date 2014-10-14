""" Scrapes images from previously extracted search results in the shape
    of csv files.
"""
import urllib
import csv
import sys
import cPickle as pickle
import os
import os.path

def load_folder(input_folder, output_folder, nb_pages, nb_images, name, url_set):
    """ Loads images from all the csv files in an input folder into an output folder.
    """
    print "Starting download for " + name
    image_index = 0
    for csv_index in range(1, nb_pages + 1):
        print "Downloading page " + repr(csv_index) + "..."
        # Load the csv lines.
        input_filename = os.path.join(input_folder, name + '_page' + repr(csv_index) + '.csv')
        with open(input_filename, 'r') as csv_file:
            for row in csv.reader(csv_file):
                # The URL of the 480 max width jpeg version is on the 10th column.
                image_url = row[9]
                # If the URL has already been encountered, skip.
                if image_url in url_set:
                    continue
                output_filename = os.path.join(output_folder,
                                               name + '_' + repr(image_index) + '.jpg')
                image_index += 1
                if not os.path.isfile(output_filename):
                    urllib.urlretrieve(image_url, output_filename)
                    url_set.add(image_url)
                if image_index == nb_images:
                    print "Finished collecting " + repr(nb_images) + " for " + name
                    return

if __name__ == "__main__":
    # Takes 3 arguments: an input folder, a number of pages per folder, and an
    # output folder.
    if len(sys.argv) < 4:
        raise ValueError("Please input an input folder, a number of pages, a number of images and an output folder.")
    input_folder = sys.argv[1]
    nb_pages = int(sys.argv[2])
    nb_images = int(sys.argv[3])
    output_folder = sys.argv[4]
    url_set_filename = os.path.join(output_folder, 'url_set')
    url_set = set()
    # Load the already encountered URL set.
    if os.path.isfile(url_set_filename):
        with open(url_set_filename, 'r') as url_set_file:
            url_set = pickle.load(url_set_file)
    try:
        # Take all the folders in the input folders one by one.
        folders = os.walk(input_folder).next()[1]
        for folder in folders:
            full_in_folder = os.path.join(input_folder, folder)
            full_out_folder = os.path.join(output_folder, folder)
            try:
                os.makedirs(full_out_folder)
            except:
                pass
            load_folder(full_in_folder, full_out_folder, nb_pages, nb_images, folder, url_set)
    finally:
        # In case of unexpected crash, still write url_set to a file.
        with open(url_set_filename, 'w') as url_set_file:
            pickle.dump(url_set, url_set_file)
