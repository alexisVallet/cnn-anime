""" Scrapes images from previously extracted search results in the shape
    of csv files.
"""
import urllib
import csv
import sys
import cPickle as pickle
import os
import os.path

def load_folder(input_folder, output_folder, nb_pages, name, url_dict):
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
                # If the URL has already been encountered, add the name to the dict
                # and skip downloading.
                if image_url in url_dict:
                    url_dict[image_url][1].add(name)
                else:
                    output_filename = os.path.join(output_folder,
                                                    name + '_' + repr(image_index) + '.jpg')
                    url_dict[image_url] = (output_filename, set([name]))
                    image_index += 1
                    if not os.path.isfile(output_filename):
                        urllib.urlretrieve(image_url, output_filename)

if __name__ == "__main__":
    # Takes 3 arguments: an input folder, a number of pages per folder, and an
    # output folder.
    if len(sys.argv) < 4:
        raise ValueError("Please input an input folder, a number of pages and an output folder.")
    input_folder = sys.argv[1]
    nb_pages = int(sys.argv[2])
    output_folder = sys.argv[3]
    url_labels_dict_fname = os.path.join(output_folder, 'url_labels_dict')
    url_labels_dict = {}
    if os.path.isfile(url_labels_dict_fname):
        with open(url_labels_dict_fname, 'r') as url_labels_dict_file:
            url_labels_dict = pickle.load(url_labels_dict_file)
    try:
        # Take all the folders in the input folders one by one.
        folders = os.walk(input_folder).next()[1]
        full_out_folder = os.path.join(output_folder, 'images')
        for folder in folders:
            full_in_folder = os.path.join(input_folder, folder)
            load_folder(full_in_folder, full_out_folder, nb_pages, folder, url_labels_dict)
    finally:
        with open(url_labels_dict_fname, 'w') as url_labels_dict_file:
            pickle.dump(url_labels_dict, url_labels_dict_file)
