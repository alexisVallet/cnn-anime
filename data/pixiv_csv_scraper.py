""" Scrapes search results off of Pixiv's iPhone app API.
"""
import os.path
import urllib
import requests
import os.path
import sys

def pixiv_scraper(tag, nb_images, php_sessionid, outputfolder):
    """ Scrapes CSV results from Pixiv's search engine. Doesn't
        actually download the corresponding image files.
    """
    nb_pages = nb_images / 50
    
    for page in range(1, nb_pages+1):
        print "Scraping page " + repr(page) + "..."
        base_url = 'http://spapi.pixiv.net/iphone/search.php'
        api_params = {
            's_mode': 's_tag',
            'order': 'popular_d',
            'word': tag,
            'p': repr(page),
            'PHPSESSID': php_sessionid
        }
        encoded_params = urllib.urlencode(api_params)
        full_url = base_url + '?' + encoded_params
        output_filename = os.path.join(
            outputfolder,
            tag + '_page' + repr(page) + '.csv'
        )
        # Pass if the filename already exists.
        if os.path.isfile(output_filename):
            continue
        urllib.urlretrieve(
            full_url, 
            filename=output_filename
        )

if __name__ == "__main__":
    # Two command-line arguments: a file listing tags, and
    # the PHP session id.
    if len(sys.argv) < 3:
        raise ValueError("Please input a file listing tags, the PHP session ID and an output folder.")
    tags_fname = sys.argv[1]
    php_sessionid = sys.argv[2]
    out_folder = sys.argv[3]

    with open(tags_fname) as tags_file:
        tags = map(lambda fn: fn[0:len(fn)-1], tags_file.readlines())
        for tag in tags:
            tag_out_folder = os.path.join(out_folder, tag)
            try:
                os.makedirs(tag_out_folder)
            except:
                pass
            print "Scraping for " + tag + "..."
            pixiv_scraper(tag, 13000, php_sessionid, tag_out_folder)
