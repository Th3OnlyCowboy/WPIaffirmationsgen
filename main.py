# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:02:06 2020

@author: OHyic

Modified on Sun Jan 16 2022

"""
# Import libraries
from GoogleImageScrapper import GoogleImageScraper
import os
import psycopg2


def add_to_db(url, isanime):
    conn_string = "postgresql://tenheller:q8NCFp26rJ1wf1qf@free-tier.gcp-us-central1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full&sslrootcert=C:\\Users\\kahel\\AppData\\Roaming\\.postgresql\\root.crt&options=--cluster%3Dtenler-test-5472"
    conn = psycopg2.connect(os.path.expandvars(conn_string))
    if isanime:
        tablename = "images_test"
    else:
        tablename = "backgrounds_test"
    try:
        conn.cursor().execute("insert into " + tablename + " VALUES (DEFAULT, \'" + url + "\', DEFAULT)")
    except psycopg2.errors.UniqueViolation:
        print("sql error\n")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Define file path
    webdriver_path = os.path.normpath(os.getcwd() + "\\webdriver\\chromedriver.exe")
    image_path = os.path.normpath(os.getcwd() + "\\photos")

    # Add new search key into array ["cat","t-shirt","apple","orange","pear","fish"]
    search_keys = ['anime png']

    # Parameters
    number_of_images = 100
    headless = False
    min_resolution = (0, 0)
    max_resolution = (9999, 9999)

    # Main program
    for search_key in search_keys:
        image_scrapper = GoogleImageScraper(webdriver_path, image_path, search_key, number_of_images, headless,
                                            min_resolution, max_resolution)
        image_urls = image_scrapper.find_image_urls()
        # image_scrapper.save_images(image_urls)
        for url in image_urls:
            print(url)
            add_to_db(url, False)

    # Release resources
    del image_scrapper
