"""
Install scripts for PlotOptiX extras.

Documentation: https://plotoptix.rnd.team
"""

__author__  = "Robert Sulej, R&D Team <dev@rnd.team>"
__status__  = "beta"
__version__ = "0.4.0"
__date__    = "05 July 2019"

import logging

logging.basicConfig(level=logging.WARN, format='[%(levelname)s] (%(threadName)-10s) %(message)s')

import requests, os

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    n = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                n += 1

                if n % 8 == 0:
                    s = int(os.path.getsize(destination) / 1024)
                    print(str(s) + " kB ready", end="\r")

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)
