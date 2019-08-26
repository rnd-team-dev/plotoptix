import os, zipfile
from plotoptix.install import download_file_from_google_drive

def install_project(file_name, id):
    """Install plotoptix examples.
    """
    print("Downloading %s..." % file_name)

    folder = os.getcwd()

    file_name = os.path.join(os.getcwd(), file_name)

    try:
        download_file_from_google_drive(id, file_name)
    except:
        print("downloading failed.")
        return False

    print("Uncompressing...                ")

    try:
        zip_ref = zipfile.ZipFile(file_name, "r")
        zip_ref.extractall(folder)
        zip_ref.close()
    except:
        print("failed.")
        return False

    print("Clean up...")

    if os.path.isfile(file_name): os.remove(file_name)

    print("All done.")
    return True
