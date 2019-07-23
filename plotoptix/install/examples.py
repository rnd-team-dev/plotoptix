import os, zipfile
from plotoptix.install import download_file_from_google_drive

def install_examples():
    """Install plotoptix examples.
    """
    print("Downloading plotoptix examples...")

    folder = os.getcwd()

    id = "1Bdq7SnvI3fA12_-LoaF31h-d5E67T_32"
    file_name = os.path.join(os.getcwd(), "examples.zip")

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
