import plotoptix, requests, os, zipfile
from plotoptix._load_lib import BIN_PATH, PLATFORM

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

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

def install_denoiser():
    """Install denoiser binaries.
    """
    print("Downloading denoiser binaries...")

    if PLATFORM == "Windows":
        id = "1qLyR7c_upFJKxZDKQCLuDRC3pc-iwuh0"
        file_name = "denoiser_libs_win.zip"
        cudnn_lib = "cudnn64_7.dll"
        denoiser_lib = "optix_denoiser.6.0.0.dll"
    elif PLATFORM == "Linux":
        id = "1LrtDm2TXx8Rs-gZVIhSkOdkCfNFz_Tsq"
        file_name = "denoiser_libs_linux.zip"
        cudnn_lib = "libcudnn.so.7.3.1"
        denoiser_lib = "liboptix_denoiser.so.6.0.0"
    else:
        raise NotImplementedError

    folder = os.path.join(os.path.dirname(plotoptix.__file__), BIN_PATH)

    file_name = os.path.join(folder, file_name)
    cudnn_lib = os.path.join(folder, cudnn_lib)
    denoiser_lib = os.path.join(folder, denoiser_lib)

    try:
        download_file_from_google_drive(id, file_name)
    except:
        print("downloading failed.")
        return False

    if os.path.isfile(cudnn_lib): os.remove(cudnn_lib)
    if os.path.isfile(denoiser_lib): os.remove(denoiser_lib)

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
