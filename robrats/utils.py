import nibabel as nib
import numpy as np
from skimage.transform import resize
import os
import tqdm
import requests
import zipfile

def make_3_channel_NIFTI(basename, T2_path, ADC_path, DCE_path):
    """
    Combines three modalities into a 3-channel MRI
    basename (string): patient name in file
    T1_path (string): 
    """
    
    img_T2 = nib.load(T2_path)
    img_ADC = nib.load(ADC_path)
    img_DCE = nib.load(DCE_path)
            
    print(f"T2 shape: {img_T2.shape}")
    print(f"ADC shape: {img_ADC.shape}")
    print(f"DCE shape: {img_DCE.shape}")

    img_T2_shape = img_T2.get_fdata().shape
    #combined_data = np.zeros((img_T2_shape[0], img_T2_shape[1], 16, 3))
    combined_data = np.zeros((img_T2_shape[0], img_T2_shape[1], img_T2_shape[2], 3))
    
    # make ADC & DCE images the same shape as the T2 image
    ADC_resized = resize(img_ADC.get_fdata(), (img_T2_shape[0], img_T2_shape[1], img_T2_shape[2]), anti_aliasing=True)
    DCE_resized = resize(img_DCE.get_fdata(), (img_T2_shape[0], img_T2_shape[1], img_T2_shape[2]), anti_aliasing=True)

    combined_data[:,:,:, 0] = img_T2.get_fdata()[:,:,:]
    combined_data[:,:,:, 1] = ADC_resized[:,:,:]
    combined_data[:,:,:, 2] = DCE_resized[:,:,:]

    #combined_data[:,:,:, 0] = img_T2.get_fdata()[:,:,3:]
    #combined_data[:,:,:, 1] = ADC_resized[:,:,3:]
    #combined_data[:,:,:, 2] = DCE_resized[:,:,3:]

    combined_nifti = nib.Nifti1Image(combined_data, img_T2.affine, nib.Nifti1Header())
    
    nib.save(combined_nifti, os.path.join("data", f"{basename}.nii.gz"))






def preprocess_niftis():
    # This function resamples each image to 0.15x0.15x1mm and combines them into a 3 channel image.
    pass


def download_url_and_unpack(url):

    # Not needed anymore since downloading from github assets (actually results in an error)
    # if "TOTALSEG_DISABLE_HTTP1" in os.environ and os.environ["TOTALSEG_DISABLE_HTTP1"]:
    #     print("Disabling HTTP/1.0")
    # else:
    #     import http.client
    #     # helps to solve incomplete read errors
    #     # https://stackoverflow.com/questions/37816596/restrict-request-to-only-ask-for-http-1-0-to-prevent-chunking-error
    #     http.client.HTTPConnection._http_vsn = 10
    #     http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

    tempfile = os.path.join("data", "tmp_download_file.zip")

    try:

        with open(tempfile, 'wb') as f:
            # session = requests.Session()  # making it slower

            with requests.get(url, stream=True) as r:
                r.raise_for_status()

                # With progress bar
                total_size = int(r.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
                progress_bar.close()

        print("Download finished. Extracting...")
        # call(['unzip', '-o', '-d', network_training_output_dir, tempfile])
        with zipfile.ZipFile(os.join("data", "tmp_download_file.zip"), 'r') as zip_f:
            zip_f.extractall("data")
        # print(f"  downloaded in {time.time()-st:.2f}s")
    except Exception as e:
        raise e
    finally:
        if tempfile.exists():
            os.remove(tempfile)