import nibabel as nib
import numpy as np
from skimage.transform import resize
import os


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
    combined_data = np.zeros((img_T2_shape[0], img_T2_shape[1], 16, 3))
    
    # make ADC & DCE images the same shape as the T2 image
    ADC_resized = resize(img_ADC.get_fdata(), (img_T2_shape[0], img_T2_shape[1], img_T2_shape[2]), anti_aliasing=True)
    DCE_resized = resize(img_DCE.get_fdata(), (img_T2_shape[0], img_T2_shape[1], img_T2_shape[2]), anti_aliasing=True)

    combined_data[:,:,:, 0] = img_T2.get_fdata()[:,:,3:]
    combined_data[:,:,:, 1] = ADC_resized[:,:,3:]
    combined_data[:,:,:, 2] = DCE_resized[:,:,3:]

    combined_nifti = nib.Nifti1Image(combined_data, img_T2.affine, nib.Nifti1Header())
    
    nib.save(combined_nifti, os.path.join("data", f"{basename}.nii.gz"))






def preprocess_niftis():
    # This function resamples each image to 0.15x0.15x1mm and combines them into a 3 channel image.
    pass