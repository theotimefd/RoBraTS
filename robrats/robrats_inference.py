import os
from utils import preprocess_niftis, make_3_channel_NIFTI
from monai.networks.nets import SegResNet, DynUNet
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch, Dataset, DataLoader
import torch
import nibabel as nib
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    EnsureChannelFirst,
    EnsureType,
    MapTransform,
    Orientation,
    SpatialCrop,
    NormalizeIntensity
)

root_dir = "/home/fehrdelt/data_ssd/data/segmentation_seb_glioblastome/final_dataset"

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            
            result.append(d[key]==2)
            
            result.append(d[key]==3)
            
            #result.append(torch.logical_or(d[key] == 4, d[key] == 5))
            result.append(d[key] == 4)

            result.append(d[key] == 5)
            d[key] = torch.stack(result, axis=0).float()
        return d

VAL_AMP = True
device = torch.device("cuda:1") #TODO changer ca pour que ca marche sur tout

model = DynUNet(spatial_dims=3,
                in_channels=3,
                out_channels=4,
                kernel_size=[3, 3, 3, 3, 3, 3],
                strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                norm_name="instance",
                deep_supervision=False,
                res_block=True,

).to(device)

infer_transform = Compose(
    [
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        #Orientation(axcodes="RAS"),
        #Spacingd(
        #    keys=["image", "label"],
        #    pixdim=(1.0, 1.0, 1.0),
        #    mode=("bilinear", "nearest"),
        #),
        #SpatialCrop(roi_center=[128, 128, 8], roi_size=[224, 224, 16]),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ]
)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# define inference method
def inference_model(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            #roi_size=(240, 240, 160),
            roi_size=(224, 224, 16),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)



def inference(t2_filepath, adc_filepath, perm_filepath, output_folder):

    make_3_channel_NIFTI("3_channel_temp", t2_filepath, adc_filepath, perm_filepath)
    test_images = [os.path.join("data", "3_channel_temp.nii.gz")]


    test_ds = Dataset(data=test_images, transform=infer_transform)
    test_loader = DataLoader(test_ds, batch_size=1 ,shuffle=True, num_workers=4)

    
    model.load_state_dict(torch.load(os.path.join("data", "best_metric_model.pth")))
    model.eval()
    
    with torch.no_grad():
        

        for test_data in test_loader:
            
            test_outputs = inference_model(test_data.to(device))
        
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]

            affine = nib.load(test_images[0]).affine


            tumor_core = nib.Nifti1Image(test_outputs[0][1,:,:,:].cpu(), affine, nib.Nifti1Header())
            nib.save(tumor_core, os.path.join(output_folder, "tumor_core.nii.gz"))

            peritumor = nib.Nifti1Image(test_outputs[0][2,:,:,:].cpu(), affine, nib.Nifti1Header())
            nib.save(peritumor, os.path.join(output_folder, "peritumor.nii.gz"))

            edema = nib.Nifti1Image(test_outputs[0][3,:,:,:].cpu(), affine, nib.Nifti1Header())
            nib.save(edema, os.path.join(output_folder, "edema.nii.gz"))

            healthy = nib.Nifti1Image(test_outputs[0][0,:,:,:].cpu(), affine, nib.Nifti1Header())
            nib.save(healthy, os.path.join(output_folder, "healthy.nii.gz"))


    
    return test_outputs

