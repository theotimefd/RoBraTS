import os
from utils import preprocess_niftis, make_3_channel_NIFTI
from monai.networks.nets import SegResNet, DynUNet
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    EnsureChannelFirst,
    EnsureType,
    ConvertToMultiChannelBasedOnBratsClasses,
    Orientation,
    SpatialCrop,
    NormalizeIntensity
)

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
        ConvertToMultiChannelBasedOnBratsClasses(),
        Orientation(axcodes="RAS"),
        #Spacingd(
        #    keys=["image", "label"],
        #    pixdim=(1.0, 1.0, 1.0),
        #    mode=("bilinear", "nearest"),
        #),
        SpatialCrop(roi_center=[128, 128, 8], roi_size=[224, 224, 16]),
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

    #three_channel_nifti = preprocess_niftis(t2_filepath, adc_filepath, perm_filepath) #TODO: finir cette fonction

    make_3_channel_NIFTI("3_channel_temp", t2_filepath, adc_filepath, perm_filepath)

    loaded_nifti = infer_transform(os.path.join("data", "3_channel_temp.nii.gz"))

    model.load_state_dict(torch.load(os.path.join("data", "best_metric_model.pth")))
    model.eval()

    with torch.no_grad():
                    
        test_inputs = loaded_nifti.to(device)

        test_outputs = inference_model(test_inputs)
        
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]

    return test_outputs

