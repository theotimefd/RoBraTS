import utils
import robrats_inference
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Segmentation of .",
                                     epilog="Théotime Fehr Delude & Sébastien Rigollet")

    parser.add_argument("-t2", metavar="filepath", dest="t2",
                        help="Path for NIFTI T2 MRI file.",
                        type=lambda p: Path(p).absolute(), required=True)
    
    parser.add_argument("-adc", metavar="filepath", dest="adc",
                        help="Path for NIFTI ADC MRI file.",
                        type=lambda p: Path(p).absolute(), required=True)
    
    parser.add_argument("-perm", metavar="filepath", dest="perm",
                        help="Path for NIFTI permeability MRI file.",
                        type=lambda p: Path(p).absolute(), required=True)
    
    parser.add_argument("-o", metavar="filepath", dest="output",
                        help="Output folder path.",
                        type=lambda p: Path(p).absolute(), required=True)


    args = parser.parse_args()

    # Download weights
    utils.download_url_and_unpack("https://github.com/theotimefd/RoBraTS/releases/download/"+"weights_v1.0.0.zip")

    robrats_inference.inference(t2_filepath=args.t2, adc_filepath=args.adc, perm_filepath=args.perm, output_folder=args.output)


if __name__ == "__main__":
    main()



