import argparse
from pathlib import Path

from robrats.robrats_inference import inference


def main():
    parser = argparse.ArgumentParser(description="Segment Glioblastoma zones from multiparametric MRI.",
                                     epilog="Theotime Fehr Delude & Sebastien Rigollet.")

    parser.add_argument("-t2", metavar="filepath", dest="t2_input",
                        help="Nifti image of the T2 weighted MR Image.",
                        type=lambda p: Path(p).absolute(), required=True)
    parser.add_argument("-adc", metavar="filepath", dest="adc_input",
                        help="Nifti image of the ADC MR Image.",
                        type=lambda p: Path(p).absolute(), required=True)
    
    parser.add_argument("-perm", metavar="filepath", dest="perm_input",
                        help="Nifti image of the vascular permeability MR Image.",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks",
                        type=lambda p: Path(p).absolute(), required=True)

    args = parser.parse_args()

    inference(args.t2_input, args.adc_input, args.perm_input, args.output)


if __name__ == "__main__":
    main()
