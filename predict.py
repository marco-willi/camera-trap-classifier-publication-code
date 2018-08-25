#! /usr/bin/env python
""" Classify images using a trained model

    Parameters
    ------------
    images_dir:
        - path to root directory that must contain 1 or more directories with
          images

    path_to_model:
        - path to model file
        - string

    model_cfg_json:
        - path to json with model config
        - Json-string with keys: "class_mapper" & "pre_processing"

    export_dir
        - path to directory to which to write the predictions

    export_fname:
        - name of output file  - a csv file (default: predictions.csv)

    batch_size:
        - number of images to process at a time (default 256)


    Usage example:

    python predict.py \
    -images_dir /my_data/new_images/ \
    -path_to_model /my_data/save/ss/ss_species_51_201708072308.hdf5 \
    -model_cfg_json /my_data/save/ss/ss_species_51_201708072308_cfg.json \
    -export_dir /my_data/save/ss/
"""
import argparse

from tools.classifier import CamTrapClassifier

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-images_dir", type=str, required=True,
        help='top level path (directory) that must contain 1 to N directories \
              in which to search for images to classify.')
    parser.add_argument(
        "-path_to_model", type=str, required=True,
        help='path to the model file (.hdf5)')
    parser.add_argument(
        "-model_cfg_json", type=str, required=True,
        help="path to the model configuration json file")
    parser.add_argument(
        "-export_fname", type=str, default="predictions.csv",
        help='filename to which to store the predictions (.csv)')
    parser.add_argument(
        "-export_dir", type=str, required=True,
        help='path (directory) in which to store the predictions csv')
    parser.add_argument(
        "-batch_size", type=int, default=256,
        help='number of images to process at a time (default 256)')
    args = vars(parser.parse_args())

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s, Value:%s" % (k, v))

    classifier = CamTrapClassifier(
                    path_to_model=args['path_to_model'],
                    model_cfg_json=args['model_cfg_json'])

    classifier.predict_path(
        path=args['images_dir'],
        output_path=args['export_dir'],
        output_file_name=args['export_fname'],
        batch_size=args['batch_size'])
