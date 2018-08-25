# _Identifying Animal Species in Camera Trap Images using Deep Learning and Citizen Science_

This repo provides the research code used in this study. We strongly recommend to use the following repo
https://github.com/marco-willi/camera-trap-classifier to train new models. The only purpose of this repo is to provide reproducability of our results and to apply the models trained in the study.

# Pre-Requisites

The following pre-requisites are necessary to run the code.

## Python

We have used Python 3.5 to train the models. The necessary modules are defined in the requirements.txt and can be installed using pip:
```
pip install -r requirements.txt
```

## Directory Structure

Prepare the following directory structure. The example shown is for training a model for the snapshot wisconsin dataset.

```
root_directory/
  images/
    snapshot_wisconsin/
      all/
        deer/
          deer1.jpg
  db/
    snapshot_wisconsin/
      subject_set_used.json
  save/
    snapshot_wisconsin/
  models/
    snapshot_wisconsin/
  logs/
    snapshot_wisconsin
```

- images/: This is for storing the images in class-specific directories.
- db/: This is for storing the dataset meta-data, including for all experiments and model runs.
- save/: This is for storing final models, files needed for prediction and predictions on test/validation data.
- models/: This is for storing intermediate models and other files.
- logs/: This is for storing log files for each model run.


# Code Structure

### config
Configuration file: models, paths and credentials.

### learning
File to run the training process. Heavily relies on Keras. Also contains different models.

### tools
Different helper functions, e.g. fetching data, classes for projects, experiments and subjects.

# Examples

```
git clone https://github.com/marco-willi/camera-trap-classifier-publication-code.git
```

## Apply a Model

The following code shows an example of how to classify images using the snapshot serengeti species model:

```
python predict.py \
-path_to_model /my_data/save/ss/ss_species_51_201708072308.hdf5 \
-model_cfg_json /my_data/save/ss/ss_species_51_201708072308_cfg.json \
-export_dir /my_data/save/ss/
```

## Train a Model

1. Open the config/config.ini file and make the following adjustments:
- projects: choose for which dataset to train the model
- paths: define paths
- experiment id: define the experiment id to run for the chosen project by choosing it in the specific project section. For Example: 'experiment_id: ss_blank_vs_non_blank_small' for the snapshot serengeti project.

2. Create folders for the specific project in each of the paths as defined in the config.ini. For example: create the following directories if training a model for snapshot_wisconsin:
- ../db/snapshot_wisconsin
- ../images/snapshot_wisconsin
- ../save/snapshot_wisconsin
- ../model/snapshot_wisconsin
- ../logs/snapshot_wisconsin

3. Make sure the images are in the /images/<project_name>/all/ directory in class specific folders.

4. Make sure the meta-data are in the /db/<project_name>/ directory.

5. Run the code:
```
python main.py
```
