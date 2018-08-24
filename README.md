# _Identifying Animal Species in Camera Trap Images using Deep Learning and Citizen Science_

This repo provides the research code used in this study. We strongly recommend to use the following repo
https://github.com/marco-willi/camera-trap-classifier to train new models. The only purpose of this repo is to provide reproducability of our results and to apply the models trained in the study.

# Pre-Requisites

The following pre-requisites are necessary to run the code.

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
