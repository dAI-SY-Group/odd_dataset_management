# ODD-Driven Dataset Management for Traceable Evaluation of Automated-Driving Perception Models

This repository contains the reproduction package for the paper 'ODD-Driven
Dataset Management for Traceable Evaluation of Automated-Driving Perception
Models'.

In the folder `kitti_meta_parser` you can find the parsing script to map KITTI
Road images to KITTI Raw metadata to filter the speed of the vehicle.

In the submodule folder `odd_dataset_management_depth_to_normal` you can find
the implementation to generate the SNE images needed for the training of the
Road-Former model.

In the submodule folder `odd_dataset_management_usnet` you can find the
implementation of the USNet model used in the paper.

In the folder `road-former_configs` you can find the config files used to train
the Road-Former models used in the paper.
