# ZeroCode_VirtualMultiplexing

## Overview

This repository contains the code as well as an online tool generated for Zero Code Virtual Multiplexing. 

It contains: 

* ZeroCode-VirtualMultiplexing on Google Colab. 
* Code to generate training and testing data from CZI format image dataset. 
* Deep Learning training and test examples. 

ZeroCode-VirtualMultiplexing is a tool for generate and use deep learning models for signal unmixing in fluorescent microscopy imaging. It is set in a user friendly interface so anyone can use it, for that it is implemented in [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). 

In the tool, for generating data from a CZI file is used the [DataGenerator.py](https://github.com/akabago/ZeroCostDL4Mic-VirtualMultiplexing/blob/main/Tools/DataGenerator.py), and the Deep Learning approach used for virtual multiplexing is based on [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

<img src="https://github.com/akabago/ZeroCostDL4Mic-VirtualMultiplexing/blob/main/Images/Data_workflow.jpg" width="650" height="400">

## How to use ZeroCode-VM notebook?

ZeroCode-VM notebook can be directly opened from GitHub into Colab by clicking on the link. It is mandatory for use it to create a local copy to your Google Drive. Once you have a copy of it follow the intructions present on the notebook. 
It is necesary to use this tool to have a Google Drive account for having a copy of the notebook as well as for uploading the files that you are going to use. 

Access here to the tool **ZeroCostDL4Mic-VirtualMultiplexing**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akabago/ZeroCostDL4Mic-VirtualMultiplexing/blob/main/ZC_VirtualMultiplexing.ipynb)

### User general workflow

Once you have copied the notebook in your Google Drive account, you can follow the general pipeline implemented shown below:

<img src ="https://github.com/akabago/ZeroCostDL4Mic-VirtualMultiplexing/blob/main/Images/User_workflow.jpg" width="650" height="400">

### What steps do you need to follow in the notebook?

If you don't know which steps to follow in the noebook, answer the questions present on the decission tree below to see which steps you need to follow. 

<img src ="https://github.com/akabago/ZeroCostDL4Mic-VirtualMultiplexing/blob/main/Images/User_steps.jpg" width="<550" height="650">

### Want to see a short demonstration?

[Click here](https://youtu.be/7jm2LcwZ-ps) to access to a short video demonstration of the tool. 

## Models

There were trained 3 different pix2pix models for virtual multiplexing with the same hyperparameter configuration and the same CZI file. They differ on the approach followed for generate the mixed signal data from the CZI file. 

### Model 1

In this model the mixed signal data comes from the third channel of the CZI file, which is an open detector channel that captures signals of channel number 1 (CDH1) and number 2 (KI67). 

* [Data generated to train the model](https://drive.google.com/drive/folders/1VEqE5jROh1UNL1dB3qM4potkI_a9uUs4?usp=sharing). 
* [Versions of the trained model and test results](https://drive.google.com/drive/folders/1PCXJfkxZPZEYHgfJfvdU5odojpx7KAl_?usp=sharing).

### Model 2

In this model the mixed signal comes from a synthetic approach in which the mixed signal is generated computationally by the aqual proportional combination between the signal of channel 1 (CDH1) and channel 2 (KI67). 

* [Data generated to train the model](https://drive.google.com/drive/folders/1PCXJfkxZPZEYHgfJfvdU5odojpx7KAl_?usp=sharing).
* [Versions of the trained model and test results](https://drive.google.com/drive/folders/1ziPplvnKBGzIQrLKoMXAEb-bqRYj6GWF?usp=sharing).

### Model 3

In this model the mixed signal comes from a weighted approach in which the mixed signal is generated computationally by the combination between the signal of channel 1 (CDH1), with a contribution of 40% and channel 2 (KI67), with a contribution of 60%. 

* [Data generated to train the model](https://drive.google.com/drive/folders/1CFPFkWV-5G1a8ioV-3Iud9YSRfyATUvD?usp=sharing).
* [Versions of the trained model and test results](https://drive.google.com/drive/folders/1MOidHNvIyCH6KySGfsI5xV0sfrWQc75L?usp=sharing).

## Predictions on real data

The trained models have been used for virtual multiplexing in a real dataset which contains mixed signal of CDH1 and KI67 markers. 

* [Data generated from the CZI for predictions](https://drive.google.com/drive/folders/1U8T5IgQB7DCYQbXamvGtSySrq_zRPePD?usp=sharing).
* [Virtual mutliplexing results using open detector model](https://drive.google.com/drive/folders/1-tjP8_GcefnkSGzoodZ3IXS4Q5Y6LLYp?usp=sharing). 
* [Virtual multiplexing results using synthetic model](https://drive.google.com/drive/folders/1LX2B9Tv8GXDpWxZOlpidj9TBp-fH5pcU?usp=sharing).
* [Virtual multiplexing results using weighted blended model](https://drive.google.com/drive/folders/1-6-26qfm-wNU6b_q2JQvuGiTTtJSe_nJ?usp=sharing).

## Remarks

The datasets, models and output generated are not incluided in this repository since they too large for GitHub, to access them they are added as links to Google Drive. 

## Acknowledgements
Work produced with the support of a 2023 Leonardo Grant for Scientific Research and Cultural Creation, BBVA Foundation






