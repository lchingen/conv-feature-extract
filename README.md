# Convolutional Feature Extractor (Autoencoder Model)

<p align='justify'>
This repository is intended to serve as a reference to a front-end convolutional
auto-encoder implementation. Specifically, it contains the complete pipeline for
training, inference testing and weights extraction functions.
</p>

<p align='justify'>
This repository also includes extraction functions that can be used to pickle
the trained weights, biases, input and output activations for validation
purposes.
</p>


## How to Use

```console

# Clone repository
>> git clone https://github.com/lchingen/conv-feature-extract
>> cd conv-feature-extract

# Download dataset (TFRecords)
>> https://tinyurl.com/y5sqxybu
>> mkdir db
>> cp celeb-face/* ./db

# Train feature extractor
>> mkdir logs
>> python3 auto-encoder/train.py

# Extract and dump trained weight, bias, ia, oa
>> mkdir dump
>> python3 auto-encoder/extract-weight.py
>> python3 auto-encoder/extract-input.py

```

## References & Notes

* To generate TFRecords from scratch, please reference the following repository:

```console    

# Clone repository
>> git clone https://github.com/lchingen/tf-data-pipe
>> rm db/*

# Download dataset
>> wget http://www.kaggle.com/jessicali9530/celeba-dataset/kernels ./db

```

* Source description:
    - config.py: training configuration
    - model_fn.py: DNN model
    - train.py: train the DNN model
    - utils.py: utility functions
    - test-img.py: test single image inference on trained network
    - test-webcam.py: test webcam streaming and DNN inference results
    - extract-weight.py: extract W and B based on restored DNN model
    - extract-input.py: extract normalized IA and W after inference
