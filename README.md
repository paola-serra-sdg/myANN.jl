# myANN

[![Build Status](https://github.com/paola-serra-sdg/myANN.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/paola-serra-sdg/myANN.jl/actions/workflows/CI.yml?query=branch%3Amaster)

# Brief examples on how to implement simple neural networks using Flux


**TODOs:**

- [x] Download a new dataset (bees and wasps)
- [x] Do preprocessing (resize and grayify) and make data usable for our models
- [x] Work with convolutional and dense network
- [x] Plot train loss function and accuracy
- [x] Change datatypes in some more general (ex: string -> abstractstring)
- [x] Make get_data more usable in general (changing the repository parameters)
- [x] Change accuracy to work with more than 2 labels
- [x] Tests for accuracy (and something else?)



**Preprocessing**

Firstly we have changed the channels of the images (greyscale images (only 1 channel)) and we have resized the images to a different resolution (fewer than before and the same for all).

Then we have stored the images in an array of dimension (length, width, n_channels, n_images) and a vector of labels. Then we have standardize data to have better performance in our model and split the data in train and test set.

The architectures that we have decided to use are:
- Dense neural network
- Convolutional neural network

We have also implemented a recurrent neural network using a sinusoidal function as an example.

**Visualization**

After training, we have plotted the loss function (with train and test data) and the accuracy (that we have implemented [here](./src/metrics.jl))



