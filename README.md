# myANN

[![Build Status](https://github.com/paola-serra-sdg/myANN.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/paola-serra-sdg/myANN.jl/actions/workflows/CI.yml?query=branch%3Amaster)

# Brief examples on how to implement simple neural networks with Flux


## TODOs:
- [x] Download a new dataset (bees and wasps)
- [x] Do preprocessing (resize and grayify) and make data usable for our models
- [x] Work with convolutional and dense network
- [ ] Recurrent network (difficult)
- [ ] Work with modules (ask to Pietro) (how can I include another script in my main?)



**Preprocessing**

Firstly we have changed the channels of the images (greyscale images (only 1 channel)) and we have resized the images to a different resolution (fewer then before and the same for all).

Then we have stores the images in an array of dimension (length, width, n_channels, n_images) and a vector of labels. Then we have standardize data to have better performance in our model and split the data in train and test set.

The architectures that we have decided to use are:
- Dense neural network
- Convolutional neural network
- Recurrent neural network (?)

Every model is stored in a different file and data are loading in the same way for dense and convolutional neural network.



