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

We have firstly change the channels of the images (from tot to 1, then we have greyscale images) and then we resize them to a different resolution (fewer then before).

