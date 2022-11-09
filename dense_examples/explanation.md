# Dense architecture for ANN with MNIST

Model architecture is based on different type of layer:
- **Dropout layer** : For each input, either sets that input to 0 (with probability p) or scales it by 1/(1-p). This is used as a regularisation, i.e. it reduces overfitting during training.
- **Dense layer** : *Dense(in::Integer, out::Integer, σ = identity)*. Expression: y = σ.(W * x .+ b). 
- **Flatten**: makes a flatten vector.

