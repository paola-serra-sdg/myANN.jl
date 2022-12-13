using Flux
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Statistics: mean, std
using Images
using Flux.Data: DataLoader
using myANN
using Plots
using Random
using ParametricMachinesDemos

# Get our data
images, labels = get_data("preprocessed_data");

# Train and test splitting
x_train, y_train, x_test, y_test = split_train_test(images, labels);

# Loading
train_data = DataLoader((x_train, y_train); batchsize = 32, shuffle = true);
test_data = DataLoader((x_test, y_test); batchsize = 32, shuffle = true);

dimensions = [300,200,200,250];

# Define the parametric machine
machine = DenseMachine(dimensions, sigmoid);

model = Flux.Chain(flatten, machine, Dense(sum(dimensions), 3), softmax) |> f64;


ps = Flux.params(model)
opt = ADAM(0.1)
loss(x,y) = logitcrossentropy(model(x), y) 

# check that learning happens correctly
epochs = Int64[]
loss_train = Float64[]
loss_test = Float64[]
acc = Float64[]

for i in 1:10
    gs = gradient(ps) do
        loss(x_train, y_train)
    end
    Flux.Optimise.update!(opt, ps, gs)
    if i % 1 == 0
        # @show loss(x_train, y_train)
        # @show loss(x_test, y_test)
        push!(epochs, i)
        push!(loss_train, loss(x_train, y_train))
        push!(loss_test, loss(x_test, y_test))
        push!(acc, accuracy(y_test, model(x_test)))
    end
end

plot(epochs, loss_train, lab="Training", c=:black, lw=2, ylims = (0,2));
plot!(epochs, loss_test, lab="Test", c=:green, lw=2, ylims = (0,2));
title!("Dense parametric machine");
yaxis!("Loss", :log);
xaxis!("Training epoch");
savefig("dense_PM.png");

plot(epochs, acc, lab="Training", c=:black, lw=2, ylims = (0,1));
title!("Dense architecture");
yaxis!("Accuracy", :log);
xaxis!("Training epoch");
savefig("densePM_accuracy.png");
