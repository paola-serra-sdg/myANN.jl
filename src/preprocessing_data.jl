using MLDatasets
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using DataFrames
using Images
using MLDataUtils
using IterTools
using Flux.Data: DataLoader

function resize_and_grayify(directory, im_name, width::Int64, height::Int64)
    resized_gray_img = Gray.(load(directory * "/" * im_name)) |> (x -> imresize(x, width, height))
    try
        save("preprocessed_" * directory * "/" * im_name, resized_gray_img)
    catch e
        if isa(e, SystemError)
            mkdir("preprocessed_" * directory)
            save("preprocessed_" * directory * "/" * im_name, resized_gray_img)
        end
    end
end

function process_images(directory, width::Int64, height::Int64)
    files_list = readdir(directory)
    map(x -> resize_and_grayify(directory, x, width, height), files_list)
end

n_resolution = 30


process_images("data/bee1", n_resolution, n_resolution);
process_images("data/bee2", n_resolution, n_resolution);
process_images("data/wasp1", n_resolution, n_resolution);
process_images("data/wasp2", n_resolution, n_resolution);

bee1_dir = readdir("preprocessed_data/bee1");
bee2_dir = readdir("preprocessed_data/bee2");
wasp1_dir = readdir("preprocessed_data/wasp1");
wasp2_dir = readdir("preprocessed_data/wasp2");

bees1 = load.("preprocessed_data/bee1/" .* bee1_dir);
bees2 = load.("preprocessed_data/bee2/" .* bee2_dir);
wasp1 = load.("preprocessed_data/wasp1/" .* wasp1_dir);
wasp2 = load.("preprocessed_data/wasp2/" .* wasp2_dir);


