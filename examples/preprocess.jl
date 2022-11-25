using myANN

n_resolution = 20


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
