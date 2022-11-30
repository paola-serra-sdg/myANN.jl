using myANN

n_resolution = 30

process_images("data/bee", n_resolution, n_resolution);
process_images("data/wasp", n_resolution, n_resolution);
process_images("data/other_insect", n_resolution, n_resolution);

bee_dir = readdir("preprocessed_data/bee");
wasp_dir = readdir("preprocessed_data/wasp");
other_dir = readdir("preprocessed_data/other_insect");
