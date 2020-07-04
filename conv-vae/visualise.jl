using BSON: @load
using Flux
using Flux: chunk
using Flux.Data: DataLoader
using ImageFiltering
using Images
using ImageIO
using MLDatasets: FashionMNIST


function get_test_loader(batch_size, shuffle::Bool)
    # The FashionMNIST test set is made up of 10k 28 by 28 greyscale images
    test_x, test_y = FashionMNIST.testdata(Float32)
    test_x = reshape(test_x, (28, 28, 1, :))
    test_x = parent(padarray(test_x, Fill(0, (2,2,0,0))))
    return DataLoader(test_x, test_y, batchsize=batch_size, shuffle=shuffle)
end


function save_to_images(x_batch, save_dir::String, prefix::String, num_images::Int64)
    @assert num_images <= size(x_batch)[4]
    for i=1:num_images
        save(joinpath(save_dir, "$prefix-$i.png"), colorview(Gray, permutedims(x_batch[:,:,1,i], (2, 1))))
    end
end

function load_model(load_dir::String, epoch::Int)
    print("Loading model...")
    @load joinpath(load_dir, "model-$epoch.bson") encoder_μ encoder_logvar decoder
    println("Done")
    return encoder_μ, encoder_logvar, decoder
end

function visualise()
    encoder_μ, encoder_logvar, decoder = load_model("results", 1)
    batch_size = 16
    shuffle = true
    dataloader = get_test_loader(batch_size, shuffle)

    for (x_batch, y_batch) in dataloader
        save_to_images(x_batch, "results", "test-image", 4)
        # save("results/gray.png", colorview(Gray, x_batch[:,:,1,1]))
        # x = x_batch[:,:,1,1]
        # image = convert_to_image(x, 1)
        # print(image)
        # save("results/image.png", image)
        μ_batch = encoder_μ(x_batch)
        logvar_batch = encoder_logvar(x_batch)
        println(size(μ_batch))
        break
    end

end


if abspath(PROGRAM_FILE) == @__FILE__
    visualise()
end