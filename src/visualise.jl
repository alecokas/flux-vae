include("vae.jl")
import .VAE
include("utils.jl")
import .Utils

using BSON: @load
using Flux
using Flux.Data: DataLoader
using ImageFiltering
using JLD: load
using MLDatasets: FashionMNIST
using Plots
using Random


function get_test_loader(batch_size, shuffle::Bool)
    # FashionMNIST test set is made up of 10k 28 by 28 greyscale images
    train_x, train_y = FashionMNIST.testdata(Float32)
    train_x = reshape(train_x, (28, 28, 1, :))
    train_x = parent(padarray(train_x, Fill(0, (2,2,0,0))))
    return DataLoader(train_x, train_y, batchsize=batch_size, shuffle=shuffle)
end

function visualise()
    # Load saved models and CLI settings
    args = Utils.load_arguments("results")
    println("ABOUT TO LOAD")
    encoder, decoder = VAE.load_model("results", 1)
    println("LOADED")

    # Load the data
    dataloader = get_test_loader(args.batch_size, args.shuffle_data)

    print("Plotting...")
    plt = scatter()
    for (x_batch, y_batch) in dataloader
        μ, logσ = encoder(x_batch)
        scatter!(μ[1,:], μ[2,:]; label="")
        break
    end
    savefig(plt, "results/scatter.png")
    println("Done")
end

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(123)
    visualise()
end