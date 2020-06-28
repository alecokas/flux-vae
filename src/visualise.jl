module VIZ

include("vae.jl")
import .VAE
include("utils.jl")
import .Utils

using BSON: @load
using Flux
using Flux: chunk
using Flux.Data: DataLoader
using ImageFiltering
using Images
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

function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(sigmoid.(x |> cpu), y_size), 32, :)...), (2, 1)))
end

function construct(x, encoder, decoder, device)
    # Compress into latent space
    μ, logσ = encoder(x)
    # Apply reparameterisation trick to sample latent
    ϵ = randn(Float32, size(logσ))
    z = μ + device(ϵ) .* exp.(logσ)
    # Reconstruct from latent sample
    x̂ = decoder(z)

    return x̂, μ, logσ
end

function visualise(encoder, decoder, args, device)
    dataloader = get_test_loader(args.batch_size, args.shuffle_data)
    # device = Utils.get_device(args.use_gpu)

    for (sample_idx, (x_batch, y_batch)) in enumerate(dataloader)
        # Reconstruction
        println("About to forward pass")
        reconstruction, _, _ = construct(x_batch, encoder, decoder, device)
        println("Finished the forward pass")
        rec_image = convert_to_image(reconstruction, args.samples_per_image)
        reconstruct_image_path = joinpath(args.save_path, "reconstruct-$sample_idx.png")
        save(reconstruct_image_path, rec_image)
        @info "Image saved: $(reconstruct_image_path)"

        # Original
        original = convert_to_image(x_batch, args.samples_per_image)
        original_image_path = joinpath(args.save_path, "original-$sample_idx.png")
        save(original_image_path, original)
        @info "Image saved: $(original_image_path)"
        break
    end

end

export visualise
end

# function visualise()
#     # Load saved models and CLI settings
#     args = Utils.load_arguments("results")
#     println("ABOUT TO LOAD")
#     encoder, decoder = VAE.load_model("results", 1)
#     println("LOADED")

#     # Load the data
#     dataloader = get_test_loader(args.batch_size, args.shuffle_data)

#     print("Plotting...")
#     plt = scatter()
#     for (x_batch, y_batch) in dataloader
#         μ, logσ = encoder(x_batch)
#         scatter!(μ[1,:], μ[2,:]; label="")
#         break
#     end
#     savefig(plt, "results/scatter.png")
#     println("Done")
# end

# if abspath(PROGRAM_FILE) == @__FILE__
#     Random.seed!(123)
#     visualise()
# end