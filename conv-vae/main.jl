using BSON: @save
using CSV
using DataFrames: DataFrame
using Flux
using Flux: logitbinarycrossentropy
using Flux.Data: DataLoader
using ImageFiltering
using MLDatasets: FashionMNIST
using ProgressMeter: Progress, next!
using Random
using Zygote


function get_train_loader(batch_size, shuffle::Bool)
    # FashionMNIST is made up of 60k 28 by 28 greyscale images
    train_x, train_y = FashionMNIST.traindata(Float32)
    train_x = reshape(train_x, (28, 28, 1, :))
    train_x = parent(padarray(train_x, Fill(0, (2,2,0,0))))
    return DataLoader(train_x, train_y, batchsize=batch_size, shuffle=shuffle, partial=false)
end


function save_model(encoder_μ, encoder_logvar, decoder, save_dir::String, epoch::Int)
    print("Saving model...")
    let encoder_μ = cpu(encoder_μ), encoder_logvar = cpu(encoder_logvar), decoder = cpu(decoder)
        @save joinpath(save_dir, "model-$epoch.bson") encoder_μ encoder_logvar decoder
    end
    println("Done")
end


# function create_vae()
#     # Define the encoder and decoder networks
#     encoder_features = Chain(
#         Conv((4, 4), 1 => 32, relu; stride = 2, pad = 1),
#         Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
#         Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
#         Flux.flatten,
#         Dense(32 * 4 * 4, 256, relu),
#         Dense(256, 256, relu)
#     )
#     encoder_μ = Chain(encoder_features, Dense(256, 10))
#     encoder_logvar = Chain(encoder_features, Dense(256, 10))

#     decoder = Chain(
#         Dense(10, 256, relu),
#         Dense(256, 256, relu),
#         Dense(256, 32 * 4 * 4, relu),
#         x -> reshape(x, (4, 4, 32, :)),
#         ConvTranspose((4, 4), 32 => 32, relu; stride = 2, pad = 1),
#         ConvTranspose((4, 4), 32 => 32, relu; stride = 2, pad = 1),
#         ConvTranspose((4, 4), 32 => 1; stride = 2, pad = 1)
#     )
#     return encoder_μ, encoder_logvar, decoder
# end


function vae_loss(encoder_μ, encoder_logvar, decoder, x, β)
    batch_size = size(x)[end]
    @assert batch_size != 0

    # Forward propagate through mean encoder and std encoders
    μ = encoder_μ(x)
    logvar = encoder_logvar(x)
    # Apply reparameterisation trick to sample latent
    z = μ + randn(Float32, size(logvar)) .* exp.(0.5f0 * logvar)
    # Reconstruct from latent sample
    x̂ = decoder(z)
    # Negative reconstruction loss Ε_q[logp_x_z]
    logp_x_z = -sum(logitbinarycrossentropy.(x̂, x)) / batch_size
    # KL(qᵩ(z|x)||p(z)) where p(z)=N(0,1) and qᵩ(z|x) models the encoder i.e. reverse KL
    # The @. macro makes sure that all operates are elementwise
    kl_q_p = 0.5f0 * sum(@. (exp(logvar) + μ^2 - logvar - 1f0)) / batch_size
    # We want to maximise the evidence lower bound (ELBO)
    # println(batch_size)
    # println(x[:,15,1,1])
    println(μ[:,1])
    println(logvar[:,1])
    println(x̂[:,15,1,1])
    println(logp_x_z)
    println(kl_q_p)
    elbo = logp_x_z - β .* kl_q_p
    # println(-elbo)
    println("====")
    return -elbo
end


function train( dataloader, num_epochs, β, optimiser, save_dir)
    encoder_features = Chain(
        Conv((4, 4), 1 => 32, relu; stride = 2, pad = 1, init=Flux.glorot_normal),
        Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1, init=Flux.glorot_normal),
        Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1, init=Flux.glorot_normal),
        Flux.flatten,
        Dense(32 * 4 * 4, 256, relu),
        Dense(256, 256, relu)
    )
    encoder_μ = Chain(encoder_features, Dense(256, 10))
    encoder_logvar = Chain(encoder_features, Dense(256, 10))

    decoder = Chain(
        Dense(10, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 32 * 4 * 4, relu),
        x -> reshape(x, (4, 4, 32, :)),
        ConvTranspose((4, 4), 32 => 32, relu; stride = 2, pad = 1, init=Flux.glorot_normal),
        ConvTranspose((4, 4), 32 => 32, relu; stride = 2, pad = 1, init=Flux.glorot_normal),
        ConvTranspose((4, 4), 32 => 1; stride = 2, pad = 1, init=Flux.glorot_normal)
    )

    # The training loop for the model
    trainable_params = Flux.params(encoder_μ, encoder_logvar, decoder)

    for epoch_num = 1:num_epochs
        acc_loss = 0.0
        progress_tracker = Progress(length(dataloader), 1, "Training epoch $epoch_num: ")
        for (x_batch, y_batch) in dataloader
            # pullback function returns the result (loss) and a pullback operator (back)
            loss, back = pullback(trainable_params) do
                vae_loss(encoder_μ, encoder_logvar, decoder, x_batch, β)
            end
            # Feed the pullback 1 to obtain the gradients and update then model parameters
            gradients = back(1f0)
            Flux.Optimise.update!(optimiser, trainable_params, gradients)
            if isnan(loss)
                break
            end
            acc_loss += loss
            next!(progress_tracker; showvalues=[(:loss, loss)])
        end
        @assert length(dataloader) > 0
        avg_loss = acc_loss / length(dataloader)
        println("avg_loss: $avg_loss")
        println("avg_loss: $avg_loss")
        println("length(dataloader): $(length(dataloader))")
        metrics = DataFrame(epoch=epoch_num, negative_elbo=avg_loss)
        println(metrics)
        CSV.write(joinpath(save_dir, "metrics.csv"), metrics, header=(epoch_num==1), append=true)
        save_model(encoder_μ, encoder_logvar, decoder, save_dir, epoch_num)
    end
    println("Training complete!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    batch_size = 16
    shuffle_data = true
    η = 0.001
    β = 1f0
    num_epochs = 4
    save_dir = "results"

    dataloader = get_train_loader(batch_size, shuffle_data)
    # encoder_μ, encoder_logvar, decoder = create_vae()

    train(dataloader, num_epochs, β, ADAM(η), save_dir)
end

