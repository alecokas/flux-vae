using BSON: @save
using Flux
using Flux: logitbinarycrossentropy
using Flux.Data: DataLoader
using ImageFiltering
using JLD: load
using Logging: SimpleLogger, with_logger
using MLDatasets: FashionMNIST
using ProgressMeter: Progress, next!
using Random
using Parameters: @with_kw


@with_kw struct CmdLineArgs
    num_epochs::Int32 = 1
    batch_size::Int32 = 16
    shuffle_data::Bool = true
    channel_depth::Int32 = 32
    kernel_width::Int32 = 4
    latent_dims::Int32 = 10
    hidden_dims::Int32 = 256
    learning_rate::Float32 = 0.0001
    β::Float32 = 1f1
    save_dir::String = "results"
    samples_per_image::Int32 = 1
end


function get_train_loader(batch_size, shuffle::Bool)
    # FashionMNIST is made up of 60k 28 by 28 greyscale images
    train_x, train_y = FashionMNIST.traindata(Float32)
    train_x = reshape(train_x, (28, 28, 1, :))
    train_x = parent(padarray(train_x, Fill(0, (2,2,0,0))))
    return DataLoader(train_x, train_y, batchsize=batch_size, shuffle=shuffle)
end

function forward_pass(x, encoder_μ, encoder_logσ, decoder)
    # Compress into latent space made up of μ and logσ
    μ = encoder_μ(x)
    logσ = encoder_logσ(x)
    # Apply reparameterisation trick to sample latent
    ϵ = randn(Float32, size(logσ))
    z = μ + ϵ .* exp.(logσ)
    # Reconstruct from latent sample
    x̂ = decoder(z)

    return x̂, μ, logσ
end

function vae_loss(encoder_μ, encoder_logσ, decoder, x, β)
    batch_size = size(x)[end]

    # Forward propagate through encoder
    μ = encoder_μ(x)
    logσ = encoder_logσ(x)
    # Apply reparameterisation trick to sample latent
    ϵ = randn(Float32, size(logσ))
    z = μ + ϵ .* exp.(logσ)
    # Reconstruct from latent sample
    x̂ = decoder(z)

    # Negative reconstruction loss Ε_q[logp_x_z]
    logp_x_z = -sum(logitbinarycrossentropy.(x̂, x)) / batch_size

    # KL(qᵩ(z|x)||p(z)) where p(z)=N(0,1) and qᵩ(z|x) models the encoder i.e. reverse KL
    # The @. macro makes sure that all operates are elementwise
    kl_q_p = 0.5f0 * sum(@. (exp(2f0*logσ) + μ^2 - 2f0*logσ - 1f0)) / batch_size
    # We want to maximise the evidence lower bound (ELBO)
    elbo = logp_x_z - β .* kl_q_p
    return -elbo
end


args = CmdLineArgs()

function train()
    args = CmdLineArgs()
    if !isdir(args.save_dir)
        mkdir(args.save_dir)
    end

    # Define the encoder network
    encoder_features = Chain(
        Conv((args.kernel_width, args.kernel_width), 1 => args.channel_depth, relu; stride = 2, pad = 1),
        Conv((args.kernel_width, args.kernel_width), args.channel_depth => args.channel_depth, relu; stride = 2, pad = 1),
        Conv((args.kernel_width, args.kernel_width), args.channel_depth => args.channel_depth, relu; stride = 2, pad = 1),
        flatten,
        Dense(args.channel_depth * args.kernel_width * args.kernel_width, args.hidden_dims, relu),
        Dense(args.hidden_dims, args.hidden_dims, relu)
    )
    encoder_μ = Chain(encoder_features, Dense(args.hidden_dims, args.latent_dims))
    encoder_logσ = Chain(encoder_features, Dense(args.hidden_dims, args.latent_dims))

    # Define the decoder network
    decoder = Chain(
        Dense(args.latent_dims, args.hidden_dims, relu),
        Dense(args.hidden_dims, args.hidden_dims, relu),
        Dense(args.hidden_dims, args.channel_depth * args.kernel_width * args.kernel_width, relu),
        x -> reshape(x, (args.kernel_width, args.kernel_width, args.channel_depth, 16)),
        ConvTranspose((args.kernel_width, args.kernel_width), args.channel_depth => args.channel_depth, relu; stride = 2, pad = 1),
        ConvTranspose((args.kernel_width, args.kernel_width), args.channel_depth => args.channel_depth, relu; stride = 2, pad = 1),
        ConvTranspose((args.kernel_width, args.kernel_width), args.channel_depth => 1; stride = 2, pad = 1)
    )

    trainable_params = Flux.params(encoder_μ, encoder_logσ, decoder)

    # Use the adam optimiser
    optimiser = ADAM(args.learning_rate, (0.9, 0.999))
    dataloader = get_train_loader(args.batch_size, args.shuffle_data)

    for epoch_num = 1:args.num_epochs
        acc_epoch_loss = 0f0
        progress_tracker = Progress(length(dataloader), 1, "Training epoch $epoch_num: ")

        for (x_batch, y_batch) in dataloader
            # pullback function returns the result (loss) and a pullback operator (back)
            loss, back = Flux.pullback(trainable_params) do
                vae_loss(encoder_μ, encoder_logσ, decoder, x_batch, args.β)
            end
            # Feed the pullback 1 to obtain the gradients and update the model parameters
            gradients = back(1f0)
            Flux.Optimise.update!(optimiser, trainable_params, gradients)

            acc_epoch_loss += loss

            next!(progress_tracker; showvalues=[(:loss, loss)])
        end
        avg_epoch_loss = acc_epoch_loss / length(dataloader)

        let encoder_μ = encoder_μ, encoder_logσ = encoder_logσ, decoder = decoder
            @save joinpath(args.save_dir, "model-$epoch_num.bson") encoder_μ encoder_logσ decoder
        end

    end
    println("Training complete!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(123)
    train()
end
