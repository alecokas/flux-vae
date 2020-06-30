# include("vae.jl")
# import .VAE
# include("utils.jl")
# import .Utils
# include("visualise.jl")
# import .VIZ

using BSON: @save
using Flux
using Flux: logitbinarycrossentropy
using Flux: params, cpu, gpu
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
    use_gpu::Bool = false
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

function save_model(encoder_μ, encoder_logσ, decoder, save_dir::String, epoch::Int)
    print("Saving model...")
    let encoder_μ = cpu(encoder_μ), encoder_logσ = cpu(encoder_logσ), decoder = cpu(decoder)
        @save joinpath(save_dir, "model-$epoch.bson") encoder_μ encoder_logσ decoder
    end
    println("Done")
end

# function forward_pass(x, encoder_μ, encoder_logσ, decoder, device)
#     # Compress into latent space
#     # μ, logσ = encoder(x)
#     μ = encoder_μ(x)
#     logσ = encoder_logσ(x)

#     # Apply reparameterisation trick to sample latent
#     ϵ = randn(Float32, size(logσ))
#     z = μ + device(ϵ) .* exp.(logσ)
#     # Reconstruct from latent sample
#     # println("WE GET TO HERE")
#     x̂ = decoder(z)
#     # println("WE GET FURTHER")

#     return x̂, μ, logσ
# end

function vae_loss(encoder_μ, encoder_logσ, decoder, x, β::Float32, device)
    batch_size = size(x)[end]

    # Forward propagate through VAE
    # x̂, μ, logσ, = forward_pass(x, encoder_μ, encoder_logσ, decoder, device)
    x̂ = encoder_latent(x)
    println("Forward pass done")
    # Negative reconstruction loss Ε_q[logp_x_z]
    logp_x_z = -sum(logitbinarycrossentropy.(x̂, x)) / batch_size
    # println("logitbinarycrossentropy done")
    # KL(qᵩ(z|x)||p(z)) where p(z)=N(0,1) and qᵩ(z|x) models the encoder i.e. reverse KL
    # The @. macro makes sure that all operates are elementwise
    # kl_q_p = 0.5f0 * sum(@. (exp(2f0*logσ) + μ^2 - 2f0*logσ - 1f0)) / batch_size
    kl_q_p = 0

    # println("kl_q_p done")
    # println("=====")
    # println(logp_x_z)
    # println(kl_q_p)
    # println("=====")

    # We want to maximise the evidence lower bound (ELBO)
    elbo = logp_x_z - β .* kl_q_p
    return -elbo
end

# function create_decoder(channel_depth::Int32, kernel_width::Int32, hidden_dims::Int32, latent_dims::Int32, device)
#     decoder = Chain(
#         Dense(latent_dims, hidden_dims, relu),
#         Dense(hidden_dims, hidden_dims, relu),
#         Dense(hidden_dims, channel_depth * kernel_width * kernel_width, relu),
#         x -> reshape(x, (kernel_width, kernel_width, channel_depth, 16)),
#         ConvTranspose((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1),
#         ConvTranspose((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1),
#         ConvTranspose((kernel_width, kernel_width), channel_depth => 1; stride = 2, pad = 1)
#     ) |> device
#     return decoder
# end

# function create_latent_encoders(channel_depth::Int32, kernel_width::Int32, hidden_dims::Int32, latent_dims::Int32, device)
#     encoder_features = Chain(
#         Conv((kernel_width, kernel_width), 1 => channel_depth, relu; stride = 2, pad = 1),
#         Conv((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1),
#         Conv((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1),
#         flatten,
#         Dense(channel_depth * kernel_width * kernel_width, hidden_dims, relu),
#         Dense(hidden_dims, hidden_dims, relu)
#     )

#     encoder_μ = Chain(encoder_features, Dense(hidden_dims, latent_dims)) |> device
#     encoder_logσ = Chain(encoder_features, Dense(hidden_dims, latent_dims)) |> device

#     return encoder_μ, encoder_logσ
# end

args = CmdLineArgs()

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
encoder_latent(x) = encoder_μ(x) + randn(10, 10) * exp.(encoder_logσ(x))

decoder = Chain(
    Dense(args.latent_dims, args.hidden_dims, relu),
    Dense(args.hidden_dims, args.hidden_dims, relu),
    Dense(args.hidden_dims, args.channel_depth * args.kernel_width * args.kernel_width, relu),
    x -> reshape(x, (args.kernel_width, args.kernel_width, args.channel_depth, 16)),
    ConvTranspose((args.kernel_width, args.kernel_width), args.channel_depth => args.channel_depth, relu; stride = 2, pad = 1),
    ConvTranspose((args.kernel_width, args.kernel_width), args.channel_depth => args.channel_depth, relu; stride = 2, pad = 1),
    ConvTranspose((args.kernel_width, args.kernel_width), args.channel_depth => 1; stride = 2, pad = 1)
)

function train()
    args = CmdLineArgs()
    # Utils.save_arguments(args.save_dir, args)
    if !isdir(args.save_dir)
        mkdir(args.save_dir)
    end
    # device = Utils.get_device(args.use_gpu)
    device = cpu

    # Create loss logger
    io = open(joinpath(args.save_dir, "log.txt"), "w+")
    loss_logger = SimpleLogger(io)
    with_logger(loss_logger) do
        @info("Training loss")
    end

    # TODO: Change this so that we automatically detect input dims
    # channel_depth::Int32, kernel_width::Int32, hidden_dims::Int32, latent_dims::Int32, device
    # encoder_μ, encoder_logσ = create_latent_encoders(args.channel_depth, args.kernel_width, args.hidden_dims, args.latent_dims, device)
    # decoder = create_decoder(args.channel_depth, args.kernel_width, args.hidden_dims, args.latent_dims, device)

    trainable_params = Flux.params(encoder_μ, encoder_logσ, decoder)

    # Use the adam optimiser
    optimiser = ADAM(args.learning_rate, (0.9, 0.999))
    best_loss = 9f4

    dataloader = get_train_loader(args.batch_size, args.shuffle_data)

    for epoch_num = 1:args.num_epochs
        acc_epoch_loss = 0f0
        progress_tracker = Progress(length(dataloader), 1, "Training epoch $epoch_num: ")

        for (x_batch, y_batch) in dataloader
            # pullback function returns the result (loss) and a pullback operator (back)
            loss, back = Flux.pullback(trainable_params) do
                vae_loss(encoder_μ, encoder_logσ, decoder, x_batch |> device, args.β, device)
            end
            println("Finish pullback")
            # Feed the pullback 1 to obtain the gradients and update the model parameters
            gradients = back(1f0)
            println("Finished back")
            Flux.Optimise.update!(optimiser, trainable_params, gradients)
            println("Update done")

            acc_epoch_loss += loss

            next!(progress_tracker; showvalues=[(:loss, loss)])
        end
        avg_epoch_loss = acc_epoch_loss / length(dataloader)

        # Log loss and save best model
        with_logger(loss_logger) do
            @info("$avg_epoch_loss")
        end
        println(avg_epoch_loss)
        if avg_epoch_loss < best_loss
            save_model(encoder_μ, encoder_logσ, decoder, args.save_dir, epoch_num)
        end

    end
    println("Training complete!")
    return encoder, decoder, args, device
end

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(123)
    encoder, decoder, args, device = train()
    # VIZ.visualise(encoder, decoder, args, device)
end
