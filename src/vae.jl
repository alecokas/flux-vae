module VAE

include("utils.jl")
import .Utils

using BSON: @save, @load
using Flux
using Flux: logitbinarycrossentropy, Conv, ConvTranspose, Dense, flatten
using Flux: params, cpu, gpu
using NNlib: relu, σ


struct Encoder
    # Encoder definition
    conv_1
    conv_2
    conv_3
    flatten_layer
    dense_1
    dense_2
    μ_layer
    logσ_layer
    Encoder(channel_depth::Int32, kernel_width::Int32, hidden_dims::Int32, latent_dims::Int32) = new(
        Conv((kernel_width, kernel_width), 1 => channel_depth, relu; stride = 2, pad = 1),
        Conv((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1),
        Conv((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1),
        flatten,
        Dense(channel_depth * kernel_width * kernel_width, hidden_dims, relu),
        Dense(hidden_dims, hidden_dims, relu),
        Dense(hidden_dims, latent_dims), # Transform into bottleneck μ representation
        Dense(hidden_dims, latent_dims)  # Transform into bottleneck logσ representation

    )

    encoder_μ = Chain(encoder_features, Dense(hidden_dims, latent_dims)) |> device
    encoder_logσ = Chain(encoder_features, Dense(hidden_dims, latent_dims)) |> device

struct Decoder
    # Decoder definition
    channel_depth
    kernel_width
    dense_1
    dense_2
    dense_3
    deconv_1
    deconv_2
    deconv_3
    Decoder(channel_depth::Int32, kernel_width::Int32, hidden_dims::Int32, latent_dims::Int32) = new(
        channel_depth,
        kernel_width,
        Dense(latent_dims, hidden_dims, relu),
        Dense(hidden_dims, hidden_dims, relu),
        Dense(hidden_dims, channel_depth * kernel_width * kernel_width, relu),
        ConvTranspose((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1),
        ConvTranspose((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1),
        ConvTranspose((kernel_width, kernel_width), channel_depth => 1, σ; stride = 2, pad = 1)
    )
end

function create_decoder(channel_depth::Int32, kernel_width::Int32, hidden_dims::Int32, latent_dims::Int32, device)
    decoder = Chain(
        Dense(latent_dims, hidden_dims, relu),
        Dense(hidden_dims, hidden_dims, relu),
        Dense(hidden_dims, channel_depth * kernel_width * kernel_width, relu),
        x -> reshape(x, (kernel_width, kernel_width, channel_depth, 16)),
        ConvTranspose((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1),
        ConvTranspose((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1),
        ConvTranspose((kernel_width, kernel_width), channel_depth => 1, σ; stride = 2, pad = 1)
    ) |> device
    return decoder
end

function forward_pass(x, encoder, decoder)
    # Compress into latent space
    # μ, logσ = encoder(x)
    μ = encoder_μ(x)
    logσ = encoder_logσ(x)

    # Apply reparameterisation trick to sample latent
    ϵ = randn(Float32, size(logσ))
    z = μ + ϵ .* exp.(logσ)
    # Reconstruct from latent sample
    x̂ = decoder(z)

    return x̂, μ, logσ
end

function vae_loss(encoder, decoder, x, β::Float32)
    batch_size = size(x)[end]

    # Forward propagate through VAE
    x̂, μ, logσ, = forward_pass(x, encoder, decoder)
    # println("Forward pass done")
    # Negative reconstruction loss Ε_q[logp_x_z]
    logp_x_z = -sum(logitbinarycrossentropy.(x̂, x)) / batch_size
    # println("logitbinarycrossentropy done")
    # KL(qᵩ(z|x)||p(z)) where p(z)=N(0,1) and qᵩ(z|x) models the encoder i.e. reverse KL
    # The @. macro makes sure that all operates are elementwise
    kl_q_p = 0.5f0 * sum(@. (exp(2f0*logσ) + μ^2 - 2f0*logσ - 1f0)) / batch_size
    # println("kl_q_p done")
    # println("=====")
    # println(logp_x_z)
    # println(kl_q_p)
    # println("=====")

    # We want to maximise the evidence lower bound (ELBO)
    elbo = logp_x_z - β .* kl_q_p
    return -elbo
end

function save_model(encoder_μ, encoder_logσ, decoder, save_dir::String, epoch::Int)
    print("Saving model...")
    let encoder_μ = cpu(encoder_μ), encoder_logσ = cpu(encoder_logσ), decoder = cpu(decoder)
        @save joinpath(save_dir, "model-$epoch.bson") encoder_μ encoder_logσ decoder
    end
    println("Done")
end

function load_model(load_dir::String, epoch::Int)
    print("Loading model...")
    @load joinpath(load_dir, "model-$epoch.bson") encoder_μ encoder_logσ decoder
    println("Done")
    return encoder_μ, encoder_logσ, decoder
end

export vae_loss, create_latent_encoders, create_decoder, save_model, load_model
end