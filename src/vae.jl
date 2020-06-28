module VAE

include("utils.jl")
import .Utils

using BSON: @save, @load
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
    Encoder(channel_depth::Int32, kernel_width::Int32, hidden_dims::Int32, latent_dims::Int32, device) = new(
        Conv((kernel_width, kernel_width), 1 => channel_depth, relu; stride = 2, pad = 1) |> device,
        Conv((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1) |> device,
        Conv((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1) |> device,
        flatten |> device,
        Dense(channel_depth * kernel_width * kernel_width, hidden_dims, relu) |> device,
        Dense(hidden_dims, hidden_dims, relu) |> device,
        Dense(hidden_dims, latent_dims) |> device, # Transform into bottleneck μ representation
        Dense(hidden_dims, latent_dims) |> device  # Transform into bottleneck logσ representation

    )
end

function (encoder::Encoder)(x)
    # Anonymous function to forward pass the encoder
    x = encoder.conv_3(encoder.conv_2(encoder.conv_1(x)))
    x = encoder.dense_2(encoder.dense_1(encoder.flatten_layer(x)))
    return encoder.μ_layer(x), encoder.logσ_layer(x)
end

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
    Decoder(channel_depth::Int32, kernel_width::Int32, hidden_dims::Int32, latent_dims::Int32, device) = new(
        channel_depth,
        kernel_width,
        Dense(latent_dims, hidden_dims, relu) |> device,
        Dense(hidden_dims, hidden_dims, relu) |> device,
        Dense(hidden_dims, channel_depth * kernel_width * kernel_width, relu) |> device,
        ConvTranspose((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1) |> device,
        ConvTranspose((kernel_width, kernel_width), channel_depth => channel_depth, relu; stride = 2, pad = 1) |> device,
        ConvTranspose((kernel_width, kernel_width), channel_depth => 1, σ; stride = 2, pad = 1) |> device
    )
end

function (decoder::Decoder)(z)
    # Anonymous function to forward pass the decoder
    batch_size = size(z)[end]
    z = decoder.dense_3(decoder.dense_2(decoder.dense_1(z)))
    # println("Line before")
    z = reshape(z, (decoder.kernel_width, decoder.kernel_width, decoder.channel_depth, batch_size))
    z = decoder.deconv_3(decoder.deconv_2(decoder.deconv_1(z)))
    # println("About to return")
    return z
end

function forward_pass(x, encoder, decoder, device)
    # Compress into latent space
    μ, logσ = encoder(x)
    # Apply reparameterisation trick to sample latent
    ϵ = randn(Float32, size(logσ))
    z = μ + device(ϵ) .* exp.(logσ)
    # Reconstruct from latent sample
    x̂ = decoder(z)

    return x̂, μ, logσ
end

function vae_loss(encoder::Encoder, decoder::Decoder, x, β::Float32, device)
    batch_size = size(x)[end]

    # Forward propagate through VAE
    x̂, μ, logσ, = forward_pass(x, encoder, decoder, device)
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

function save_model(encoder::Encoder, decoder::Decoder, save_dir::String, epoch::Int)
    print("Saving model...")
    let encoder = cpu(encoder), decoder = cpu(decoder)
        @save joinpath(save_dir, "model-$epoch.bson") encoder decoder
    end
    println("Done")
end

function load_model(load_dir::String, epoch::Int)
    print("Loading model...")
    @load joinpath(load_dir, "model-$epoch.bson") encoder decoder
    println("Done")
    return encoder, decoder
end

export vae_loss, Decoder, Encoder, save_model, load_model
end