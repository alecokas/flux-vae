using Flux
using Flux: logitbinarycrossentropy
using Flux.Data: DataLoader
using ImageFiltering
using MLDatasets: MNIST
using Random


function get_train_loader(batch_size, shuffle::Bool)
    # MNIST is made up of 60k 28 by 28 greyscale images
    train_x, train_y = MNIST.traindata(Float32)
    train_x = reshape(train_x, (28 * 28, :))
    # train_x = parent(padarray(train_x, Fill(0, (2,2,0,0))))
    return DataLoader(train_x, train_y, batchsize=batch_size, shuffle=shuffle)
end

function vae_loss(encoder_μ, encoder_logσ, decoder, x)
    batch_size = size(x)[end]

    # Forward propagate through encoder
    μ = encoder_μ(x)
    logσ = encoder_logσ(x)
    # Apply reparameterisation trick to sample latent
    z = μ + randn(Float32, size(logσ)) .* exp.(logσ)
    # Reconstruct from latent sample
    x̂ = decoder(z)

    # Negative reconstruction loss Ε_q[logp_x_z]
    logp_x_z = -sum(logitbinarycrossentropy.(x̂, x)) / batch_size

    # KL(qᵩ(z|x)||p(z)) where p(z)=N(0,1) and qᵩ(z|x) models the encoder i.e. reverse KL
    # The @. macro makes sure that all operates are elementwise
    kl_q_p = 0.5f0 * sum(@. (exp(2f0*logσ) + μ^2 - 2f0*logσ - 1f0)) / batch_size
    # We want to maximise the evidence lower bound (ELBO)
    β = 1
    elbo = logp_x_z - β .* kl_q_p
    return -elbo
end

function train()
    # Define the encoder network
    encoder_features = Chain(
        Dense(28 * 28, 512, relu),
    )
    encoder_μ = Chain(encoder_features, Dense(512, 10))
    encoder_logσ = Chain(encoder_features, Dense(512, 10))

    # Define the decoder network
    decoder = Chain(
        Dense(10, 512, relu),
        Dense(512, 28 * 28),
    )

    trainable_params = Flux.params(encoder_μ, encoder_logσ, decoder)

    # Use the adam optimiser
    optimiser = ADAM(0.0001, (0.9, 0.999))
    batch_size = 16
    shuffle_data = true
    dataloader = get_train_loader(batch_size, shuffle_data)

    for epoch_num = 1:5
        println(epoch_num)
        for (x_batch, y_batch) in dataloader
            # pullback function returns the result (loss) and a pullback operator (back)
            loss, back = Flux.pullback(trainable_params) do
                vae_loss(encoder_μ, encoder_logσ, decoder, x_batch)
            end
            # Feed the pullback 1 to obtain the gradients and update the model parameters
            gradients = back(1f0)
            Flux.Optimise.update!(optimiser, trainable_params, gradients)
            println(loss)
        end

    end
    println("Training complete!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(123)
    train()
end
