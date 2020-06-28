include("vae.jl")
import .VAE
include("utils.jl")
import .Utils
include("visualise.jl")
import .VIZ

using Flux: ADAM
using Flux: params, pullback, cpu, gpu
using Flux.Optimise: update!
using Flux.Data: DataLoader
using ImageFiltering
using JLD: load
using Logging: SimpleLogger, with_logger
using MLDatasets: FashionMNIST
using ProgressMeter: Progress, next!
using Random


function get_train_loader(batch_size, shuffle::Bool)
    # FashionMNIST is made up of 60k 28 by 28 greyscale images
    train_x, train_y = FashionMNIST.traindata(Float32)
    train_x = reshape(train_x, (28, 28, 1, :))
    train_x = parent(padarray(train_x, Fill(0, (2,2,0,0))))
    return DataLoader(train_x, train_y, batchsize=batch_size, shuffle=shuffle)
end

function train()
    args = Utils.CmdLineArgs()
    Utils.save_arguments(args.save_dir, args)
    if !isdir(args.save_dir)
        mkdir(args.save_dir)
    end
    device = Utils.get_device(args.use_gpu)

    # Create loss logger
    io = open(joinpath(args.save_dir, "log.txt"), "w+")
    loss_logger = SimpleLogger(io)
    with_logger(loss_logger) do
        @info("Training loss")
    end

    # TODO: Change this so that we automatically detect input dims
    # channel_depth::Int32, kernel_width::Int32, hidden_dims::Int32, latent_dims::Int32, device
    encoder = VAE.Encoder(args.channel_depth, args.kernel_width, args.hidden_dims, args.latent_dims, device)
    decoder = VAE.Decoder(args.channel_depth, args.kernel_width, args.hidden_dims, args.latent_dims, device)
    trainable_params = params(
        encoder.conv_1, encoder.conv_2, encoder.conv_3, encoder.dense_1, encoder.dense_2,
        encoder.μ_layer, encoder.logσ_layer, decoder.dense_1, decoder.dense_2,
        decoder.dense_3, decoder.deconv_1 , decoder.deconv_2, decoder.deconv_3
    )

    # Use the adam optimiser
    optimiser = ADAM(args.learning_rate)
    best_loss = 9f4

    dataloader = get_train_loader(args.batch_size, args.shuffle_data)

    for epoch_num = 1:args.num_epochs
        acc_epoch_loss = 0f0
        progress_tracker = Progress(length(dataloader), 1, "Training epoch $epoch_num: ")

        for (x_batch, y_batch) in dataloader
            # pullback function returns the result (loss) and a pullback operator (back)
            loss, back = pullback(trainable_params) do
                VAE.vae_loss(encoder, decoder, x_batch |> device, args.β, device)
            end
            # println("Finish pullback")
            # Feed the pullback 1 to obtain the gradients and update the model parameters
            gradients = back(1f0)
            # println("Finished back")
            update!(optimiser, trainable_params, gradients)
            # println("Update done")

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
            VAE.save_model(encoder, decoder, args.save_dir, epoch_num)
        end

    end
    println("Training complete!")
    return encoder, decoder, args
end

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(123)
    encoder, decoder, args = train()
    VIZ.visualise(encoder, decoder, args)
end
