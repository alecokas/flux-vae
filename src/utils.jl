module Utils

using BSON: @save, @load
using DrWatson: struct2dict
using Flux: cpu, gpu
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

function save_arguments(save_dir::String, args)
    let args = struct2dict(args)
        @save joinpath(save_dir, "args.bson") args
    end
end

function load_arguments(load_dir::String)
    println("Created empty args from $load_dir")
    @load joinpath(load_dir, "args.bson") args
    args = CmdLineArgs(; args...)
    return args
end

export save_arguments, load_arguments, CmdLineArgs

end
