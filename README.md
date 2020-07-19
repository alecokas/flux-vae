# flux-vae
This repository accompanies my blog post ["Convolutional VAE in Flux"](http://alecokas.github.io/julia/flux/vae/2020/07/22/convolutional-vae-in-flux.html) where we take a look at variational autoencoders and do a walk-through demo of a [Flux](https://fluxml.ai/) implementation of a convolutional VAE using the FashionMNIST benchmark dataset.

<div style="text-align:center"><img src="/images/vae.png" width="600"/>
</div><br/>

## Training
You can train the model directly from these scripts by running the following in your terminal:
```
julia --project=vaeenv conv-vae/main.jl
```
You should see a progress tracker and the training loss displayed in your terminal.

## Visualisation
Similarly, you can visualise reconstructed images from the test set by running:
```
julia --project=vaeenv conv-vae/visualise.jl
```
Some original test set samples (on the left) with the coresponding reconstructions (on the right):

<img src="/images/recon.png" alt="reconstruction-drawing" width="400"/>

## Dependencies
This code has been tested using Julia version 1.4.1. The package environment status is as follows:
```
  [fbb218c0] BSON v0.2.6
  [336ed68f] CSV v0.7.1
  [3895d2a7] CUDAapi v4.0.0
  [35d6a980] ColorSchemes v3.9.0
  [3a865a2d] CuArrays v2.2.0
  [a93c6f00] DataFrames v0.21.4
  [31c24e10] Distributions v0.23.2
  [ced4e74d] DistributionsAD v0.5.2
  [587475ba] Flux v0.10.4
  [6a3955dd] ImageFiltering v0.6.13
  [82e4d734] ImageIO v0.2.0
  [916415d5] Images v0.22.2
  [c8e1da08] IterTools v1.3.0
  [eb30cadb] MLDatasets v0.5.2
  [442fdcdd] Measures v0.3.1
  [a3a9e032] NIfTI v0.4.1
  [d96e819e] Parameters v0.12.1
  [91a5bcdd] Plots v1.3.5
  [92933f4c] ProgressMeter v1.3.1
  [e88e6eb3] Zygote v0.4.20
  [9a3f8284] Random 
  [10745b16] Statistics 
```
You can also have a look in the `vaeenv` directory where I have commited my `Project.toml` and `Manifest.toml` files for you to inspect the dependences.


To cite this work, please site the linked blog post:
<div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight">
<code>@article{kastanos20fluxvae,
  title   = "Convolutional VAE in Flux",
  author  = "Alexandros Kastanos",
  journal = "alecokas.github.io",
  year    = "2020",
  url     = "http://alecokas.github.io/julia/flux/vae/2020/07/22/convolutional-vae-in-flux.html"
}
</code></pre></div></div>
