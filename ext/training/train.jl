# =============================================================================
# Training Loop
# =============================================================================

using Optimisers: Optimisers, AdamW, Momentum, OptimiserChain, WeightDecay
using Lux.Training: TrainState, single_train_step!, AutoReactant
using Reactant: Reactant
using JLD2: JLD2, jldsave
using Logging: Logging

"""
    TrainingResult

Result of training. Holds final model params, state, and best metric.
"""
mutable struct TrainingResult{P, S}
    ps::P                   # Lux parameters
    st::S                   # Lux state (BatchNorm running stats, etc.)
    epoch::Int
    iteration::Int          # optimizer iterations taken (all epochs)
    best_fpr::Float64
end

function _make_optimizer(cfg::TrainingConfig)
    lr = Float32(cfg.lr)
    wd = Float32(cfg.weight_decay)
    return if cfg.optimizer === :adamw
        AdamW(; eta = lr, lambda = wd)
    elseif cfg.optimizer === :sgd
        OptimiserChain(Momentum(lr, 0.9f0), WeightDecay(wd))
    else
        error("Unknown optimizer: $(cfg.optimizer). Use :adamw or :sgd.")
    end
end

"""
    train_descriptor(train_path, val_path=nothing;
                     cfg=TrainingConfig(), output_dir="checkpoints",
                     resume=nothing)

Train a BlobDescriptorNetwork on annotation data.

Uses Lux + Zygote for automatic differentiation with GPU acceleration
(falls back to CPU if no GPU is available).

Learning rate decays linearly from `cfg.lr` to 0 over the full training run.
Validation (FPR@95) runs every `cfg.eval_interval` iterations and at epoch boundaries.

# Arguments
- `train_path`: path to training annotations HDF5 (from `annotate_blobs`)
- `val_path`: optional path to validation annotations HDF5
- `cfg`: `TrainingConfig` with all hyperparameters
- `output_dir`: directory for checkpoints (created if needed)
- `resume`: path to checkpoint JLD2 to resume from (restores params, state, epoch,
  and iteration count; optimizer is restarted fresh)

# Checkpoints (JLD2)
Saved to `output_dir/`:
- `checkpoint_epoch_N.jld2` — every epoch
- `best_model.jld2` — lowest FPR@95 on validation

Each contains `:ps`, `:st`, `:opt_state`, `:cfg`, `:epoch`, `:iteration`, `:fpr`.

# Returns
`TrainingResult` with final parameters.
"""
function train_descriptor(
        train_path::AbstractString,
        val_path::Union{Nothing, AbstractString} = nothing;
        cfg::TrainingConfig = TrainingConfig(),
        output_dir::AbstractString = "checkpoints",
        resume::Union{Nothing, AbstractString} = nothing
    )
    mkpath(output_dir)
    rng = Random.MersenneTwister(42)

    # --- TensorBoard logger (writes tfevents to output_dir/tb/) ---
    tb_logger = TBLogger(joinpath(output_dir, "tb"); min_level = Logging.Info)
    @info "TensorBoard logdir: $(joinpath(output_dir, "tb"))"

    # --- Data ---
    @info "Loading training data..."
    train_data = load_training_data(train_path; min_track_length = cfg.min_track_length)

    val_data = if !isnothing(val_path)
        @info "Loading validation data..."
        load_training_data(val_path; min_track_length = cfg.min_track_length)
    else
        nothing
    end

    # --- Model + Optimizer ---
    model = BlobDescriptorNetwork(cfg)
    ps, st = Lux.setup(rng, model)
    opt = _make_optimizer(cfg)

    start_epoch = 1
    global_iteration = 0
    best_fpr = Inf

    # --- Resume from checkpoint ---
    if !isnothing(resume)
        ckpt = JLD2.load(resume)
        ps = ckpt["ps"]
        st = ckpt["st"]
        start_epoch = ckpt["epoch"] + 1
        global_iteration = haskey(ckpt, "iteration") ? ckpt["iteration"] : ckpt["step"]   # "step": checkpoints written before 2026-09-29
        best_fpr = get(ckpt, "fpr", Inf)
        @info "Resumed from $resume (epoch $(start_epoch - 1), iteration $global_iteration, optimizer restarted)"
    end

    # Move to XLA device (Reactant). AutoReactant() traces & compiles the
    # gradient graph on the first single_train_step! call — no separate
    # forward warm-up needed (see Discourse #136884 reply #2).
    dev = Lux.xla_device()
    ps = ps |> dev
    st = st |> dev

    tstate = TrainState(model, ps, st, opt)

    # --- Compute total iterations for LR schedule ---
    sampler_proto = BalancedBatchSampler(Random.MersenneTwister(0), train_data, cfg)
    iterations_per_epoch = length(sampler_proto)
    total_iterations = cfg.epochs * iterations_per_epoch

    @info "Model: $(Lux.parameterlength(model)) params, batch=$(batch_size(cfg)), " *
        "iterations/epoch=$iterations_per_epoch, total=$total_iterations, optimizer=$(cfg.optimizer), " *
        "device=$dev"
    flush(stderr)

    # --- Loss function for Lux.Training ---
    function loss_fn(model, ps, st, (x, labels))
        embeddings, st = model(x, ps, st)
        loss = supcon_loss_mat(embeddings, labels; τ = cfg.temperature)
        return loss, st, (;)
    end

    # --- Training ---
    for epoch in start_epoch:cfg.epochs
        epoch_loss = 0.0
        n_batches = 0

        for (batch_x, batch_labels) in BalancedBatchSampler(rng, train_data, cfg)
            global_iteration += 1

            # Linear LR decay: lr * (1 - iteration/total_iterations)
            lr = Float32(cfg.lr * max(0.0, 1.0 - global_iteration / total_iterations))
            Optimisers.adjust!(tstate, lr)

            # Move batch to device
            batch_x_dev = batch_x |> dev
            batch_labels_dev = batch_labels |> dev

            _, loss, _, tstate = single_train_step!(
                AutoReactant(), loss_fn, (batch_x_dev, batch_labels_dev), tstate;
                return_gradients = Val(false)
            )

            batch_loss = Float64(loss)
            epoch_loss += batch_loss
            n_batches += 1

            # --- Per-iteration eval & logging ---
            if cfg.eval_interval > 0 && mod(global_iteration, cfg.eval_interval) == 0
                fpr = _maybe_evaluate(model, tstate, val_data, cfg)
                @info "Iteration $global_iteration (epoch $epoch): loss=$(round(batch_loss; digits = 5)), " *
                    "lr=$(round(lr; digits = 6)), FPR@95=$(round(fpr; digits = 5))"
                flush(stderr)
                with_logger(tb_logger) do
                    @info "train" loss = batch_loss lr = lr iteration = global_iteration
                    @info "val"   fpr_at_95 = fpr  iteration = global_iteration
                end
                best_fpr = _maybe_save_best(
                    output_dir, model, tstate, cfg,
                    epoch, global_iteration, fpr, best_fpr
                )
            end
        end

        # --- End-of-epoch eval & checkpoint ---
        avg_loss = epoch_loss / max(n_batches, 1)
        fpr = _maybe_evaluate(model, tstate, val_data, cfg)
        lr = Float32(cfg.lr * max(0.0, 1.0 - global_iteration / total_iterations))

        @info "Epoch $epoch/$(cfg.epochs): avg_loss=$(round(avg_loss; digits = 5)), " *
            "lr=$(round(lr; digits = 6)), FPR@95=$(round(fpr; digits = 5))"
        flush(stderr)
        with_logger(tb_logger) do
            @info "epoch" avg_loss = avg_loss fpr_at_95 = fpr lr = lr iteration = global_iteration
        end

        ps_cpu = tstate.parameters |> Lux.cpu_device()
        st_cpu = tstate.states |> Lux.cpu_device()
        opt_cpu = tstate.optimizer_state |> Lux.cpu_device()

        _save_checkpoint(
            output_dir, "checkpoint_epoch_$(epoch).jld2",
            ps_cpu, st_cpu, opt_cpu, cfg, epoch, global_iteration, fpr
        )
        best_fpr = _maybe_save_best(
            output_dir, model, tstate, cfg,
            epoch, global_iteration, fpr, best_fpr
        )
    end

    ps_final = tstate.parameters |> Lux.cpu_device()
    st_final = tstate.states |> Lux.cpu_device()
    @info "Training complete. Best FPR@95=$(round(best_fpr; digits = 5))"
    return TrainingResult(ps_final, st_final, cfg.epochs, global_iteration, best_fpr)
end

# =============================================================================
# Training helpers
# =============================================================================

"""Evaluate FPR@95 on validation data (CPU). Returns NaN if no val data."""
function _maybe_evaluate(model, tstate, val_data, cfg::TrainingConfig; n_eval_batches::Int = 10)
    isnothing(val_data) && return NaN
    ps_cpu = tstate.parameters |> Lux.cpu_device()
    st_cpu = tstate.states |> Lux.cpu_device()
    st_eval = Lux.testmode(st_cpu)
    rng = Random.MersenneTwister(0)
    sampler = BalancedBatchSampler(rng, val_data, cfg)

    # Pool all embeddings and labels
    all_embeddings = Float32[]
    all_labels = Int32[]
    for (i, (batch_x, batch_labels)) in enumerate(sampler)
        i > n_eval_batches && break
        embeddings, _ = model(batch_x, ps_cpu, st_eval)
        append!(all_embeddings, vec(embeddings))
        append!(all_labels, batch_labels)
    end

    # Reshape and compute ONE threshold on pooled data
    D = cfg.descriptor_dim
    embeddings = reshape(all_embeddings, D, length(all_labels))

    # Single FPR@95 on entire validation set
    return fpr_at_95_recall(embeddings, all_labels)
end

"""Save best model if FPR improved. Returns updated best_fpr."""
function _maybe_save_best(
        output_dir, model, tstate, cfg,
        epoch, iteration, fpr, best_fpr
    )
    (isnan(fpr) || fpr >= best_fpr) && return best_fpr
    ps_cpu = tstate.parameters |> Lux.cpu_device()
    st_cpu = tstate.states |> Lux.cpu_device()
    opt_cpu = tstate.optimizer_state |> Lux.cpu_device()
    _save_checkpoint(
        output_dir, "best_model.jld2",
        ps_cpu, st_cpu, opt_cpu, cfg, epoch, iteration, fpr
    )
    @info "  New best model (FPR@95=$(round(fpr; digits = 5)))"
    return fpr
end

function _save_checkpoint(dir, filename, ps, st, opt_state, cfg, epoch, iteration, fpr)
    path = joinpath(dir, filename)
    return jldsave(path; ps, st, opt_state, cfg, epoch, iteration, fpr)
end
