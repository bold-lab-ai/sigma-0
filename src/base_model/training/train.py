import os
import sys
import time
import json
import jax
from jax import random
from jax.experimental.multihost_utils import sync_global_devices
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
import wandb
import gc

from base_model.training.state import init_train_state, load_checkpoint, save_checkpoint, deduplicate_trainstate, remap_train_state_step
from base_model.data.dataloading import create_lobster_prediction_dataset, create_lobster_train_loader
from base_model.data.lobster_dataloader import LOBSTER_Dataset
from base_model.training.train_helpers import reduce_lr_on_plateau, train_epoch, validate, \
    create_jit_train_step, create_jit_eval_step, create_learning_rate_schedule, \
    StepWatchdog, TIME_START_I, TIME_END_I, LR_MIN_FRACTION, DiLoCoState
from base_model.encoding.encoding import Message_Tokenizer
from base_model.evaluation.message_perplexity import message_perplexity
from base_model.sharding.sharding_utils import initialize_mesh, create_state_shardings




MAMBA3_ARCHITECTURES = frozenset({'mamba3', 'hybrid_mamba3'})


def pallas_needs_shard_map(args):
    """Why MAMBA3_SSD_IMPL=pallas must not run under the single-node jit.

    Returns the refusal text, or None when the run is fine. A Pallas custom
    call has no sharding rule, so the plain single-node jit replicates it and
    every GPU runs the whole batch (2.8x slower than the XLA reference, job
    6393829). SINGLE_NODE_SHARD_MAP=1 with --hierarchical runs the shard_map
    step instead, one shard per GPU. Multi-node jobs already take that path;
    one GPU, tensor parallelism and the other architectures are not affected.
    """
    if jax.process_count() > 1 or getattr(args, 'num_devices', 1) <= 1:
        return None
    if getattr(args, 'tp_size', 1) > 1:
        return None
    from base_model.models.registry import architecture_from_args
    if architecture_from_args(args) not in MAMBA3_ARCHITECTURES:
        return None
    from base_model.models.mamba3_pallas import ssd_impl_mode
    if ssd_impl_mode() != 'pallas':
        return None
    if getattr(args, 'hierarchical', False) and os.environ.get('SINGLE_NODE_SHARD_MAP', '') == '1':
        return None
    return ("MAMBA3_SSD_IMPL=pallas on one node needs SINGLE_NODE_SHARD_MAP=1 and "
            "HIERARCHICAL=True. Under the single-node jit the kernels have no sharding "
            "rule and run the whole batch on every GPU, 2.8x slower than "
            "MAMBA3_SSD_IMPL=xla.")


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0
    best_test_last_order_loss = 100000000
    best_test_last_order_acc = -10000.0
    best_test_last_order_nll = 100000000
    best_test_all_orders_nll = 100000000

    #for parameter sweep: get args from wandb server
    is_main_process = getattr(args, 'process_index', 0) == 0
    if args is None:
        args = wandb.config
    else:
        if args.USE_WANDB and is_main_process:
            # Rank 0: online sync to wandb cloud
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
            wandb_name = os.environ.get("WANDB_NAME") or (f"j{slurm_job_id}" if slurm_job_id else None)
            wandb_run_id = os.environ.get("WANDB_RUN_ID") or None
            wandb_resume = os.environ.get("WANDB_RESUME") if wandb_run_id else None
            run = wandb.init(
                project=args.wandb_project,
                job_type='model_training',
                config=vars(args),
                entity=args.wandb_entity,
                name=wandb_name,
                id=wandb_run_id,
                resume=wandb_resume,
            )
        elif args.USE_WANDB:
            # Non-rank-0: local logging only, no duplicate cloud runs
            run = wandb.init(mode='offline')
        else:
            run = wandb.init(mode='offline')

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
    wandb.log({"block_size": block_size})

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    ds = 'lobster-prediction'
    #create_dataset_fn =  Datasets[ds]

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    mask_fn=None
    if args.masking == 'causal':
        mask_fn = LOBSTER_Dataset.causal_mask
    elif args.masking == 'random':
        mask_fn = LOBSTER_Dataset.random_mask
    elif args.masking == 'last_pos':
         mask_fn = LOBSTER_Dataset.last_pos_mask
    elif args.masking == 'none':
         mask_fn = LOBSTER_Dataset.no_mask
    else:
        ValueError('Issue with mask function: logic for '+args.masking+' not implemented.')

    is_distributed = getattr(args, 'is_distributed', False)
    process_rank = getattr(args, 'process_index', 0)
    process_count = getattr(args, 'process_count', 1)
    grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
    refusal = pallas_needs_shard_map(args)
    if refusal:
        raise SystemExit(refusal)

    (lobster_dataset, trainloader, _, _, _,
        n_classes, seq_len, in_dim, book_seq_len, book_dim, train_size) = \
        create_lobster_prediction_dataset(
            args.dir_name,
            seed=args.jax_seed,
            mask_fn=mask_fn,
            msg_seq_len=args.msg_seq_len,
            micro_bsz=args.micro_bsz,
            num_devices=args.num_devices,
            tp_size=getattr(args, 'tp_size', 1),
            use_book_data=args.use_book_data,
            use_simple_book=args.use_simple_book,
            book_transform=args.book_transform,
            book_ablation=getattr(args, 'book_ablation', 'real'),
            n_data_workers=args.n_data_workers,
            prefetch_factor=getattr(args, 'prefetch_factor', 2),
            shuffle_train=args.shuffle_train,
            rand_offset=args.random_offsets_train,
            debug_overfit=args.debug_overfit,
            val_split=float(getattr(args, 'val_split', 0.001) or 0.0),
            test_split=0.0,
            test_dir_name=getattr(args, 'test_dir_name', None),
            use_distributed_sampler=is_distributed,
            process_rank=process_rank,
            process_count=process_count,
            uniform_ticker_batches=getattr(args, 'uniform_ticker_batches', False),
            max_train_batches=(getattr(args, 'max_train_batches', 0) or None),
            tickers=getattr(args, 'tickers', None),
            data_root=getattr(args, 'data_root', None),
            train_date_range=getattr(args, 'train_date_range', None),
            # Production training is deliberately train-only (test_split=0).
            # Do not discover or touch the held-out test period during fitting;
            # fixed validation and LOBBench run as separate checkpoint jobs.
            test_date_range=None,
            token_mode=getattr(args, 'token_mode', '26tok'),
        )

    print(f"[*] Starting S5 Training on {ds} =>> Initializing...")
    if args.debug_loading:
        state=None
        val_model=None
        init_hidden=None
    else:
        state, model_cls = init_train_state(
            args,
            n_classes=n_classes,
            seq_len=seq_len,
            book_dim=book_dim,
            book_seq_len=book_seq_len,
            train_size=train_size,
            print_shapes=True
        )

        # Initialize mesh first (needed for restore and JIT step functions)
        # Multi-node: global mesh over ALL devices for cross-node gradient sync
        # Single-node: local mesh over num_devices GPUs
        use_hierarchical = getattr(args, 'hierarchical', False)
        tp_size = getattr(args, 'tp_size', 1)
        if jax.process_count() > 1:
            mesh = initialize_mesh(jax.device_count(),
                                   hierarchical=use_hierarchical,
                                   tp_size=tp_size)
        else:
            # SINGLE_NODE_SHARD_MAP=1 keeps --hierarchical on one node: a
            # (1, N) ('nodes', 'gpus') mesh and the shard_map training step that
            # multi-node jobs run. Unset, the historical single-node path (1D
            # 'data' mesh under jit) is unchanged. Custom kernels (Pallas) have
            # no sharding rule under jit and get replicated on every device;
            # inside shard_map each device runs its own shard.
            single_node_shard_map = os.environ.get('SINGLE_NODE_SHARD_MAP', '') == '1'
            if use_hierarchical and tp_size <= 1 and not single_node_shard_map:
                print("[Sharding] Single-node: forcing hierarchical=False (1D mesh); "
                      "SINGLE_NODE_SHARD_MAP=1 selects the (1, N) shard_map path")
                use_hierarchical = False
            mesh = initialize_mesh(args.num_devices, hierarchical=use_hierarchical,
                                   tp_size=tp_size,
                                   allow_single_node_hierarchical=single_node_shard_map)

        # DiLoCoState wrapping must match the saved checkpoint structure:
        #   - DiLoCo-era ckpts (post-988fb8b)  → wrap BEFORE restore
        #   - Pre-DiLoCo ckpts                 → wrap AFTER restore (fresh outer_momentum)
        # Probe the saved _METADATA to choose the right order.
        diloco_enabled = getattr(args, 'diloco_outer', 'none') == 'nesterov'
        restore_active = (args.restore is not None and args.restore != '')
        ckpt_has_outer = False
        if diloco_enabled and restore_active:
            try:
                _restore_step = args.restore_step
                if _restore_step is None:
                    _step_dirs = [d for d in os.listdir(args.restore) if d.isdigit()]
                    _restore_step = max(int(d) for d in _step_dirs) if _step_dirs else None
                if _restore_step is not None:
                    _md_path = os.path.join(args.restore, str(_restore_step),
                                             'state', '_METADATA')
                    if os.path.exists(_md_path):
                        with open(_md_path) as _f:
                            _md = json.load(_f)
                        ckpt_has_outer = any(
                            'outer_momentum' in k
                            for k in _md.get('tree_metadata', {}).keys())
                        print(f"[DiLoCo] Probed ckpt at step {_restore_step}: "
                              f"outer_momentum "
                              f"{'present' if ckpt_has_outer else 'absent'}")
            except Exception as _e:  # noqa: BLE001
                print(f"[DiLoCo] Could not probe ckpt structure ({_e}); "
                      "assuming pre-DiLoCo")

        if diloco_enabled and ckpt_has_outer:
            _outer_m = jax.tree.map(jnp.zeros_like, state.params)
            state = DiLoCoState(train_state=state, outer_momentum=_outer_m)
            print(f"[DiLoCo] Enabled Nesterov outer optimizer "
                  f"(lr={getattr(args, 'diloco_outer_lr', 0.7)}, "
                  f"β={getattr(args, 'diloco_outer_momentum', 0.9)}). "
                  f"Wrapped BEFORE restore (matches DiLoCo-era ckpt).")
        elif diloco_enabled and not restore_active:
            # Fresh DiLoCo run, no restore — wrap now with zeros
            _outer_m = jax.tree.map(jnp.zeros_like, state.params)
            state = DiLoCoState(train_state=state, outer_momentum=_outer_m)
            print(f"[DiLoCo] Enabled Nesterov outer optimizer (fresh start). "
                  f"Wrapped state in DiLoCoState with zero outer_momentum.")

        restored_metrics = {}
        if restore_active:
            print(f"[*] Restoring weights from {args.restore}")
            ckpt = load_checkpoint(
                state,
                args.restore,
                # args.__dict__,
                step=args.restore_step,
                mesh=mesh,
                partial_restore=getattr(args, 'partial_restore', False),
            )
            state = ckpt['model']
            # Debug: verify restored state
            print(f"[Restore] state.step = {int(state.step)}")
            print(f"[Restore] Restored metrics: {ckpt.get('metrics', {})}")
            # Check optimizer momentum is non-zero (proves Adam state restored)
            # Handle both plain multi_transform and chain(clip, multi_transform) structures
            _opt = state.opt_state
            if isinstance(_opt, tuple):
                _opt = _opt[-1]  # unwrap chain → last element is MultiTransformState
            ssm_inner = _opt.inner_states['ssm'].inner_state
            adam_state = ssm_inner[0]  # ScaleByAdamState (optax schedule mode)
            mu_leaves = jax.tree_util.tree_leaves(adam_state.mu)
            nu_leaves = jax.tree_util.tree_leaves(adam_state.nu)
            mu_norms = [float(jnp.linalg.norm(m)) for m in mu_leaves[:3]]
            nu_norms = [float(jnp.linalg.norm(n)) for n in nu_leaves[:3]]
            print(f"[Restore] Adam mu norms (first 3 params): {mu_norms}")
            print(f"[Restore] Adam nu norms (first 3 params): {nu_norms}")
            schedule_count = ssm_inner[1].count  # ScaleByScheduleState
            print(f"[Restore] Schedule count = {int(schedule_count)}")
            restored_metrics = ckpt.get('metrics', {})

            # --- Elastic Resume: remap step when device count or grad_accum changes ---
            ckpt_config = ckpt.get('config', {})
            original_process_count = ckpt_config.get('process_count', process_count)
            original_grad_accum = ckpt_config.get('grad_accum_steps', 1)
            if original_process_count != process_count or original_grad_accum != grad_accum_steps:
                # Compute original optimizer_steps_per_epoch (respecting curtail if checkpoint used it)
                original_curtail = ckpt_config.get('curtail_epochs', None)
                raw_original_micro_spe = train_size // (args.micro_bsz * args.num_devices * original_process_count)
                original_micro_spe = min(raw_original_micro_spe, original_curtail + 1) if original_curtail is not None else raw_original_micro_spe
                original_spe = original_micro_spe // max(original_grad_accum, 1)

                # Compute new optimizer_steps_per_epoch (respecting current curtail setting)
                raw_new_micro_spe = train_size // (args.micro_bsz * args.num_devices * process_count)
                new_micro_spe = min(raw_new_micro_spe, args.curtail_epochs + 1) if args.curtail_epochs is not None else raw_new_micro_spe
                new_spe = new_micro_spe // max(grad_accum_steps, 1)

                restored_epoch = int(state.step) // max(original_spe, 1)
                step_within_epoch = int(state.step) % max(original_spe, 1)
                scaled_step_within = round(step_within_epoch * new_spe / original_spe) if original_spe > 0 else 0
                remapped_step = restored_epoch * new_spe + scaled_step_within
                print(f"[Elastic Resume] process_count changed: {original_process_count} → {process_count}")
                print(f"[Elastic Resume] grad_accum changed: {original_grad_accum} → {grad_accum_steps}")
                print(f"[Elastic Resume] optimizer_steps/epoch: {original_spe} → {new_spe}")
                print(f"[Elastic Resume] intra-epoch: {step_within_epoch}/{original_spe} → {scaled_step_within}/{new_spe} ({step_within_epoch/max(original_spe,1)*100:.1f}%)")
                print(f"[Elastic Resume] state.step {int(state.step)} → {remapped_step} (epoch {restored_epoch})")
                state = remap_train_state_step(state, remapped_step)

            # --- Curriculum / Fine-tune: reset step to 0 so LR schedule starts fresh ---
            if getattr(args, 'restore_reset_schedule', False):
                print(f"[Restore Reset] state.step {int(state.step)} → 0 (fresh LR schedule)")
                state = remap_train_state_step(state, 0)

        # Pre-DiLoCo ckpt + DiLoCo enabled: wrap AFTER restore with fresh outer_momentum
        # (the saved state had no outer_momentum subtree, so wrapping before restore
        # would have produced a pytree-mismatch error).
        if diloco_enabled and not ckpt_has_outer and restore_active:
            _outer_m = jax.tree.map(jnp.zeros_like, state.params)
            state = DiLoCoState(train_state=state, outer_momentum=_outer_m)
            print(f"[DiLoCo] Enabled Nesterov outer optimizer "
                  f"(lr={getattr(args, 'diloco_outer_lr', 0.7)}, "
                  f"β={getattr(args, 'diloco_outer_momentum', 0.9)}). "
                  f"Wrapped AFTER restore with fresh zero outer_momentum "
                  f"(pre-DiLoCo ckpt).")

        val_model = model_cls(training=False, step_rescale=1)
        from base_model.models.registry import build_backbone

        backbone = build_backbone(args)
        carry_kwargs = dict(backbone.carry_kwargs)
        if backbone.is_attention:
            from base_model.models.registry import attention_config_for_width
            carry_kwargs['transformer_config_book'] = attention_config_for_width(
                backbone, book_dim)
        hidden_size = int(args.ssm_size_base)
        if getattr(args, 'architecture', None) == 's5' and args.conj_sym:
            hidden_size //= 2
        init_hidden = model_cls().initialize_carry(
            batch_size=args.micro_bsz,
            hidden_size=hidden_size,
            n_message_layers=args.n_message_layers,
            n_book_pre_layers=args.n_book_pre_layers,
            n_book_post_layers=args.n_book_post_layers,
            n_fused_layers=args.n_layers,
            h_size_ema=args.d_model,
            d_book=book_dim,
            **carry_kwargs)

        # Move state to host numpy (device-agnostic) then shard to global mesh.
        # This handles both init (jax array on local device) and restore (numpy from checkpoint).
        state = jax.device_get(state)
        # Fix: ensure all processes have identical state before device_put.
        # numpy.linalg.eigh in make_DPLR_HiPPO (ssm_init.py:69) can produce
        # bit-level differences across nodes (LAPACK non-determinism).
        # device_put uses strict assert_equal, so broadcast rank 0's state.
        if jax.process_count() > 1:
            from jax.experimental.multihost_utils import broadcast_one_to_all
            state = broadcast_one_to_all(state)
        state_shardings = create_state_shardings(state, mesh)
        state = jax.device_put(state, state_shardings)
        total_devices = jax.device_count() if jax.process_count() > 1 else args.num_devices
        print(f"[*] State distributed via sharding (replicated across {total_devices} devices)")

        local_steps_k = getattr(args, 'local_steps_k', 0)
        if local_steps_k > 0:
            assert use_hierarchical or tp_size > 1, \
                "Local Steps (--local_steps_k>0) requires --hierarchical=True or --tp_size>1"
            print(f"[*] Local Steps enabled: sync params every {local_steps_k} steps "
                  f"(inner optimizer unchanged, only intra-node grad sync per step)")

        jit_train_step = create_jit_train_step(
            mesh, state, has_book_data=args.use_book_data,
            hierarchical=use_hierarchical,
            batchnorm=args.batchnorm, ignore_times=args.ignore_times,
            local_steps_k=local_steps_k,
            grad_accum_steps=grad_accum_steps,
            diloco_outer=getattr(args, 'diloco_outer', 'none'),
            diloco_outer_lr=getattr(args, 'diloco_outer_lr', 0.7),
            diloco_outer_momentum=getattr(args, 'diloco_outer_momentum', 0.9))
        jit_eval_step = create_jit_eval_step(mesh, state, has_book_data=args.use_book_data)

    # Freeze the scientific parameter-match contract at runtime. Launch files
    # export EXPECTED_PARAMS; without this check a stale architecture option
    # could silently produce a mislabeled full experiment matrix.
    _num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    _expected_params_raw = os.environ.get("EXPECTED_PARAMS")
    if _expected_params_raw:
        try:
            _expected_params = int(_expected_params_raw)
        except ValueError as exc:
            raise RuntimeError(
                f"EXPECTED_PARAMS must be an integer, got {_expected_params_raw!r}"
            ) from exc
        if _num_params != _expected_params:
            raise RuntimeError(
                "parameter-count mismatch for "
                f"{getattr(args, 'architecture', getattr(args, 'ssm_type', 'unknown'))}: "
                f"initialized {_num_params:,}, expected {_expected_params:,}"
            )
        if is_main_process:
            print(f"[Protocol] Parameter-count match: {_num_params:,}")

    # ── The tokenizer that actually built these tensors ──
    # A token mode that never reaches Python does not raise: the dataset falls
    # back to 26 tokens, the model is built to match it, training runs, and the
    # checkpoint records "26tok" under a job that announced 24. That happened --
    # node_wrapper.sh passed --token_mode=26tok literally -- and the only thing
    # that gave it away was the tokens-per-step arithmetic. Assert instead.
    _requested_mode = str(getattr(args, "token_mode", "26tok"))
    _tokens_per_message = seq_len // int(args.msg_seq_len)
    if seq_len % int(args.msg_seq_len) != 0:
        raise ValueError(
            f"seq_len {seq_len} is not a whole number of messages of "
            f"{args.msg_seq_len}; the dataset and the model disagree about the layout")
    if f"{_tokens_per_message}tok" != _requested_mode:
        raise ValueError(
            f"token_mode={_requested_mode} was requested but the dataset produced "
            f"{_tokens_per_message} tokens per message ({seq_len} / {args.msg_seq_len}). "
            "The mode did not reach the tokenizer, so this run would train a "
            f"{_tokens_per_message}-token model and record it as {_requested_mode}.")
    if is_main_process:
        print(f"[token-mode] {_requested_mode} confirmed: {_tokens_per_message} "
              f"tokens/message, seq_len {seq_len}")

    # ── Compute FLOPs for throughput tracking ──
    try:
        from base_model.models.flops import compute_flops_per_step, print_flops_summary
        _tokens_per_step = args.micro_bsz * total_devices * seq_len
        # Mamba3 has a measured hardware-correction factor (~1.8-2.5x);
        # for other SSM types (gdn, s5, kda) leave correction at 1.0 to
        # avoid silently inflating MFU.
        _is_mamba3 = getattr(args, 'ssm_type', 'gdn') == 'mamba3'
        _flops_info = compute_flops_per_step(
            _num_params, _tokens_per_step,
            correction=None if _is_mamba3 else 1.0,
            d_model=getattr(args, 'd_model', 1024),
            seq_len=seq_len,
            chunk_size=getattr(args, 'mamba3_chunk_size', 64),
            d_state=getattr(args, 'mamba3_d_state', 128),
            headdim=getattr(args, 'mamba3_headdim', 64),
            expand=getattr(args, 'mamba3_expand', 2),
        )
        _flops_per_step = _flops_info["flops_per_step"]
        if is_main_process:
            print_flops_summary(_flops_info, num_gpus=total_devices)
            if args.USE_WANDB:
                wandb.config.update({
                    "flops_per_step": _flops_per_step,
                    "flops_correction": _flops_info["correction_factor"],
                    "num_params": _num_params,
                    "tokens_per_step": _tokens_per_step,
                }, allow_val_change=True)
    except Exception as e:
        print(f"[FLOPs] Warning: could not compute FLOPs: {e}")
        _flops_per_step = None

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = int(state.step)  # for per step learning rate decay (restored from checkpoint or 0)

    # Restore best metrics from checkpoint if available
    if restored_metrics:
        best_loss = restored_metrics.get('loss_val_ar', best_loss)
        best_acc = restored_metrics.get('acc_val_ar', best_acc)
        best_val_loss = restored_metrics.get('loss_val_ar', best_val_loss)
        best_test_loss = restored_metrics.get('loss_test_rnn', best_test_loss)
        best_test_acc = restored_metrics.get('acc_test_rnn', best_test_acc)
        print(f"[Restore] Best metrics restored: val_loss={best_loss:.5f}, val_acc={best_acc:.4f}, "
              f"test_loss={best_test_loss:.5f}, test_acc={best_test_acc:.4f}")
    micro_steps_per_epoch = int(train_size / (args.micro_bsz * args.num_devices * process_count)) if args.curtail_epochs is None else args.curtail_epochs+1
    # steps_per_epoch in optimizer updates (= micro_steps // K)
    steps_per_epoch = micro_steps_per_epoch // grad_accum_steps
    if grad_accum_steps > 1:
        print(f"[GradAccum] K={grad_accum_steps}: micro_steps/epoch={micro_steps_per_epoch}, "
              f"optimizer_steps/epoch={steps_per_epoch}, "
              f"effective_bsz={args.micro_bsz * args.num_devices * process_count * grad_accum_steps}")

    # ---- Token budget, in millions -----------------------------------------
    # Sample counts are not comparable across runs once sequence length moves.
    # A batch of 64 at 500 messages and a batch of 16 at 2,000 carry the same
    # 832k tokens, and reading the first as "four times larger" is the mistake
    # this line exists to prevent. Tokens are the unit the loss is averaged in
    # and the unit scaling laws are written in, so report those.
    from base_model.encoding.encoding import Message_Tokenizer as _MT   # local: keeps module import graph flat
    _tok_per_msg = _MT.MSG_LEN if getattr(args, 'token_mode', '26tok') != '1tok' else 1
    _tok_per_sample = args.msg_seq_len * _tok_per_msg
    _samples_micro = args.micro_bsz * args.num_devices * process_count
    _tok_micro = _samples_micro * _tok_per_sample
    _tok_step = _tok_micro * grad_accum_steps
    _planned_steps = int(os.environ.get('COSINE_STEPS', '0')) or steps_per_epoch * args.epochs
    print(f"[Tokens] per sample: {_tok_per_sample:,} ({args.msg_seq_len} msg x {_tok_per_msg} tok)")
    print(f"[Tokens] per micro-batch: {_tok_micro/1e6:.4f}M  ({_samples_micro} samples "
          f"= {args.micro_bsz}/gpu x {args.num_devices} gpu x {process_count} proc)")
    print(f"[Tokens] per optimizer step: {_tok_step/1e6:.4f}M  (x K={grad_accum_steps})")
    print(f"[Tokens] planned total: {_tok_step*_planned_steps/1e9:.3f}B over {_planned_steps:,} steps")

    # Scaling-law training jobs are train-only. Held-out loss is computed later
    # by a separate fixed-eval job, so no validation/test callback is wired into
    # the training loop.
    train_only_validate_every_n_steps = 0

    # Create LR schedule functions for wandb logging (optax manages LR inside JIT)
    total_steps = steps_per_epoch * args.epochs
    # COSINE_STEPS override: decouple cosine period from dataset size.
    # When training << 1 epoch, set COSINE_STEPS to your actual training budget
    # so the LR decays meaningfully (e.g. COSINE_STEPS=200000 for ~200k steps).
    cosine_steps_override = int(os.environ.get('COSINE_STEPS', '0'))
    if cosine_steps_override > 0:
        total_steps = cosine_steps_override
        print(f"[Schedule] COSINE_STEPS override: cosine period = {total_steps} "
              f"(dataset epoch = {steps_per_epoch * args.epochs})")
    warmup_end_step = int(steps_per_epoch * args.warmup_end)

    effective_lr_min = args.lr_min if args.lr_min > 0 else lr * LR_MIN_FRACTION
    effective_ssm_lr_min = args.lr_min if args.lr_min > 0 else ssm_lr * LR_MIN_FRACTION

    lr_schedule_fn = create_learning_rate_schedule(
        base_lr=lr, warmup_end_step=warmup_end_step,
        total_steps=total_steps, lr_min=effective_lr_min,
        use_cosine_anneal=args.cosine_anneal)
    ssm_lr_schedule_fn = create_learning_rate_schedule(
        base_lr=ssm_lr, warmup_end_step=warmup_end_step,
        total_steps=total_steps, lr_min=effective_ssm_lr_min,
        use_cosine_anneal=args.cosine_anneal)

    # print("USING VERY INFREQUENT CHECKPOINTING FOR TINY EPOCH SIZE ")

    # Global mesh: ALL ranks must create CheckpointManager so Orbax barriers work.
    # Use SLURM_JOB_ID for consistent path across ranks (wandb run names differ per rank).
    # Orbax primary_host=0 ensures only rank 0 writes; others just participate in barriers.
    slurm_jid = os.environ.get("SLURM_JOB_ID", "local")
    ckpt_base = os.environ.get('CHECKPOINT_BASE_DIR', 'checkpoints')
    explicit_ckpt_dir = os.environ.get('CHECKPOINT_RUN_DIR')
    if explicit_ckpt_dir:
        ckpt_dir = os.path.abspath(explicit_ckpt_dir)
    else:
        ckpt_dir = os.path.abspath(f'{ckpt_base}/{run.name}_{run.id}_{slurm_jid}/') if is_main_process else \
                   os.path.abspath(f'{ckpt_base}/job_{slurm_jid}/')
    if process_count > 1:
        # Multi-node: broadcast rank 0's checkpoint dir to all ranks
        if is_main_process:
            # Encode path as fixed-length byte array
            path_bytes = ckpt_dir.encode('utf-8')
            path_arr = jnp.array(list(path_bytes) + [0] * (256 - len(path_bytes)), dtype=jnp.uint8)
        else:
            path_arr = jnp.zeros(256, dtype=jnp.uint8)
        path_arr = jax.make_array_from_process_local_data(
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), path_arr
        )
        path_arr = jax.experimental.multihost_utils.broadcast_one_to_all(path_arr)
        path_bytes = bytes(jax.device_get(path_arr).tolist()).rstrip(b'\x00')
        ckpt_dir = path_bytes.decode('utf-8')
        print(f"[Rank {process_rank}] Checkpoint dir: {ckpt_dir}")

    mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        create=True,
        max_to_keep=10,
        keep_period=5,
        # Disable async: forked subprocesses corrupt NCCL after cuInit in child
        enable_async_checkpointing=False,
    )
    ckpt_mgr = ocp.CheckpointManager(
        ckpt_dir,
        item_names=('state', 'metadata'),
        options=mgr_options,
        metadata=vars(args) if is_main_process else {}
    )


    if args.ignore_times:
        # Removing the 5 abs time tokens from the length of the sequence.
        dt = [[x] for (x,) in zip([*range(seq_len-5*args.msg_seq_len)])]
    else:
        dt = [[x] for (x,) in zip([*range(seq_len)])]
    ce_table=wandb.Table(columns=["tok"] ,data=dt)

    ignore_times=args.ignore_times
    batchnorm=args.batchnorm

    start_epoch = 0
    resume_from_step_auto = None
    if args.restore is not None and args.restore != '':
        # Always infer from state.step (works for both epoch-end and mid-epoch checkpoints)
        start_epoch = int(state.step) // max(steps_per_epoch, 1)
        # Auto-compute resume_from_step for mid-epoch checkpoints
        step_within_epoch = int(state.step) % max(steps_per_epoch, 1)
        if step_within_epoch > 0:
            resume_from_step_auto = step_within_epoch
            print(f"[Restore] Resuming from epoch {start_epoch}, batch_idx={resume_from_step_auto} "
                  f"(state.step={int(state.step)}, steps_per_epoch={steps_per_epoch})")
        else:
            print(f"[Restore] Resuming training from epoch {start_epoch} (of {args.epochs})")

    # Mid-epoch checkpoint: callback + resume state
    job_start_time = time.monotonic()
    # CLI arg overrides auto-computed value
    resume_from_step = getattr(args, 'resume_from_step', None) or resume_from_step_auto

    # Reshard via host roundtrip — avoids creating a new NCCL clique that can
    # deadlock on CXI (Slingshot) when memory registrations go stale.
    # Cherry-picked from K3 Mamba2 (dbb1e445).
    if is_distributed:
        from jax.experimental.multihost_utils import broadcast_one_to_all as _bcast

        def _reshard_for_ckpt(s):
            # device_get may produce bit-level FP differences across hosts
            # (Muon Newton-Schulz, NCCL reduction non-determinism, etc.).
            # Broadcast rank-0's copy so device_put's assert_equal passes.
            host_s = jax.device_get(s)
            host_s = _bcast(host_s)
            return jax.device_put(host_s, state_shardings)

        # On resume: force a dummy checkpoint save while Orbax CXI endpoints
        # are still warm from the restore.  Without this, the first real save
        # (even at 5 min) can hit stale CXI registrations → SIGABRT (RC:265).
        # Cherry-picked from K3 Mamba2 (0721c624).
        if args.restore is not None and args.restore != '':
            _warm_ckpt = {
                'model': _reshard_for_ckpt(state),
                'config': vars(args) if is_main_process else {},
                'metrics': {'loss_train': 0.0, 'epoch': 0, 'step_in_epoch': 0}
            }
            save_checkpoint(ckpt_mgr, _warm_ckpt, int(state.step))
            del _warm_ckpt
            gc.collect()
            if is_main_process:
                print("[*] CXI warm: dummy checkpoint save completed (freed)")

    def step_checkpoint_callback(cb_state, cb_epoch, cb_batch_idx, cb_loss, save_flag=True, **kwargs):
        """Mid-epoch: log to wandb and optionally save checkpoint."""
        global_step = int(cb_state.step)
        if is_main_process and args.USE_WANDB:
            # The learning rate must be logged at the same cadence as the loss.
            # When diagnosing a divergence the first thing to rule out is "did
            # the schedule raise the LR over this stretch" -- and with LR logged
            # only at epoch end that exclusion cannot be made: two curves that
            # do not share sample points cannot be read against each other.
            log_dict = {
                "step_loss": float(cb_loss),
                "epoch": cb_epoch + 1,
                "step_in_epoch": cb_batch_idx + 1,
                "global_step": global_step,
                "lr": float(lr_schedule_fn(global_step)),
                "ssm_lr": float(ssm_lr_schedule_fn(global_step)),
            }
            # Add throughput metrics if available
            throughput = kwargs.get("throughput")
            if throughput:
                log_dict.update({
                    "throughput/step_time_s": throughput["step_time_s"],
                    "throughput/mfu_pct": throughput["mfu_pct"],
                    "throughput/tflops": throughput["achieved_tflops"],
                    "throughput/tokens_per_sec": throughput["tokens_per_sec"],
                })
            # Per-group gradient norms (LOG_GRAD_NORMS=1)
            grad_norms = kwargs.get("grad_norms")
            if grad_norms is not None:
                _gn_names = ['global', 'muon', 'ssm', 'regular', 'in_proj', 'out_proj']
                for _gn_name, _gn_val in zip(_gn_names, grad_norms):
                    log_dict[f"grad_norms/{_gn_name}"] = float(_gn_val)
                _max_gn = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
                if _max_gn > 0:
                    log_dict["grad_norms/clip_ratio"] = float(grad_norms[0]) / _max_gn
            # LR at step cadence (it was epoch-end only, so an LR-schedule
            # artefact could never be ruled out from the loss curve alone).
            log_dict["lr"] = float(lr_schedule_fn(global_step))
            log_dict["ssm_lr"] = float(ssm_lr_schedule_fn(global_step))
            # Per-position CE decomposition (LOG_CE_BY_POSITION=1): the loop
            # computes this vector every step and used to throw it away. The
            # near-zero positions are what dilute the flat 26-position mean.
            ce_by_pos = kwargs.get("ce_by_pos")
            if ce_by_pos is not None:
                _ce_vec = jax.device_get(ce_by_pos)
                _msgs = int(args.msg_seq_len)
                _tpm = int(_ce_vec.shape[0]) // _msgs
                _pos = _ce_vec.reshape(_msgs, _tpm).mean(axis=0)
                for _p in range(_tpm):
                    log_dict[f"ce_pos/p{_p:02d}"] = float(_pos[_p])
                _tok_lens = Message_Tokenizer.TOK_LENS
                if _tpm == int(sum(_tok_lens)):
                    _fields = list(enumerate(Message_Tokenizer.FIELDS))
                else:
                    # ignore_times: the two absolute-time fields are excluded
                    # from the loss, so their positions are absent here too.
                    _fields = [(i, f) for i, f in enumerate(Message_Tokenizer.FIELDS)
                               if i not in (6, 7)]
                _idx = 0
                _field_nats = {}
                for _fi, _fname in _fields:
                    _n = int(_tok_lens[_fi])
                    _mean = float(_pos[_idx:_idx + _n].mean())
                    log_dict[f"ce_field/{_fname}"] = _mean
                    # nats this field costs per message = per-token mean x width.
                    # Summed over fields this is the message NLL, which is the one
                    # loss number that stays comparable when the tokenizer changes.
                    _field_nats[_fname] = _mean * _n
                    _idx += _n
                _mp = message_perplexity(
                    _field_nats,
                    token_mode=str(getattr(args, "token_mode", "26tok")),
                    field_widths={f: int(_tok_lens[i])
                                  for i, f in enumerate(Message_Tokenizer.FIELDS)},
                    expected_fields=list(Message_Tokenizer.FIELDS),
                ).as_dict()
                for _variant in ("main", "full"):
                    _v = _mp["variants"][_variant]
                    log_dict[f"msg_ppl/{_variant}_nats"] = _v["nats_per_message"]
                    log_dict[f"msg_ppl/{_variant}_bits"] = _v["bits_per_message"]
                for _role, _nats in _mp["components_nats_per_message"].items():
                    log_dict[f"msg_ppl/component_{_role}_nats"] = _nats
                # 0 under ignore_times, and that is the point: it says out loud
                # that `full` did not cover the whole message on this run.
                log_dict["msg_ppl/fields_missing"] = float(len(_mp["missing_fields"]))
            wandb.log(log_dict, step=global_step)
        if save_flag:
            if is_distributed:
                ckpt_st = _reshard_for_ckpt(cb_state)
            else:
                ckpt_st = deduplicate_trainstate(cb_state)
            ckpt = {
                'model': ckpt_st,
                'config': vars(args) if is_main_process else {},
                'metrics': {
                    'loss_train': float(cb_loss),
                    'epoch': cb_epoch,
                    'step_in_epoch': cb_batch_idx,
                }
            }
            try:
                save_checkpoint(ckpt_mgr, ckpt, global_step)
                if is_main_process:
                    print(f"[Checkpoint] Mid-epoch save: epoch={cb_epoch}, "
                          f"step={cb_batch_idx}, global_step={global_step}")
            except (OSError, ValueError) as e:
                print(f"[Checkpoint] WARNING: mid-epoch save failed: {e}")

    for epoch in range(start_epoch, args.epochs):
        # Free residual memory before training.
        gc.collect()

        # Mid-epoch resume: rebuild trainloader with sampler-level skip
        # so DataLoader never calls __getitem__ for already-completed batches.
        if resume_from_step is not None and resume_from_step > 0:
            print(f"[Resume] Rebuilding trainloader with sampler skip: "
                  f"step={resume_from_step}, epoch={epoch}")
            trainloader = create_lobster_train_loader(
                lobster_dataset,
                seed=args.jax_seed,
                per_process_bsz=args.micro_bsz * args.num_devices,
                num_workers=args.n_data_workers,
                reset_train_offsets=False,
                shuffle=args.shuffle_train,
                use_distributed_sampler=is_distributed,
                process_rank=process_rank,
                process_count=process_count,
                resume_from_step=resume_from_step,
                resume_epoch=epoch,
                uniform_ticker_batches=getattr(args, 'uniform_ticker_batches', False),
                max_train_batches=(getattr(args, 'max_train_batches', 0) or None),
                tickers=getattr(args, 'tickers', None),
            )
        else:
            # Update DistributedSampler epoch for proper cross-epoch shuffling
            if hasattr(trainloader, 'sampler') and hasattr(trainloader.sampler, 'set_epoch'):
                trainloader.sampler.set_epoch(epoch)

        print(f"[*] Starting Training Epoch {epoch + 1}...")
        print(f"[*] Step {step} - LR managed by optax schedules")
        print('Training on', args.num_devices, 'devices.')
        train_rng, skey = random.split(train_rng)

        #Pass an initial hidden state to be used in case of the 'RNN' forward pass being used.
        state, train_loss, ce_by_tok, interrupted_at_step = train_epoch(state,
                                              skey,
                                              trainloader,
                                              seq_len,
                                              batchnorm,
                                              None,  # lr_params=None → optax schedules
                                              args.num_devices,
                                              args.debug_loading,
                                              args.enable_profiler,
                                              args.curtail_epochs,
                                              init_hidden,
                                              epoch,
                                              ignore_times,
                                              args.log_ce_tables,
                                              mesh=mesh,
                                              jit_train_step_fn=jit_train_step,
                                              checkpoint_callback=step_checkpoint_callback,
                                              checkpoint_every_n_steps=getattr(args, 'checkpoint_every_n_steps', 'auto'),
                                              job_start_time=job_start_time,
                                              max_job_hours=getattr(args, 'max_job_hours', 24.0),
                                              save_before_timeout_minutes=getattr(args, 'save_before_timeout_minutes', 30),
                                              resume_from_step=resume_from_step,
                                              validate_callback=None,
                                              validate_every_n_steps=train_only_validate_every_n_steps,
                                              flops_per_step=_flops_per_step,
                                              num_gpus=total_devices,
                                              )
        # resume_from_step only applies to the first epoch after restore.
        # If we rebuilt the trainloader with sampler skip, restore the normal
        # loader for subsequent epochs (so DistributedSampler.set_epoch works).
        if resume_from_step is not None:
            trainloader = create_lobster_train_loader(
                lobster_dataset,
                seed=args.jax_seed,
                per_process_bsz=args.micro_bsz * args.num_devices,
                num_workers=args.n_data_workers,
                reset_train_offsets=False,
                shuffle=args.shuffle_train,
                use_distributed_sampler=is_distributed,
                process_rank=process_rank,
                process_count=process_count,
                uniform_ticker_batches=getattr(args, 'uniform_ticker_batches', False),
                max_train_batches=(getattr(args, 'max_train_batches', 0) or None),
                tickers=getattr(args, 'tickers', None),
            )
        resume_from_step = None
        step = int(state.step)

        # Handle timeout interrupt: skip validation + epoch-end checkpoint, exit
        if interrupted_at_step is not None:
            if is_main_process:
                print(f"[Train] Epoch {epoch+1} interrupted at step {interrupted_at_step} due to timeout")
                print(f"[Train] To resume: RESTORE_PATH={ckpt_dir} RESTORE_STEP={step} "
                      f"RESUME_FROM_STEP={interrupted_at_step}")
            break

        if args.random_offsets_train:
            # Refresh random offsets in-place without rebuilding DataLoader.
            # This keeps persistent workers alive across epochs.
            lobster_dataset.reset_train_offsets()

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(f"\tTrain Loss: {train_loss:.5f}")

        # Save checkpoint — ALL ranks must call save() for Orbax barrier sync.
        # Orbax primary_host=0 ensures only rank 0 writes to disk.
        # Multi-host: re-shard state to ensure consistent NamedSharding for Orbax.
        # Single-host: deduplicate to single device first.
        if is_distributed:
            ckpt_state = _reshard_for_ckpt(state)
        else:
            ckpt_state = deduplicate_trainstate(state)
        ckpt = {
            'model': ckpt_state,
            'config': vars(args) if is_main_process else {},
            'metrics': {
                'loss_train': float(train_loss),
            }
        }
        try:
            save_checkpoint(ckpt_mgr, ckpt, int(state.step))
            if is_main_process:
                print(f"[Checkpoint] Epoch-end save: epoch={epoch}, global_step={int(state.step)}")
        except (OSError, ValueError) as e:
            print(f"\n[FATAL] Checkpoint save failed at epoch {epoch}: {e}")
            print("[FATAL] Likely disk quota or serialization issue. Exiting.")
            if ckpt_mgr is not None:
                try:
                    ckpt_mgr.close()
                except Exception:
                    pass
            sys.exit(1)
        del ckpt
        del ckpt_state

        current_lr = float(lr_schedule_fn(step))
        current_ssm_lr = float(ssm_lr_schedule_fn(step))
        wandb.log(
            {
                "Training Loss": train_loss,
                "lr": current_lr,
                "ssm_lr": current_ssm_lr,
            }
        )
        wandb.run.summary["Final Training Loss"] = train_loss

        gc.collect()
        if is_distributed:
            sync_global_devices(f"post_epoch_{epoch}")
        continue

    # Wait for async checkpoint writes to complete before exiting
    if ckpt_mgr is not None:
        ckpt_mgr.wait_until_finished()
        ckpt_mgr.close()
