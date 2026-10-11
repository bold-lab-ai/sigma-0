import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Preallocating ~90% of the card is right when the job owns it, and wrong when the
# run is attached to an allocation that is already training on it: the neighbour's
# memory is already resident, so grabbing 90% of the total takes memory that is not
# free and can OOM a job that has been running for hours. setdefault keeps the
# owning-the-card behaviour for every caller that sets nothing, and lets an attached
# run cap itself:
#   XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.25
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
# os.environ['XLA_FLAGS'] ='--xla_gpu_deterministic_ops=true'


os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")

# import torch
# torch.multiprocessing.set_start_method('spawn')

# Add parent folder to path (to run this file from subdirectories)
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(parent_folder_path)
from base_model.encoding.encoding import SUPPORTED_TOKEN_MODES  # noqa: E402

print(sys.path)
from matching_engine.jaxob.jorderbook import OrderBook
import matching_engine.jaxob.JaxOrderBookArrays as job
from base_model.models.registry import available_architectures

# from argparse import Namespace
from glob import glob
import numpy as onp
import pandas as pd
# from functools import partial
# from typing import Union, Optional
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
# from line_profiler import LineProfiler

import jax
import jax.numpy as jnp
from jax.nn import one_hot
# from jax import random
# from jax.scipy.linalg import block_diag
# from flax import jax_utils
# from flax.training import checkpoints
# import orbax

#from lob.lob_seq_model import BatchLobPredModel
# from lob.train_helpers import create_train_state, eval_step, prep_batch, cross_entropy_loss, compute_accuracy
# SSM init handled by init_train_state (supports gdn + mamba3 via ssm_type)
from base_model.encoding.encoding import Vocab, Message_Tokenizer
assert Message_Tokenizer.MSG_LEN == 26
print(f"[Encoding] Using 26tok encoding (MSG_LEN={Message_Tokenizer.MSG_LEN})")
# from lobster_dataloader import LOBSTER_Dataset, LOBSTER_Subset, LOBSTER_Sampler, LOBSTER

import base_model.data.preproc as preproc
from time import time
# import inference
from base_model.inference import generate as inference
import base_model.inference.validation as valh
from base_model.training.state import init_train_state, load_checkpoint, load_metadata, load_args_from_checkpoint
# import lob.encoding as encoding


import base_model.inference.evaluation as eval
from base_model.data.preproc import transform_L2_state



##################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', type=str, default='GOOG', help='stock to evaluate')
    parser.add_argument('--checkpoint_step', type=int, default=None, help='Which checkpoint step to load')
    parser.add_argument('--test_split', type=float, default=0.1, help='Which test split to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument("--n_sequences", type=int, default=1024, help="Total number of sequences to generate (before rank splitting)")
    parser.add_argument("--n_cond_msgs", type=int, default=250, help="Number of conditioning messages")
    parser.add_argument("--n_gen_msgs", type=int, default=250, help="Number of messages to generate")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to preprocessed data directory")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save inference results")
    parser.add_argument("--sample_indices_file", type=str, default=None, help="File with dataset indices for HF-matched mode (one per line)")
    parser.add_argument("--window_starts_file", type=str, default=None,
                        help="JSON from run/base_model/eval/pick_trade_anchors.py giving explicit "
                             "(file_idx, seq_start) conditioning windows. Each window becomes one "
                             "dataset index. Mutually exclusive with --sample_indices_file.")
    parser.add_argument("--wide_book_dir", type=str, default=None,
                        help="Path to wider L2 book .npy files (e.g. L100) for simulator init")
    parser.add_argument("--wide_levels", type=int, default=10,
                        help="Number of book levels in wide_book_dir data (default: 10 = no change)")
    parser.add_argument(
        # Taken from the registry rather than repeated here. Every other
        # entrypoint already does this; the literal copy left here meant a
        # newly registered backbone trained fine and then could not be
        # generated from, which surfaces only at evaluation time.
        "--architecture", default=None,
        choices=list(available_architectures()),
        help="Optional checkpoint-architecture assertion")
    # Multi-GPU args
    parser.add_argument("--rank", type=int, default=0, help="Rank of this process (0-indexed)")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
    # Model code override (parsed early in sys.argv for path injection, declared here for --help)
    parser.add_argument("--model_code_dir", type=str, default=None,
                        help="deprecated and rejected; sigma-0 owns the model runtime")
    parser.add_argument("--token_mode", type=str, default='26tok',
                        choices=list(SUPPORTED_TOKEN_MODES),
                        help="which lossless encoding builds the tensors; "
                             "the zoo is 26-token but 24/23/22 are selectable now")
    parser.add_argument("--gt_compare", action='store_true', default=False,
                        help="Enable ground-truth comparison: token distributions, divergence trajectory, validity rate")
    parser.add_argument("--save_format", type=str, default='csv',
                        help="Output format (csv or npy). Accepted for pipeline compatibility.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed for generation sampling (per-rank: seed+rank). "
                             "Vary across runs for replicate/seed error bars. Index selection stays fixed at 42.")
    # do(Q) injection: place one order of our choosing at one generation step
    # (see docs/inference.md). Off unless --inject_step is given;
    # when off, sample_new is called without any inject kwargs, so the call is
    # literally the pre-injection call and the traced program is unchanged.
    parser.add_argument("--inject_step", type=int, default=None,
                        help="Generation step (0-indexed, message-level) to inject at; omit for no injection")
    parser.add_argument("--inject_event_type", type=int, default=4, choices=[1, 2, 3, 4],
                        help="LOBSTER event type of the injected order (1=new limit, 4=execution)")
    parser.add_argument("--inject_side", type=int, default=1, choices=[0, 1],
                        help="Side of the injected order: 1=bid, 0=ask")
    parser.add_argument("--inject_qty", type=int, default=0,
                        help="Quantity of the injected order (shares); required >= 1 when --inject_step is given")
    parser.add_argument("--inject_price_offset_ticks", type=int, default=0,
                        help="Price offset in ticks relative to the rollout's live tick-floored mid "
                             "(negative = toward/into the bid side)")

    run_args = parser.parse_args()
    if run_args.model_code_dir:
        parser.error("--model_code_dir is no longer supported; model code is vendored in sigma-0")
    if run_args.inject_step is None:
        # Catch the half-specified intervention: inject knobs without a step
        # would silently run an observational rollout that the caller believes
        # is interventional.
        _inject_knobs = {
            '--inject_event_type': (run_args.inject_event_type, 4),
            '--inject_side': (run_args.inject_side, 1),
            '--inject_qty': (run_args.inject_qty, 0),
            '--inject_price_offset_ticks': (run_args.inject_price_offset_ticks, 0),
        }
        _set_without_step = [k for k, (v, d) in _inject_knobs.items() if v != d]
        if _set_without_step:
            parser.error(f"{', '.join(_set_without_step)} given without --inject_step; "
                         "an injection needs its step")
    elif run_args.inject_qty < 1:
        # sample_new's default qty is 0: forgetting it injects an empty order,
        # the book never changes, and the run reads as a null effect.
        parser.error("--inject_step given but --inject_qty < 1; an empty order is not an intervention")

    overfit_debug = False

    # Resolve paths: CLI args take priority, fall back to hardcoded defaults
    data_dir = run_args.data_dir
    ckpt_path = run_args.ckpt_path
    save_dir = run_args.save_dir

    if data_dir is None or ckpt_path is None or save_dir is None:
        parser.error("--data_dir, --ckpt_path, and --save_dir are required")

    ##################################################

    n_gen_msgs = run_args.n_gen_msgs
    n_messages_conditional = run_args.n_cond_msgs
    n_eval_messages = n_gen_msgs  # how many to load from dataset
    token_mode = run_args.token_mode
    if token_mode == '1tok':
        # 1tok: seq lengths still in tokens for dataset loading (same data format)
        # but cond_seq_len is only used for slicing in sample_new
        eval_seq_len = (n_eval_messages-1) * Message_Tokenizer.MSG_LEN
        cond_seq_len = n_messages_conditional * Message_Tokenizer.MSG_LEN
    else:
        eval_seq_len = (n_eval_messages-1) * Message_Tokenizer.MSG_LEN
        cond_seq_len = (n_messages_conditional) * Message_Tokenizer.MSG_LEN
    data_levels = 10
    # TODO: deprecated - remove from functions
    sim_book_levels = 20 # 10  # order book simulator levels
    sim_queue_len = 100  # per price in sim, how many orders in queue

    v = Vocab()
    n_classes = len(v)
    eval_book_seq_len = eval_seq_len

    # Per-rank RNG: different sampling randomness per rank. Base seed from --seed
    # (default 42) so replicate runs vary; index selection below stays fixed at 42
    # so every seed scores the SAME windows.
    rng = jax.random.key(run_args.seed + run_args.rank)
    rng, rng_ = jax.random.split(rng)
    if overfit_debug:
        sample_top_n = 1
    else:
        sample_top_n = -1
    tick_size = 100

    # load train state from disk

    args = load_metadata(ckpt_path)
    from base_model.models.registry import (architecture_from_args, stamp_architecture,
                             architecture_has_attention as _architecture_has_attention)

    checkpoint_architecture = architecture_from_args(args)
    if run_args.architecture and run_args.architecture != checkpoint_architecture:
        raise ValueError(
            f"checkpoint architecture is {checkpoint_architecture!r}, but "
            f"--architecture={run_args.architecture!r} was requested")
    stamp_architecture(args, checkpoint_architecture)
    args.token_mode = '26tok'
    # Any architecture with attention layers needs a KV cache long enough for
    # the whole request, and that includes hybrids. Keying this on an explicit
    # list of names let hybrid_mamba3 fall through to registry.py's default of
    # 26,000, which happens to cover (250 + 250) x 26 = 13,000 and so never
    # showed at the published context length. At (1000 + 1000) x 26 = 52,000 the
    # cache is exactly half of what generation writes into it; the position index
    # clamps, keys land in the wrong slots, and the model appears to have lost
    # the ability to refer to anything, near or far. It reads like an attention
    # length-extrapolation result and it is a two-line bug.
    #
    # Ask what the backbone is rather than what it is called.
    if _architecture_has_attention(checkpoint_architecture):
        # Cache tensors are runtime state rather than parameters, so inference
        # may safely enlarge them to cover the requested prefix plus generation.
        required_cache_len = (
            n_messages_conditional + n_gen_msgs
        ) * Message_Tokenizer.MSG_LEN
        checkpoint_cache_len = int(
            getattr(args, 'attention_max_cache_len', required_cache_len)
        )
        if checkpoint_cache_len < required_cache_len:
            print(
                f"[attention-cache] expanding max_cache_len "
                f"{checkpoint_cache_len} -> {required_cache_len} tokens"
            )
            args.attention_max_cache_len = required_cache_len
        else:
            print(f"[attention-cache] max_cache_len {checkpoint_cache_len} "
                  f">= required {required_cache_len}, unchanged")
    if not getattr(args, 'book_transform', False):
        raise ValueError(
            "the integrated autoregressive inference path requires checkpoints "
            "trained with book_transform=True"
        )
    # Reconstruct the training-time book width from checkpoint metadata.  The
    # standard launcher uses the volume-image transform (book_depth + 3); raw
    # book checkpoints retain their on-disk width and can stamp d_book directly.
    book_dim = getattr(args, 'd_book', None)
    if book_dim is None and getattr(args, 'book_transform', False):
        book_dim = int(getattr(args, 'book_depth', 500)) + 3
    if book_dim is None:
        book_candidates = sorted(glob(os.path.join(data_dir, '*book*.npy')))
        if not book_candidates:
            raise ValueError(
                "cannot infer raw book width: checkpoint has no d_book and "
                f"no *book*.npy file exists in {data_dir!r}")
        book_dim = int(onp.load(book_candidates[0], mmap_mode='r').shape[1])
    args.d_book = int(book_dim)
    tp_size = getattr(args, 'tp_size', 1)
    if tp_size > 1:
        args.num_devices = tp_size
    else:
        args.num_devices = 1
    args.bsz=1
    # init_train_state builds a dummy forward pass of shape (micro_bsz, seq_len, ...).
    # At long generation (seq_len = (n_gen-1) * MSG_LEN ≈ 10^5) this dominates device memory.
    # The training micro_bsz is irrelevant for inference, so force 1.
    args.micro_bsz = 1
    # Force standard optimizer for inference — avoids Muon/optax version issues
    # (we only need params, not the optimizer state)
    args.opt_config = "standard"

    # Hierarchical no-book: route to inference wrapper that accepts book args but discards them
    if getattr(args, 'hierarchical_nobook', False):
        args.no_book_inference_wrapper = True

    # Legacy Mamba3 BCNorm/out_norm epsilon compat for pre-2026-04-22 checkpoints.
    # Opt-in via MAMBA3_LEGACY_NORM=1; inert otherwise. Must run before the model is
    # built so Mamba3SSM.setup picks up the training-time normalizers.
    from base_model.models.legacy_norm import maybe_install_mamba3_legacy_norm
    maybe_install_mamba3_legacy_norm(getattr(args, 'ssm_type', None))

    new_train_state, model_cls = init_train_state(
        args,
        n_classes=n_classes,
        seq_len=eval_seq_len,
        book_dim=book_dim,
        book_seq_len=eval_book_seq_len,
    )

    # Set up TP mesh for inference if checkpoint was trained with TP
    if tp_size > 1:
        from base_model.sharding.sharding_utils import create_simple_mesh
        tp_mesh = create_simple_mesh(tp_size, tp_size=tp_size)
        import base_model.inference.validation as valh
        valh._TP_MESH = tp_mesh
        m3_expand = getattr(args, 'mamba3_expand', 2)
        m3_headdim = getattr(args, 'mamba3_headdim', 64)
        d_inner = m3_expand * args.d_model
        valh._TP_NH = max(1, d_inner // m3_headdim)
        # Remove inner @jax.jit from apply_model — outer jit(shard_map(vmap(generate))) handles it
        valh.apply_model = valh._apply_model_impl
        valh.apply_model_1tok = valh._apply_model_1tok_impl
        print(f"[TP] Inference with tp_size={tp_size}, mesh={tp_mesh}, nh={valh._TP_NH}")


    # jax.tree_util.tree_map(lambda x: x.shape,state)
    _step = 0 if run_args.checkpoint_step is None else run_args.checkpoint_step
    try:
        ckpt = load_checkpoint(
            new_train_state,
            ckpt_path,
            step=_step,
            train=False,
            partial_restore=True,
        )
    except (ValueError, TypeError) as _ckpt_err:
        # opt_state tree structure may differ between optax versions.
        # For inference we only need params — read directly via TensorStore.
        print(f"[load_checkpoint] StandardRestore failed: {_ckpt_err}")
        print("[load_checkpoint] TensorStore direct-read fallback (params only)...")
        import json, ast
        import numpy as _onp
        import tensorstore as ts
        import orbax.checkpoint as ocp
        from base_model.training.state import deduplicate_trainstate

        _abs_path = os.path.abspath(ckpt_path)
        _mngr = ocp.CheckpointManager(
            _abs_path, item_names=('state', 'metadata'),
            options=ocp.CheckpointManagerOptions(),
        )
        if _step == 0:
            _step = _mngr.latest_step()
        _meta = _mngr.restore(_step, args=ocp.args.Composite(
            metadata=ocp.args.JsonRestore()))['metadata']

        _state_dir = os.path.join(_abs_path, str(_step), 'state')
        _mj = json.loads(open(os.path.join(_state_dir, '_METADATA')).read())
        _use_ocdbt = (os.path.exists(os.path.join(_state_dir, 'ocdbt.base_path'))
                      or os.path.exists(os.path.join(_state_dir, 'manifest.ocdbt')))
        _use_zarr3 = _mj.get('use_zarr3', False)
        _flat = {tuple(ast.literal_eval(k)): v
                 for k, v in _mj['tree_metadata'].items()}

        print(f"  Reading {sum(1 for k in _flat if k[0]=='params')} arrays "
              f"(OCDBT={_use_ocdbt})")

        def _make_tspec(state_dir, name, use_ocdbt, use_zarr3):
            """Build a TensorStore spec for a single array (orbax-version-agnostic)."""
            if use_ocdbt:
                return {'driver': 'zarr3' if use_zarr3 else 'zarr',
                        'kvstore': {'driver': 'ocdbt',
                                    'base': f'file://{state_dir}',
                                    'path': name}}
            else:
                return {'driver': 'zarr3' if use_zarr3 else 'zarr',
                        'kvstore': {'driver': 'file',
                                    'path': os.path.join(state_dir, name)}}

        _raw = {}
        for kp in _flat:
            if kp[0] != 'params':
                continue
            _tspec = _make_tspec(_state_dir, '.'.join(kp), _use_ocdbt, _use_zarr3)
            _raw[kp[1:]] = _onp.asarray(
                ts.open(_tspec, open=True).result().read().result())

        _params = {}
        for kp, arr in _raw.items():
            d = _params
            for key in kp[:-1]:
                d = d.setdefault(key, {})
            d[kp[-1]] = arr
        print(f"  Loaded {len(_raw)} param arrays")

        _dedup = deduplicate_trainstate(new_train_state)
        ckpt = _meta
        ckpt['model'] = _dedup.replace(params=_params)

    state = ckpt['model']

    if 'message_encoder' in state.params:
        print(state.params['message_encoder']['encoder']['embedding'].shape)
    else:
        print(f"[1tok model] field_embedding keys: {list(state.params.get('field_embedding', {}).keys())}")


    import chex
    chex.clear_trace_counter()

    model = model_cls(training=False, step_rescale=1.0)

    ##################################################

    import base_model.inference.evaluation as eval

    msg_files = sorted(glob(str(data_dir) + '/*message*.npy'))
    book_files = sorted(glob(str(data_dir) + '/*book*.npy'))

    # Event-anchored mode: windows are chosen offline so each one ends on a trade,
    # instead of falling wherever the fixed grid puts them (only ~2% of grid windows
    # end on a trade).
    explicit_windows = None
    window_spec = None
    if run_args.window_starts_file is not None:
        if run_args.sample_indices_file is not None:
            raise SystemExit("--window_starts_file and --sample_indices_file are mutually "
                             "exclusive: both decide which windows get generated")
        with open(run_args.window_starts_file) as fh:
            window_spec = json.load(fh)
        explicit_windows = onp.asarray(window_spec["windows"], dtype=onp.int64)
        expected_n = window_spec["n_cond_msgs"] + window_spec["n_gen_msgs"]
        if expected_n != n_messages_conditional + n_eval_messages:
            raise SystemExit(
                f"--window_starts_file was built for {window_spec['n_cond_msgs']}+"
                f"{window_spec['n_gen_msgs']} messages but this run asks for "
                f"{n_messages_conditional}+{n_eval_messages}. The trade would no longer "
                f"land on the last conditioning message.")
        print(f"[window_starts] {len(explicit_windows)} anchored windows from "
              f"{run_args.window_starts_file}")

    ds = inference.get_dataset(data_dir,
                               n_messages_conditional,
                               n_eval_messages,
                               test_split= run_args.test_split,
                               wide_book_dir=run_args.wide_book_dir,
                               explicit_windows=explicit_windows,
                            #    day_indeces= [0],
                            #    limit_seq=4
                               )

    if window_spec is not None:
        # file_idx is a position in the resolved file list, and that list depends on
        # test_split and on which files survived validation. Resolving it differently
        # here would point every window at another trading day without any error, so
        # compare the lists rather than trusting that the flags matched.
        recorded = [os.path.abspath(f) for f in window_spec["message_files"]]
        actual = [os.path.abspath(f) for f in ds.message_files]
        if recorded != actual:
            raise SystemExit(
                "file list mismatch: --window_starts_file was built against "
                f"{len(recorded)} message files but this run resolved {len(actual)}. "
                f"file_idx would point at the wrong day. Recorded test_split="
                f"{window_spec['test_split']}, this run used {run_args.test_split}.\n"
                f"  first recorded: {recorded[0] if recorded else '(none)'}\n"
                f"  first actual  : {actual[0] if actual else '(none)'}")
        print(f"[window_starts] file list matches ({len(actual)} files)")

    print("Dataset length: ", len(ds))

    ##################################################

    import logging
    # logging.basicConfig(filename='ar_debug.log', level=logging.DEBUG)
    _debug_log_path = os.environ.get('GENERATION_DEBUG_LOG', f'/tmp/generation_debug_{os.getpid()}.log')
    fhandler = logging.FileHandler(filename=_debug_log_path, mode='w')
    logger = logging.getLogger()
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(fhandler)
    logger.setLevel(logging.WARNING)
    # logger.setLevel(logging.DEBUG)

    ##################################################

    # Compute this rank's dataset indices
    rank = run_args.rank
    world_size = run_args.world_size
    n_total = run_args.n_sequences

    if explicit_windows is not None:
        # Every dataset index is one anchored window, so there is nothing to sample:
        # the choice was already made offline. n_sequences only truncates, and the
        # picker shuffled after balancing directions, so a prefix stays balanced.
        all_indices = list(range(len(ds)))
        if n_total < len(all_indices):
            print(f"[window_starts] using first {n_total} of {len(all_indices)} anchored windows")
            all_indices = all_indices[:n_total]
        n_total = len(all_indices)
    elif run_args.sample_indices_file is not None:
        # HF-matched mode: read indices from file
        with open(run_args.sample_indices_file, 'r') as f:
            all_indices = [int(line.strip()) for line in f if line.strip()]
        n_total = len(all_indices)
    else:
        # Random mode: generate full index set with shared seed 42
        rng_idx = jax.random.key(42)
        all_indices = jax.random.choice(
            rng_idx,
            jnp.arange(len(ds), dtype=jnp.int32),
            shape=(n_total,),
            replace=False
        ).tolist()

    # Interleaved split: rank takes every world_size-th index starting from rank
    rank_indices = all_indices[rank::world_size]
    n_samples = len(rank_indices)

    # Persist the exact evaluation pool before any per-rank batch padding.  The
    # production launcher forbids padding, but retaining both the global and
    # assigned lists makes the common-window contract independently auditable
    # across architectures and training seeds.
    index_bytes = json.dumps(all_indices, separators=(",", ":")).encode("utf-8")
    index_manifest = {
        "selection_seed": (42 if (run_args.sample_indices_file is None
                                  and explicit_windows is None) else None),
        "sample_indices_file": run_args.sample_indices_file,
        "dataset_length": len(ds),
        "requested_sequences": n_total,
        "world_size": world_size,
        "rank": rank,
        "all_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "all_indices": all_indices,
        "rank_indices": rank_indices,
    }
    if explicit_windows is not None:
        # Added only when event-anchored generation is actually in use. This
        # manifest is hashed into the do(Q) injection arm digest
        # (lobpipeline/injection_verification.py:82), so an unconditional field
        # would change the digest of every run that never asked for this feature
        # and break a byte-identity claim measured on a specific node. Off means
        # off, the same rule the injection flags follow.
        index_manifest["window_starts_file"] = run_args.window_starts_file
        index_manifest["event_anchored"] = True
        # Kept so a result directory says which windows produced it without the
        # picker's JSON having to still exist.
        index_manifest["anchor_windows"] = explicit_windows[rank_indices].tolist()
    index_path = Path(save_dir) / f"sample_indices_rank{rank}.json"
    if index_path.exists():
        raise FileExistsError(f"refusing to replace sample-index manifest: {index_path}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_tmp = index_path.with_suffix(index_path.suffix + f".tmp.{os.getpid()}")
    index_tmp.write_text(
        json.dumps(index_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index_tmp.replace(index_path)
    print(
        f"[Rank {rank}/{world_size}] Sample-index manifest: {index_path} "
        f"sha256={index_manifest['all_indices_sha256']}"
    )

    print(f"[Rank {rank}/{world_size}] Processing {n_samples} sequences out of {n_total} total")
    print(f"[Rank {rank}/{world_size}] GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"[Rank {rank}/{world_size}] First indices: {rank_indices[:5]}")

    batch_size = run_args.batch_size

    # Pad rank_indices to be divisible by batch_size
    if n_samples % batch_size != 0:
        n_padded = ((n_samples + batch_size - 1) // batch_size) * batch_size
        n_pad = n_padded - n_samples
        # Tile indices to cover padding (handles n_pad > n_samples, e.g. many ranks / small dataset)
        rank_indices = (rank_indices * (n_padded // n_samples + 1))[:n_padded]
        print(f"[Rank {rank}/{world_size}] Padded {n_pad} indices to fill last batch ({n_padded} total)")

    # m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = inference.sample_new(
    # saves data to disk
    # When --inject_step is absent this dict is empty and the sample_new call
    # is byte-for-byte the pre-injection call (no kwargs, no traced change).
    inject_kwargs = {}
    if run_args.inject_step is not None:
        inject_kwargs = dict(
            inject_step=run_args.inject_step,
            inject_event_type=run_args.inject_event_type,
            inject_side=run_args.inject_side,
            inject_qty=run_args.inject_qty,
            inject_price_offset_ticks=run_args.inject_price_offset_ticks,
        )
    start=time()
    inference.sample_new(
        len(rank_indices),
        batch_size,
        ds,
        rng,
        cond_seq_len,
        n_messages_conditional,
        n_gen_msgs,
        state,
        model,
        args.batchnorm,
        v.ENCODING,
        run_args.stock,
        n_vol_series=int(getattr(args, 'book_depth', 500)),
        save_folder=save_dir,
        sample_top_n= sample_top_n,
        args=args,
        conditional= True if n_messages_conditional>0 else False,
        overfit_debug=overfit_debug,
        sample_indices=rank_indices,
        wide_levels=run_args.wide_levels,
        token_mode=token_mode,
        gt_compare=run_args.gt_compare,
        **inject_kwargs,
    )
    print(f"[Rank {rank}/{world_size}] Generation time for {n_samples} sequences across {batch_size} batch size: {time()-start}")
