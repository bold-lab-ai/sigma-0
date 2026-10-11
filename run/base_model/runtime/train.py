# CAVE: only for debugging purposes
import os
from base_model.encoding.encoding import SUPPORTED_TOKEN_MODES  # noqa: E402
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=48'
# no GPU use at all
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# allocate and de-allocate memory as needed (SLOW)
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# TODO: change this if num_devices changes (is less than all of the available ones11)
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
# os.environ["NCCL_DEBUG"]="INFO"

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
if __name__ == "__main__":
        pass
else:
        # Forces all generated worker processes to not run on GPU.
        #  Required at this high level, because the init func in the
        # worker spawn interface happens after init. of the CUDA process.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["JAX_PLATFORMS"] = "cpu"

from base_model.data.dataloading import Datasets

if __name__ == "__main__":
        import argparse
        from base_model.utils.cli import str2bool
        # Set visible GPUs from SLURM config (GPUS_PER_NODE set in batch script)
        _n_gpus = int(os.environ.get('GPUS_PER_NODE', '4'))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(_n_gpus))
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        # Use setdefault so batch script env vars take precedence
        os.environ.setdefault("NCCL_TIMEOUT", "600")
        os.environ.setdefault("NCCL_IB_DISABLE", "0")
        os.environ.setdefault("NCCL_P2P_DISABLE", "0")

        #physical_devices = tf.config.list_physical_devices('GPU')
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #tf.config.experimental.set_visible_devices([], "GPU")

        parser = argparse.ArgumentParser()

        parser.add_argument("--USE_WANDB", type=str2bool, default=True,
                                                help="log with wandb?")
        parser.add_argument("--wandb_project", type=str, default="LOBS5v2",
                                                help="wandb project name")
        parser.add_argument("--wandb_entity", type=str, default="sasrey",
                                                help="wandb entity name, e.g. username")
        parser.add_argument("--dir_name", type=str, default='./data/LOBS5v2Cached/',
                                                help="name of directory where data is cached")
        parser.add_argument("--dataset", type=str, choices=Datasets.keys(),
                                                default='lobster-prediction',
                                                help="dataset name")
        parser.add_argument("--masking", type=str, choices={'causal', 'random','last_pos','none'},
                                                default='causal',  # random
                                                help="causal, random or last position masking of sequences")
        parser.add_argument("--use_book_data", type=str2bool, default=False,
                                                  help="use book data in addition to message data")
        parser.add_argument("--merging", type=str, choices={'projected', 'padded'},
                                                default='projected',
                                                help="Method for merging the book model with the message model. Cannot use RNN mode with projected mode.")
        parser.add_argument("--use_simple_book", type=str2bool, default=False,
                                                  help="use raw price (-p0) and volume series instead of 'volume image representation'")
        parser.add_argument("--book_transform", type=str2bool, default=False,
                                                  help="transform loaded book data to volume image repr. in dataloader")
        parser.add_argument("--book_depth", type=int, default=500,
                                                  help="number of tick levels to use in book data [if book_transform=True]")
        parser.add_argument("--book_ablation", type=str, default='real',
                                                  choices=['real', 'zero', 'noise', 'shuffle'],
                                                  help="book input ablation: real=normal, zero=all-zeros, noise=Gaussian, shuffle=time-permuted")
        parser.add_argument("--restore", type=str,
                                                  help="if given restore from given checkpoint dir")
        parser.add_argument("--restore_step", type=int,
                                                help="global step number to restore from (used as CheckpointManager key)")
        parser.add_argument("--resume_from_step", type=int, default=None,
                                                help="resume from this batch_idx within the start epoch (skip earlier batches)")
        parser.add_argument("--partial_restore", action='store_true', default=False,
                                                help="allow partial restore when model structure changed (strict=False)")
        parser.add_argument("--restore_reset_schedule", type=str2bool, default=False,
                                                help="after restore, reset state.step to 0 so LR schedule starts fresh (for curriculum/fine-tune)")
        parser.add_argument("--checkpoint_every_n_steps", type=str, default="auto",
                                                help="'auto' (30min), integer N, or '0' to disable mid-epoch checkpoints")
        parser.add_argument("--max_job_hours", type=float, default=24.0,
                                                help="max job duration in hours (for timeout checkpoint)")
        parser.add_argument("--save_before_timeout_minutes", type=int, default=30,
                                                help="save checkpoint this many minutes before job timeout")
        parser.add_argument("--msg_seq_len", type=int, default=500,  # 500
                                                help="How many past messages to include in each sample")
        parser.add_argument("--n_data_workers", type=int, default=0,
                                                  help="number of workers used in DataLoader")
        parser.add_argument("--prefetch_factor", type=int, default=2,
                                                  help="DataLoader prefetch_factor (batches per worker to buffer)")

        # Model Parameters
        parser.add_argument("--n_message_layers", type=int, default=2,  # 2
                                                help="Number of layers after fusing message and book data")
        parser.add_argument("--n_book_pre_layers", type=int, default=1,  # 1
                                                help="Number of layers taking in raw book data (before projecting dimensions)")
        parser.add_argument("--n_book_post_layers", type=int, default=1,  # 1
                                                help="Number of book seq layers after projecting book data dimensions")
        parser.add_argument("--n_layers", type=int, default=6,  #6
                                                help="Number of layers after fusing message and book data")
        parser.add_argument("--hierarchical_nobook", type=str2bool, default=False,
                                                help="Hierarchical 2-stage no-book architecture (msg_enc + Dense bottleneck + fused)")
        parser.add_argument("--d_model", type=int, default=32,  #128, 32, 16
                                                help="Number of features, i.e. H, "
                                                          "dimension of layer inputs/outputs")
        parser.add_argument("--ssm_size_base", type=int, default=32,  # 256
                                                help="SSM Latent size, i.e. P")
        parser.add_argument("--blocks", type=int, default=8,  # 8, 4
                                                help="How many blocks, J, to initialize with")
        parser.add_argument("--C_init", type=str, default="trunc_standard_normal",
                                                choices=["trunc_standard_normal", "lecun_normal", "complex_normal"],
                                                help="Options for initialization of C: \\"
                                                          "trunc_standard_normal: sample from trunc. std. normal then multiply by V \\ " \
                                                          "lecun_normal sample from lecun normal, then multiply by V\\ " \
                                                          "complex_normal: sample directly from complex standard normal")
        parser.add_argument("--discretization", type=str, default="zoh", choices=["zoh", "bilinear"])
        parser.add_argument("--mode", type=str, default="none", choices=["none","pool", "last","ema"],
                                                help="options: (for classification tasks) \\" \
                                                          " none: no aggregation, raw output at decoder stage \\" \
                                                          " pool: mean pooling \\" \
                                                          "last: take last element \\" \
                                                          "ema : take exponential moving avg across all")
        from base_model.models.registry import available_architectures
        parser.add_argument(
                # Taken from the registry rather than repeated here: a literal
                # copy silently rejects any newly registered backbone.
                "--architecture", type=str, default=None,
                choices=list(available_architectures()),
                help="Sequence backbone (default: mamba3)")
        parser.add_argument("--ssm_type", type=str, default=None,
                                                help="Deprecated architecture alias; conflicts are rejected")
        parser.add_argument("--model_type", type=str, default=None,
                                                help="Deprecated attention-model alias; conflicts are rejected")
        parser.add_argument("--token_mode", type=str, default="26tok",
                                                choices=list(SUPPORTED_TOKEN_MODES),
                                                help="which lossless encoding builds the tensors; "
                                                     "the zoo is 26-token but 24/23/22 are selectable now")
        parser.add_argument("--gdn_num_heads", type=int, default=None,
                                                help="GDN: number of attention heads (default: d_model // gdn_head_dim)")
        parser.add_argument("--gdn_head_dim", type=int, default=128,
                                                help="GDN: key/query dimension per head")
        parser.add_argument("--gdn_expand_v", type=int, default=2,
                                                help="GDN: value expansion factor (head_v_dim = head_dim * expand_v)")
        parser.add_argument("--gdn_use_conv", type=str2bool, default=True,
                                                help="GDN: apply causal depthwise Conv1d(k=4) on q,k,v")
        parser.add_argument("--gdn_chunk_size", type=int, default=64,
                                                help="GDN: chunkwise parallel chunk size")
        # Mamba3 args
        parser.add_argument("--mamba3_d_state", type=int, default=128,
                                                help="Mamba3: SSM state dimension")
        parser.add_argument("--mamba3_expand", type=int, default=2,
                                                help="Mamba3: expansion factor for d_inner")
        parser.add_argument("--mamba3_headdim", type=int, default=64,
                                                help="Mamba3: per-head dimension")
        parser.add_argument("--mamba3_chunk_size", type=int, default=64,
                                                help="Mamba3: chunkwise chunk size")
        parser.add_argument("--mamba3_rope_fraction", type=float, default=0.5,
                                                help="Mamba3: fraction of d_state used for RoPE")
        parser.add_argument("--mamba3_use_triton", type=str2bool, default=False,
                                                help="Mamba3: use Triton kernels (True, experimental) or pure JAX (False, default)")
        parser.add_argument("--mamba3_use_cuda", type=str2bool, default=False,
                                                help="Mamba3: use CUDA FFI state-scan kernel for SSD phases 4+5+6 (True) or pure JAX (False, default)")
        # Hybrid (architecture=hybrid_mamba3): a mamba3 trunk with a few global
        # attention positions, placed by the Nemotron-H rule unless overridden.
        parser.add_argument("--hybrid_attn_layers", type=str, default=None,
                                                help="Hybrid: comma-separated 0-based trunk positions for attention "
                                                     "(default: Nemotron-H placement for the given n_layers)")
        parser.add_argument("--hybrid_attn_heads", type=int, default=10,
                                                help="Hybrid: attention heads; d_model/heads must be <=256 and a "
                                                     "multiple of 8 for the Pallas causal kernel")
        parser.add_argument("--hybrid_attn_d_ff", type=int, default=0,
                                                help="Hybrid: attention-block FFN width (0 = 4*d_model). Lower it to "
                                                     "parameter-match the mamba3 layer the attention displaced")
        parser.add_argument("--hybrid_attn_flash", type=str2bool, default=True,
                                                help="Hybrid: Pallas memory-efficient causal attention. Off means an "
                                                     "LxL matrix, which at 13k tokens will not fit")
        parser.add_argument("--hybrid_attn_positional_encoding", type=str2bool, default=False,
                                                help="Hybrid: add sinusoidal PE inside attention layers. Nemotron uses "
                                                     "none, and mamba3 already rotates its state with RoPE")
        # Transformer / NSA args
        parser.add_argument("--n_heads", type=int, default=16)
        parser.add_argument("--d_ff", type=int, default=0,
                                                help="Attention FFN width; zero selects 4*d_model")
        parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
        parser.add_argument("--use_flash", type=str2bool, default=False)
        parser.add_argument("--remat", type=str2bool, default=False)
        parser.add_argument("--attention_max_cache_len", type=int, default=26000)
        parser.add_argument("--block_size", type=int, default=26,
                                                help="NSA compressed block size; must equal one 26-token message")
        parser.add_argument("--block_counts", type=int, default=16)
        parser.add_argument("--window_size", type=int, default=512)
        parser.add_argument("--pe_type", choices=["rope", "sinusoidal"], default="rope")
        parser.add_argument("--rope_base", type=float, default=10000.0)
        parser.add_argument("--activation_fn", default="half_glu1", type=str,
                                                choices=["full_glu", "half_glu1", "half_glu2", "gelu"])
        parser.add_argument("--conj_sym", type=str2bool, default=True,
                                                help="whether to enforce conjugate symmetry")
        parser.add_argument("--clip_eigs", type=str2bool, default=False,
                                                help="whether to enforce the left-half plane condition")
        parser.add_argument("--bidirectional", type=str2bool, default=False,  #False,
                                                help="whether to use bidirectional model")
        parser.add_argument("--dt_min", type=float, default=0.001,
                                                help="min value to sample initial timescale params from")
        parser.add_argument("--dt_max", type=float, default=0.1,
                                                help="max value to sample initial timescale params from")

        # Optimization Parameters
        parser.add_argument("--prenorm", type=str2bool, default=True,
                                                help="True: use prenorm, False: use postnorm")
        parser.add_argument("--batchnorm", type=str2bool, default=True,
                                                help="True: use batchnorm, False: use layernorm")
        parser.add_argument("--bn_momentum", type=float, default=0.95,
                                                help="batchnorm momentum")
        parser.add_argument("--micro_bsz", type=int, default=16,
                                                help="per-GPU (micro) batch size")
        parser.add_argument("--num_devices", type=int, default=1,
                                                  help="number of devices (GPUs) to use")
        parser.add_argument("--epochs", type=int, default=100,  #100, 20
                                                help="max number of epochs")
        parser.add_argument("--early_stop_patience", type=int, default=1000,
                                                help="number of epochs to continue training when val loss plateaus")
        parser.add_argument("--ssm_lr_base", type=float, default=1e-3,
                                                help="initial ssm learning rate")
        parser.add_argument("--lr_factor", type=float, default=1,
                                                help="global learning rate = lr_factor*ssm_lr_base")
        parser.add_argument("--dt_global", type=str2bool, default=False,
                                                help="Treat timescale parameter as global parameter or SSM parameter")
        parser.add_argument("--lr_min", type=float, default=0,
                                                help="minimum learning rate. 0 = auto (5%% of base LR)")
        parser.add_argument("--cosine_anneal", type=str2bool, default=True,
                                                help="whether to use cosine annealing schedule")
        parser.add_argument("--warmup_end", type=float, default=0.01,
                                                help="epoch (or fraction) to end linear warmup. Default 0.01 = 1%% of training.")
        parser.add_argument("--lr_patience", type=int, default=1000000,
                                                help="patience before decaying learning rate for lr_decay_on_val_plateau")
        parser.add_argument("--reduce_factor", type=float, default=0.9,
                                                help="factor to decay learning rate for lr_decay_on_val_plateau")
        parser.add_argument("--p_dropout", type=float, default=0.0,
                                                help="probability of dropout")
        parser.add_argument("--weight_decay", type=float, default=0.05,
                                                help="weight decay value")
        parser.add_argument("--max_grad_norm", type=float, default=1.0,
                                                help="max gradient norm for clipping (0 to disable)")
        parser.add_argument("--opt_config", type=str, default="standard", choices=['standard',
                                                                                                                                                              'BandCdecay',
                                                                                                                                                              'BfastandCdecay',
                                                                                                                                                              'noBCdecay',
                                                                                                                                                              'muon'],
                                                help="Opt configurations: \\ " \
                              "standard:       no weight decay on B (ssm lr), weight decay on C (global lr) \\" \
                                  "BandCdecay:     weight decay on B (ssm lr), weight decay on C (global lr) \\" \
                                  "BfastandCdecay: weight decay on B (global lr), weight decay on C (global lr) \\" \
                                  "noBCdecay:      no weight decay on B (ssm lr), no weight decay on C (ssm lr) \\" \
                                  "muon:           Muon (Newton-Schulz) for 2D kernel weights, Adam for SSM, AdamW for rest")
        parser.add_argument("--muon_lr", type=float, default=0.02,
                                                help="Learning rate for Muon kernel weights (paper default 0.02)")
        parser.add_argument("--muon_wd", type=float, default=None,
                                                help="Weight decay for Muon params (default: same as --weight_decay)")
        parser.add_argument("--jax_seed", type=int, default=1919,
                                                help="seed randomness")
        parser.add_argument("--debug_loading", type=str2bool, default=False,
                                                help="Set flag to True to skip any training and just run the loading process.")
        parser.add_argument("--enable_profiler", type=str2bool, default=False,
                                        help="Set flag to True to use the TB profiler.")
        parser.add_argument("--curtail_epochs", type=int, default=None,
                                help="End epoch after n steps. Default is None, never. ")
        parser.add_argument("--mini_epochs", type=int, default=40,
                                help="Number of mini-epochs per data epoch. Validation/test/checkpoint "
                                          "happen after each mini-epoch. Default 40 for single-epoch training.")
        parser.add_argument("--random_offsets_train", type=str2bool, default=True,
                                help="Whether or not the training data is offset randomly at each epoch.")
        parser.add_argument("--shuffle_train", type=str2bool, default=True,
                                help="Whether or not the training data shuffled.")
        parser.add_argument("--ignore_times", type=str2bool, default=True,
                    help="Ignore the loss due to predicting the time.")
        parser.add_argument("--test_dir_name", type=str, default=None,
                                        help="directory for test data (optional, uses --dir_name if not specified)")
        # Multi-ticker training support
        parser.add_argument("--tickers", type=str, default=None,
                                        help="Comma-separated ticker list for multi-asset training (e.g. GOOG,AAPL,NVDA)")
        parser.add_argument("--uniform_ticker_batches", type=str2bool, default=False,
                                        help="Cycle tickers uniformly with one ticker per global batch")
        parser.add_argument("--max_train_batches", type=int, default=0,
                                        help="Finite micro-batch budget used by the uniform ticker sampler")
        parser.add_argument("--data_root", type=str, default=None,
                                        help="Root directory containing per-ticker subdirectories")
        parser.add_argument("--train_date_range", type=str, default=None,
                                        help="Inclusive date range for training data (YYYY-MM-DD,YYYY-MM-DD)")
        parser.add_argument("--test_date_range", type=str, default=None,
                                        help="Inclusive date range for test data (YYYY-MM-DD,YYYY-MM-DD)")
        parser.add_argument("--debug_overfit", type=str2bool, default=False,
                                help="Runs the training loop in overfit mode on a single batch of data. Validation and testing are from the same set. ")
        parser.add_argument("--log_ce_tables", type=str2bool, default=False,
                                help="Logs the CE values on a per token level to wandb. Memory intensive.")
        parser.add_argument("--hierarchical", type=str2bool, default=True,
                                help="Use hierarchical AllReduce via shard_map with 2D mesh (nodes, gpus). "
                                          "Decomposes flat AllReduce(N*4) into pmean(gpus)+pmean(nodes).")
        parser.add_argument("--tp_size", type=int, default=1,
                                help="Tensor parallelism: split SSM heads across this many intra-node GPUs. "
                                          "1=pure DP (default). 4=head-parallel TP within node, DP across nodes.")
        parser.add_argument("--local_steps_k", type=int, default=10,
                                help="Local Steps: each node trains independently for K steps, "
                                          "then params averaged via pmean('nodes'). 0=disabled (standard AllReduce). "
                                          "K>0 requires --hierarchical=True. Inner optimizer (Adam/AdamW) is unchanged.")
        parser.add_argument("--diloco_outer", type=str, default="none",
                                choices=["none", "nesterov"],
                                help="Outer-loop optimizer for local-SGD. 'none'=naive FedAvg (pmean params). "
                                          "'nesterov'=DiLoCo: pseudo-gradient (theta_anchor - theta_local) averaged across "
                                          "nodes, then Nesterov-momentum outer step. Requires --local_steps_k>0.")
        parser.add_argument("--diloco_outer_lr", type=float, default=0.7,
                                help="Outer learning rate for DiLoCo Nesterov (default 0.7, DiLoCo paper Table 1).")
        parser.add_argument("--diloco_outer_momentum", type=float, default=0.9,
                                help="Outer Nesterov momentum beta for DiLoCo (default 0.9, DiLoCo paper Table 1).")
        parser.add_argument("--grad_accum_steps", type=int, default=1,
                                help="Gradient accumulation: accumulate K micro-batches before AllReduce. "
                                          "Effective BSZ = micro_bsz * num_devices * process_count * K. "
                                          "Default 1 (no accumulation). Mutually exclusive with local_steps_k>0.")
        parser.add_argument("--no_validation", action="store_true",
                                help="Skip all validation during training. For scaling law experiments "
                                          "where validation is done separately post-training.")
        parser.add_argument("--skip_test_eval", action="store_true",
                                help="Skip post-training per-ticker test evaluation. For FLOPs profiling "
                                          "runs where eval would pollute dmon-measured FLOPs/tok. Default: False.")
        parser.add_argument("--val_split", type=float, default=0.001,
                                help="Fraction of training files to hold out for validation (default: 0.001 = 0.1%%). "
                                          "Set to 0 to use all data for training.")

        args = parser.parse_args()

        # Mutual exclusion: grad_accum and local_steps cannot be used together
        if getattr(args, 'grad_accum_steps', 1) > 1 and getattr(args, 'local_steps_k', 0) > 0:
                parser.error("--grad_accum_steps>1 and --local_steps_k>0 are mutually exclusive. "
                                          "Use one or the other.")
        # DiLoCo Nesterov requires local-steps mode
        if getattr(args, 'diloco_outer', 'none') == 'nesterov' and getattr(args, 'local_steps_k', 0) < 2:
                parser.error("--diloco_outer=nesterov requires --local_steps_k>=2 "
                                          "(K=1 degenerates to Nesterov-AllReduce; K=0 is disabled).")

        # Post-parse: multi-ticker string args → list/tuple
        if args.tickers is not None:
                args.tickers = [t.strip() for t in args.tickers.split(',')]
        if args.train_date_range is not None:
                parts = args.train_date_range.split(',')
                assert len(parts) == 2, f"train_date_range must be YYYY-MM-DD,YYYY-MM-DD, got: {args.train_date_range}"
                args.train_date_range = (parts[0].strip(), parts[1].strip())
        if args.test_date_range is not None:
                parts = args.test_date_range.split(',')
                assert len(parts) == 2, f"test_date_range must be YYYY-MM-DD,YYYY-MM-DD, got: {args.test_date_range}"
                args.test_date_range = (parts[0].strip(), parts[1].strip())

        # === Multi-node distributed training ===
        import jax
        from jax.experimental import multihost_utils

        process_count = int(os.environ.get('SLURM_NNODES', '1'))
        is_distributed = process_count > 1

        if is_distributed:
                coordinator_address = os.environ.get('JAX_COORDINATOR_ADDRESS')
                if coordinator_address:
                        num_processes = int(os.environ.get('SLURM_NTASKS', process_count))
                        process_id = int(os.environ.get('SLURM_PROCID', '0'))
                        n_local_gpus = int(os.environ.get('GPUS_PER_NODE', '4'))
                        print(f"[*] Initializing JAX distributed: coord={coordinator_address}, "
                                    f"pid={process_id}/{num_processes}, local_gpus={n_local_gpus}")
                        jax.distributed.initialize(
                                coordinator_address=coordinator_address,
                                num_processes=num_processes,
                                process_id=process_id,
                                local_device_ids=list(range(n_local_gpus)),
                        )
                else:
                        print("[*] Initializing JAX distributed via SLURM auto-detection")
                        jax.distributed.initialize()

                process_index = jax.process_index()
                process_count = jax.process_count()
                args.num_devices = jax.local_device_count()
                print(f"[*] JAX distributed: rank {process_index}/{process_count}, "
                            f"{args.num_devices} local GPUs, {jax.device_count()} total GPUs")

                # Sync barrier: wait for all nodes before proceeding
                print(f"[*] Sync barrier: waiting for all {process_count} nodes...")
                import time
                sync_start = time.time()
                multihost_utils.sync_global_devices("jax_distributed_init")
                print(f"[*] All {process_count} nodes synchronized (took {time.time() - sync_start:.2f}s)")
        else:
                process_index = 0
                process_count = 1

        args.is_distributed = is_distributed
        args.process_index = process_index
        args.process_count = process_count

        import torch
        torch.multiprocessing.set_start_method('spawn')

        from base_model.training.train import train
        train(args)

        # Clean shutdown for multi-node
        if is_distributed:
                print(f"[*] Process {process_index}: final sync...")
                multihost_utils.sync_global_devices("end-of-train")
                print(f"[*] Process {process_index}: shutting down JAX distributed...")
                jax.distributed.shutdown()
                print(f"[*] Process {process_index}: shutdown complete")
