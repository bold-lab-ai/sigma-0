"""Fused Pallas SSD versus the XLA reference on one GPU: parity, then speed.

Run inside a Slurm job (see ssd_kernel_bench.sbatch). Prints one JSON record
per configuration on stdout so the sbatch can keep the raw log and the
results database gets the numbers with their job id.

    python run/base_model/bench/ssd_kernel_bench.py --L 13000 --nh 32 --N 128 --P 64 --CS 64

Parity: forward and every input gradient of the fused path against the
reference at f32/HIGHEST (tolerance 1e-4 relative at this length; the
reference itself accumulates over 200 chunks). Speed: median of timed
iterations after warm-up, forward alone and forward+backward, for
xla/highest, xla/tf32, pallas/highest, pallas/tf32, pallas/bf16.

The fused path takes B, C and the rotation angle rates once for all heads
(as the model has them); the reference takes them repeated over heads, and
its Q, K and angle gradients are summed over heads for the parity check.
"""
import argparse
import json
import os
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np


def _inputs(L, nh, N, P, seed=0):
    """Fused-path arguments: Q, K (L, N) and angles (L, 1, N // 4) shared over heads."""
    k = jax.random.split(jax.random.PRNGKey(seed), 11)
    Q = jax.random.normal(k[0], (L, N))
    K = jax.random.normal(k[1], (L, N))
    V = jax.random.normal(k[2], (L, nh, P))
    Z = jax.random.normal(k[3], (L, nh, P))
    DT = jax.nn.softplus(jax.random.normal(k[4], (nh, L)) - 2.0)
    A = -jax.nn.softplus(jax.random.normal(k[5], (nh, L)))
    ADT = A * DT
    Trap = jax.random.normal(k[6], (nh, L))
    angles = jax.random.normal(k[7], (L, 1, N // 4))
    Q_bias = 0.1 * jax.random.normal(k[8], (nh, N))
    K_bias = 0.1 * jax.random.normal(k[9], (nh, N))
    D = jax.random.normal(k[10], (nh,))
    return (Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z)


def _ref_args(args):
    """The reference's view of the same problem: Q, K, angles repeated over heads."""
    Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z = args
    nh = V.shape[1]
    rep = lambda x: jnp.repeat(x, nh, axis=1)
    return (rep(Q[:, None, :]), rep(K[:, None, :]), V, ADT, DT, Trap, rep(angles),
            Q_bias, K_bias, D, Z)


def _ref_grads_to_shared(g):
    g = list(g)
    g[0] = jnp.sum(g[0], axis=1)
    g[1] = jnp.sum(g[1], axis=1)
    g[6] = jnp.sum(g[6], axis=1, keepdims=True)
    return tuple(g)


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def _timeit(fn, args, warmup, iters):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=13000)
    ap.add_argument("--nh", type=int, default=32)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--CS", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--num_warps", type=int, default=4)
    ap.add_argument("--num_stages", type=int, default=2)
    ap.add_argument("--skip_parity", action="store_true")
    ap.add_argument("--decompose", action="store_true",
                    help="also time phases 1-2 (XLA preprocessing) and the fused "
                         "kernels on already-chunked inputs, separately")
    ap.add_argument("--vmap_batch", type=int, default=0,
                    help="time jax.vmap over this many sequences on one device (how the "
                         "model batches); 0 skips")
    ap.add_argument("--shard", type=int, default=0,
                    help="time jit(vmap(...)) with a batch of this size sharded over that "
                         "many devices on a 1-D 'data' mesh (the single-node training "
                         "path); reports per-device time. 0 skips")
    a = ap.parse_args()

    from base_model.models.mamba3_jax import mamba3_ssd_chunked_jax
    from base_model.models.mamba3_pallas import mamba3_ssd_chunked_pallas

    if jax.default_backend() != "gpu":
        # A silent CPU fallback (the CUDA plugin failing to load cuSPARSE, for
        # instance) would time the interpreter and call it a kernel result.
        raise SystemExit(
            f"ssd_kernel_bench needs a GPU backend, got {jax.default_backend()!r}. "
            "Load the CUDA libraries the training launcher uses before running.")

    print(json.dumps({"backend": jax.default_backend(), "device": str(jax.devices()[0]),
                      "jax": jax.__version__, "shape": vars(a)}))
    args = _inputs(a.L, a.nh, a.N, a.P)
    cot = jax.random.normal(jax.random.PRNGKey(7), (a.L, a.nh, a.P))

    def make(impl, precision, cdt):
        """(fwd, fwd+bwd) on the fused-path arguments; the xla arm repeats B, C
        and the angles over heads inside the jitted function, as the model does
        on that path, so its timing includes that traffic."""
        os.environ["MAMBA3_CONTRACTION_PRECISION"] = precision
        if impl == "xla":
            f = lambda *x: mamba3_ssd_chunked_jax(*_ref_args(x), chunk_size=a.CS)
        else:
            f = lambda *x: mamba3_ssd_chunked_pallas(
                *x, chunk_size=a.CS, compute_dtype=jnp.dtype(cdt), interpret=False,
                num_warps=a.num_warps, num_stages=a.num_stages)
        fwd = jax.jit(f)
        fwdbwd = jax.jit(jax.value_and_grad(lambda *x: jnp.sum(f(*x) * cot),
                                            argnums=tuple(range(len(args)))))
        return fwd, fwdbwd

    configs = [("xla", "highest", "float32"), ("xla", "default", "float32"),
               ("pallas", "highest", "float32"), ("pallas", "default", "float32"),
               ("pallas", "default", "bfloat16")]

    if not a.skip_parity:
        ref_fwd, ref_fb = make("xla", "highest", "float32")
        y_ref = jax.block_until_ready(ref_fwd(*args))
        _, g_ref = ref_fb(*args)          # already w.r.t. the shared Q, K, angles (jit'd repeat)
        for impl, prec, cdt in configs[2:]:
            fwd, fb = make(impl, prec, cdt)
            y = jax.block_until_ready(fwd(*args))
            _, g = fb(*args)
            rec = {"parity": f"{impl}/{prec}/{cdt}", "fwd_rel": _rel(y, y_ref),
                   "grad_rel_max": max(_rel(x, r) for x, r in zip(g, g_ref)),
                   "finite": bool(np.isfinite(np.asarray(y)).all())}
            print(json.dumps(rec), flush=True)

    for impl, prec, cdt in configs:
        fwd, fb = make(impl, prec, cdt)
        t_f, tmin_f = _timeit(fwd, args, a.warmup, a.iters)
        t_fb, tmin_fb = _timeit(fb, args, a.warmup, a.iters)
        print(json.dumps({"speed": f"{impl}/{prec}/{cdt}", "fwd_ms": round(1e3 * t_f, 3),
                          "fwd_min_ms": round(1e3 * tmin_f, 3),
                          "fwdbwd_ms": round(1e3 * t_fb, 3),
                          "fwdbwd_min_ms": round(1e3 * tmin_fb, 3)}), flush=True)

    def _batched_fns(impl, prec, cdt, B, mesh=None):
        """jit(vmap(ssd)) over a leading batch axis, optionally sharded on 'data'."""
        from jax.sharding import NamedSharding, PartitionSpec as P
        os.environ["MAMBA3_CONTRACTION_PRECISION"] = prec
        if impl == "xla":
            f = lambda *x: mamba3_ssd_chunked_jax(*x, chunk_size=a.CS)
        else:
            f = lambda *x: mamba3_ssd_chunked_pallas(
                *x, chunk_size=a.CS, compute_dtype=jnp.dtype(cdt), interpret=False,
                num_warps=a.num_warps, num_stages=a.num_stages)
        if impl == "xla":
            f = lambda *x: mamba3_ssd_chunked_jax(*_ref_args(x), chunk_size=a.CS)
        vf = jax.vmap(f)
        batched = tuple(jnp.stack([x] * B) for x in args)
        cotb = jnp.stack([cot] * B)
        loss = lambda *x: jnp.sum(vf(*x) * cotb)
        if mesh is None:
            fwd = jax.jit(vf)
            fb = jax.jit(jax.value_and_grad(loss, argnums=tuple(range(len(args)))))
        else:
            shd = NamedSharding(mesh, P("data"))
            batched = tuple(jax.device_put(x, shd) for x in batched)
            cotb = jax.device_put(cotb, shd)
            fwd = jax.jit(vf, in_shardings=(shd,) * len(args), out_shardings=shd)
            fb = jax.jit(jax.value_and_grad(lambda *x: jnp.sum(vf(*x) * cotb),
                                            argnums=tuple(range(len(args)))),
                         in_shardings=(shd,) * len(args))
        return fwd, fb, batched

    if a.vmap_batch:
        B = a.vmap_batch
        for impl, prec, cdt in [("xla", "default", "float32"), ("pallas", "default", "float32"),
                                ("pallas", "default", "bfloat16")]:
            fwd, fb, batched = _batched_fns(impl, prec, cdt, B)
            t_f, _ = _timeit(fwd, batched, a.warmup, a.iters)
            t_fb, _ = _timeit(fb, batched, a.warmup, a.iters)
            print(json.dumps({"vmap": f"{impl}/{prec}/{cdt}", "batch": B,
                              "fwd_ms_per_seq": round(1e3 * t_f / B, 3),
                              "fwdbwd_ms_per_seq": round(1e3 * t_fb / B, 3)}), flush=True)

    if a.shard:
        from jax.sharding import Mesh
        devs = jax.devices()[:a.shard]
        if len(devs) < a.shard:
            print(json.dumps({"shard": "skipped", "reason": f"only {len(devs)} devices visible"}))
        else:
            mesh = Mesh(np.array(devs), axis_names=("data",))
            B = a.shard
            for impl, prec, cdt in [("xla", "default", "float32"), ("pallas", "default", "float32")]:
                fwd, fb, batched = _batched_fns(impl, prec, cdt, B, mesh=mesh)
                t_f, _ = _timeit(fwd, batched, a.warmup, a.iters)
                t_fb, _ = _timeit(fb, batched, a.warmup, a.iters)
                # One sequence per device: the wall time IS the per-device time. If the
                # kernel were replicated by the partitioner it would read ~B x the
                # single-device figure from --vmap_batch 1.
                print(json.dumps({"shard": f"{impl}/{prec}/{cdt}", "devices": B, "seq_per_device": 1,
                                  "fwd_ms": round(1e3 * t_f, 3), "fwdbwd_ms": round(1e3 * t_fb, 3)}), flush=True)

    if a.decompose:
        from base_model.models import mamba3_pallas as mp
        from base_model.models.mamba3_jax import preprocess_and_chunk
        Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z = args

        # (a) The reference's phases 1-2 on repeated inputs: what the fusion removes.
        def pre(*x):
            Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z = _ref_args(x)
            return preprocess_and_chunk(Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, Z,
                                        chunk_size=a.CS)[:6]
        pre_j = jax.jit(pre)
        jax.block_until_ready(pre_j(*args))
        t_pre, _ = _timeit(pre_j, args, a.warmup, a.iters)
        pre_fb = jax.jit(jax.value_and_grad(lambda *x: sum(jnp.sum(t) for t in pre(*x)),
                                            argnums=tuple(range(len(args)))))
        t_pre_fb, _ = _timeit(pre_fb, args, a.warmup, a.iters)
        print(json.dumps({"decompose": "reference phases 1-2 on per-head inputs (XLA, incl. the head repeat)",
                          "fwd_ms": round(1e3 * t_pre, 3), "fwdbwd_ms": round(1e3 * t_pre_fb, 3)}), flush=True)

        for prec, cdt in [("default", "float32"), ("default", "bfloat16")]:
            os.environ["MAMBA3_CONTRACTION_PRECISION"] = prec
            kernels = mp._build(cdt, mp._dot_precision().name, False, a.num_warps, a.num_stages)
            phase1, chunk_state, chunk_scan = kernels
            io = jnp.dtype(cdt)
            contraction_prec = mp.mamba3_contraction_precision()

            # (b) The fused path's JAX side: angle cumsum, weights, padding, chunked layouts.
            def prep_only(*x):
                prep, L, L_pad = mp.prepare_fused_inputs(*x, chunk_size=a.CS, io_dtype=io)
                return prep
            prep_j = jax.jit(prep_only)
            prep = jax.block_until_ready(prep_j(*args))
            t_prep, _ = _timeit(prep_j, args, a.warmup, a.iters)
            prep_fb = jax.jit(jax.value_and_grad(
                lambda *x: sum(jnp.sum(t.astype(jnp.float32)) for t in prep_only(*x).values()),
                argnums=tuple(range(len(args)))))
            t_prep_fb, _ = _timeit(prep_fb, args, a.warmup, a.iters)

            # (c) All kernels and phase 5 on prepared inputs.
            _, L0, L_pad0 = mp.prepare_fused_inputs(*args, chunk_size=a.CS, io_dtype=io)
            keys = sorted(prep.keys())
            def kernels_fn(*vals):
                p = dict(zip(keys, vals))
                return mp.run_fused_kernels(p, L0, L_pad0, kernels=kernels, precision=contraction_prec,
                                            chunk_size=a.CS, io_dtype=io)
            pvals = tuple(prep[k] for k in keys)
            cot_k = jax.random.normal(jax.random.PRNGKey(8), (L0, a.nh, a.P))
            k_fwd = jax.jit(kernels_fn)
            k_fb = jax.jit(jax.value_and_grad(
                lambda *v: jnp.sum(kernels_fn(*v).astype(jnp.float32) * cot_k),
                argnums=tuple(range(len(keys)))))
            t_k, _ = _timeit(k_fwd, pvals, a.warmup, a.iters)
            t_kfb, _ = _timeit(k_fb, pvals, a.warmup, a.iters)

            # (d) The phase-1 kernel alone, forward and forward+backward.
            p1_args = (prep["Cp"], prep["Bp"], prep["qbias"], prep["kbias"], prep["ang"], prep["scale"])
            def p1(*v):
                return phase1(*v, a.CS, io.name)
            p1_fwd = jax.jit(p1)
            outs = jax.block_until_ready(p1_fwd(*p1_args))
            cots = tuple(jax.random.normal(jax.random.PRNGKey(3 + i), o.shape, jnp.float32) for i, o in enumerate(outs))
            p1_fb = jax.jit(jax.grad(lambda *v: sum(jnp.sum(o.astype(jnp.float32) * c) for o, c in zip(p1(*v), cots)),
                                     argnums=tuple(range(len(p1_args)))))
            t_p1, _ = _timeit(p1_fwd, p1_args, a.warmup, a.iters)
            t_p1fb, _ = _timeit(p1_fb, p1_args, a.warmup, a.iters)

            # (e) chunk_state + state passing, and state passing alone (forward).
            passing = (mp.state_passing_blocked if mp.state_passing_mode() == "blocked"
                       else mp.state_passing_jax)
            K_c = outs[1]
            def state_only(K_c, V_c, A_cumsum):
                states = chunk_state(K_c, V_c, A_cumsum)
                return passing(states, A_cumsum[:, :, -1], contraction_prec)
            s_args = (K_c, prep["V_c"], prep["A_cumsum"])
            t_s, _ = _timeit(jax.jit(state_only), s_args, a.warmup, a.iters)
            st0 = jax.block_until_ready(jax.jit(chunk_state)(*s_args))
            p_fwd = jax.jit(lambda s, e: passing(s, e, contraction_prec))
            t_p, _ = _timeit(p_fwd, (st0, prep["A_cumsum"][:, :, -1]), a.warmup, a.iters)
            print(json.dumps({"decompose": f"fused path ({cdt} tiles, state passing {mp.state_passing_mode()})",
                              "jax_side_fwd_ms": round(1e3 * t_prep, 3),
                              "jax_side_fwdbwd_ms": round(1e3 * t_prep_fb, 3),
                              "phase1_kernel_fwd_ms": round(1e3 * t_p1, 3),
                              "phase1_kernel_fwdbwd_ms": round(1e3 * t_p1fb, 3),
                              "all_kernels+phase5_fwd_ms": round(1e3 * t_k, 3),
                              "all_kernels+phase5_fwdbwd_ms": round(1e3 * t_kfb, 3),
                              "chunk_state+state_passing_fwd_ms": round(1e3 * t_s, 3),
                              "state_passing_only_fwd_ms": round(1e3 * t_p, 3)}), flush=True)


if __name__ == "__main__":
    main()
