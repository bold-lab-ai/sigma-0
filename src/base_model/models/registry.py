"""Strict registry for the sequence backbones used by the LOB model.

The archived experiments grew two independent selectors (``model_type`` and
``ssm_type``) and several branches silently fell through to another model.
This module is the only place where an architecture name is resolved and
turned into a layer factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Mapping


DEFAULT_ARCHITECTURE = "mamba3"

# Read from the active encoding rather than frozen at 26: lob.encoding is a
# runtime selector, and an NSA block has to match whatever message length is in
# force. Kept as module attributes so existing `registry.TOKENS_PER_MESSAGE`
# readers keep working, but resolved on access.
def __getattr__(name):
    if name in ("TOKEN_MODE", "TOKENS_PER_MESSAGE"):
        from base_model.encoding.encoding import get_token_mode, Message_Tokenizer
        return get_token_mode() if name == "TOKEN_MODE" else int(Message_Tokenizer.MSG_LEN)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class BackboneDefinition:
    name: str
    wrapper_policy: str
    supports_prefill: bool
    supports_stepwise_inference: bool = True
    # Whether the stack contains attention layers of any kind -- every layer
    # (transformer, nsa) or only a few positions (hybrids). Deliberately a
    # declared fact, not derived from wrapper_policy: a hybrid is wrapped
    # externally because most of its layers are recurrent, yet its attention
    # positions are very much real.
    has_attention: bool = False


@dataclass(frozen=True)
class BackboneBuild:
    definition: BackboneDefinition
    layer_factory: Callable[..., Any]
    carry_kwargs: Mapping[str, Any]
    attention_config: Mapping[str, Any] | None = None
    # Hybrid stacks carry a second factory for the few positions where a global
    # mixer replaces the recurrent one. Homogeneous backbones leave both unset.
    attn_layer_factory: Callable[..., Any] | None = None
    attn_layers: tuple = ()

    @property
    def is_attention(self) -> bool:
        return self.definition.wrapper_policy == "internal"

    @property
    def is_hybrid(self) -> bool:
        return self.attn_layer_factory is not None


_DEFINITIONS = {
    "mamba3": BackboneDefinition("mamba3", "external", False),
    "s5": BackboneDefinition("s5", "external", False),
    "gdn": BackboneDefinition("gdn", "external", False),
    "kda": BackboneDefinition("kda", "external", False),
    "transformer": BackboneDefinition("transformer", "internal", True, has_attention=True),
    "nsa": BackboneDefinition("nsa", "internal", True, has_attention=True),
    # Nemotron-H style: a mamba3 trunk with a few global attention positions.
    # The stack is externally wrapped because most layers are recurrent; the
    # attention positions are self-wrapping and SequenceLayer detects them.
    "hybrid_mamba3": BackboneDefinition("hybrid_mamba3", "external", False, has_attention=True),
}


# --- Nemotron-H attention placement -----------------------------------------
# Nemotron Nano 9B v2 is the only calibration point we have: 4 attention layers
# among 31 sequence-mixing layers, at relative depths {0.250, 0.375, 0.536,
# 0.696}. Note the denominator: 25 of its 56 layers are MLP-only and mix no
# tokens at all, so scaling by 4/56 would put attention roughly twice too
# sparse. Evaluating the rule back at L=31 reproduces Nemotron's own positions
# to within one layer, which is the only self-check available.
NEMOTRON_ATTENTION_FRACTION = 4 / 31
NEMOTRON_DEPTH_BAND = (0.26, 0.71)


def nemotron_attention_indices(n_layers: int) -> tuple[int, ...]:
    """0-based positions of the attention layers in an n_layers trunk."""
    k = max(1, round(NEMOTRON_ATTENTION_FRACTION * n_layers))
    if k == 1:
        depths = [0.48]  # centroid of Nemotron's four relative depths
    else:
        lo, hi = NEMOTRON_DEPTH_BAND
        depths = [lo + (hi - lo) * i / (k - 1) for i in range(k)]
    # Never first, never last: with no positional encoding of its own, an
    # attention layer needs the recurrent layers below it to establish order,
    # and Nemotron keeps 8 recurrent layers after its last attention too.
    idx = {min(max(round(d * n_layers), 2), n_layers - 2) for d in depths}
    return tuple(sorted(idx))


def architecture_has_attention(name: str | None) -> bool:
    """True when this architecture builds attention layers of any kind.

    Reads the declared has_attention field rather than deriving from
    wrapper_policy or matching a literal name list. Any caller that asks "does
    this need a KV cache" or "does this need attention config" must ask this.
    Three separate defects in this repo have come from a check keyed on an
    exact architecture name that a newly registered backbone fell through --
    and the first version of this very function kept a name set beside a
    comment arguing against name sets.
    """
    definition = _DEFINITIONS.get(name or "")
    return definition is not None and definition.has_attention


def available_architectures() -> tuple[str, ...]:
    """Return stable CLI choices, with the default first."""
    return tuple(_DEFINITIONS)


def get_definition(name: str) -> BackboneDefinition:
    try:
        return _DEFINITIONS[name]
    except KeyError as exc:
        if name == "mamba2":
            raise ValueError(
                "architecture 'mamba2' is intentionally not part of this model zoo; "
                "use 'mamba3' for new experiments"
            ) from exc
        choices = ", ".join(available_architectures())
        raise ValueError(f"unknown architecture {name!r}; choose one of: {choices}") from exc


def resolve_architecture(
    architecture: str | None = None,
    *,
    ssm_type: str | None = None,
    model_type: str | None = None,
) -> str:
    """Resolve the canonical selector and reject ambiguous legacy settings."""
    architecture = architecture.lower() if architecture else None
    ssm_type = ssm_type.lower() if ssm_type else None
    model_type = model_type.lower() if model_type else None

    if model_type not in {None, "ssm", *_DEFINITIONS}:
        # Unknown historical metadata must never turn into the Mamba3 default
        # or be hidden by a valid ssm_type.
        get_definition(model_type)

    # Historical attention checkpoints stored model_type=transformer|nsa and
    # the parser-default ssm_type=s5.  Conversely, recurrent checkpoints often
    # stored model_type=s5 alongside their real ssm_type.  Interpret those
    # sentinel combinations before applying the strict conflict check.
    legacy: str | None
    if model_type in {"transformer", "nsa"}:
        if ssm_type not in {None, "s5", model_type}:
            raise ValueError(
                f"conflicting legacy selectors: model_type={model_type!r}, "
                f"ssm_type={ssm_type!r}"
            )
        legacy = model_type
    elif ssm_type:
        legacy = ssm_type
    elif model_type in _DEFINITIONS:
        legacy = model_type
    else:
        legacy = None

    candidates: list[tuple[str, str]] = []
    if architecture:
        candidates.append(("architecture", architecture))
    if legacy:
        candidates.append(("legacy selector", legacy))

    if not candidates:
        return DEFAULT_ARCHITECTURE

    names = {value for _, value in candidates}
    if len(names) != 1:
        rendered = ", ".join(f"{key}={value!r}" for key, value in candidates)
        raise ValueError(f"conflicting architecture selectors: {rendered}")

    selected = candidates[0][1]
    get_definition(selected)
    return selected


def architecture_from_args(args: Any) -> str:
    return resolve_architecture(
        getattr(args, "architecture", None),
        ssm_type=getattr(args, "ssm_type", None),
        model_type=getattr(args, "model_type", None),
    )


def attention_config_for_width(
    backbone: BackboneBuild,
    width: int,
    *,
    max_cache_len: int | None = None,
) -> dict[str, Any]:
    """Return cache metadata matching an attention layer's effective width.

    The LOB book pre-encoder runs at ``d_book`` rather than ``d_model``.  Both
    archived attention implementations reduce the head count until it divides
    that width, so carry creation must apply the identical rule.
    """
    # Keyed on whether an attention cache exists, not on whether the whole
    # backbone is attention. A hybrid stack is externally wrapped, so
    # is_attention is False for it, yet its attention positions still need a
    # KV cache sized the same way.
    if backbone.attention_config is None:
        raise ValueError(
            f"architecture {backbone.definition.name!r} does not use an attention cache"
        )
    config = dict(backbone.attention_config)
    n_heads = int(config["n_heads"])
    width = int(width)
    while n_heads > 1 and width % n_heads:
        n_heads -= 1
    config["n_heads"] = n_heads
    config["head_dim"] = width // n_heads
    if max_cache_len is not None:
        config["max_cache_len"] = int(max_cache_len)
    return config


def recurrent_carry_for_width(
    carry_kwargs: Mapping[str, Any],
    width: int,
) -> dict[str, Any]:
    """Resize recurrent carry metadata exactly as the layer does in setup."""
    config = dict(carry_kwargs)
    width = int(width)
    name = config.get("ssm_type")
    if name == "mamba3":
        expand = int(config["expand"])
        requested_headdim = int(config.get("requested_headdim", config["headdim"]))
        d_inner_target = expand * width
        headdim = min(requested_headdim, d_inner_target)
        n_heads = max(1, d_inner_target // headdim)
        tp_size = int(config.get("tp_size", 1))
        effective_tp = tp_size if n_heads % tp_size == 0 else 1
        config["headdim"] = headdim
        config["n_heads"] = n_heads // effective_tp
    elif name in {"gdn", "kda"}:
        requested_heads = int(config.get("requested_num_heads", config["num_heads"]))
        requested_head_dim = int(config.get("requested_head_dim", config["head_dim"]))
        head_dim = min(requested_head_dim, width)
        num_heads = min(requested_heads, max(1, width // requested_head_dim))
        config["num_heads"] = num_heads
        config["head_dim"] = head_dim
        config["head_v_dim"] = head_dim * int(config["expand_v"])
    return config


# Architectures whose layer builders take the compute dtype from --dtype.
# build_backbone refuses a non-float32 dtype for any other name.
DTYPE_AWARE_ARCHITECTURES = frozenset({"transformer", "nsa", "mamba3", "hybrid_mamba3"})


def _compute_dtype(args):
    import jax.numpy as jnp
    return {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }.get(str(getattr(args, "dtype", "float32")), jnp.float32)


def build_backbone(args: Any, *, print_shapes: bool = False) -> BackboneBuild:
    """Build the selected legacy layer behind one common descriptor."""
    name = architecture_from_args(args)
    definition = get_definition(name)
    H = int(args.d_model)

    # --dtype is consumed by the transformer and NSA builders only. For every
    # other architecture the flag used to be accepted, forwarded by the batch
    # script, written to the wandb config and read by nothing, so a run tagged
    # bfloat16 trained in float32 and a whole measurement campaign credited a
    # switch that never fired (PR #32, 2026-09). A knob that reaches no code is
    # refused rather than ignored until the architecture actually honours it.
    requested_dtype = str(getattr(args, "dtype", "float32"))
    if requested_dtype != "float32" and name not in DTYPE_AWARE_ARCHITECTURES:
        raise ValueError(
            f"--dtype={requested_dtype} has no effect on architecture '{name}': "
            f"only {sorted(DTYPE_AWARE_ARCHITECTURES)} read it. Pass float32 "
            "(the value this architecture always ran at) or pick one of those."
        )

    attn_layer_factory = None
    attn_layers: tuple = ()
    attention_config = None

    if name in {"mamba3", "hybrid_mamba3"}:
        from base_model.models.mamba3 import init_Mamba3SSM

        d_state = int(getattr(args, "mamba3_d_state", 128))
        expand = int(getattr(args, "mamba3_expand", 2))
        headdim = int(getattr(args, "mamba3_headdim", 64))
        chunk_size = int(getattr(args, "mamba3_chunk_size", 64))
        rope_fraction = float(getattr(args, "mamba3_rope_fraction", 0.5))
        tp_size = int(getattr(args, "tp_size", 1))
        layer_factory = init_Mamba3SSM(
            H=H,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            rope_fraction=rope_fraction,
            use_triton=bool(getattr(args, "mamba3_use_triton", False)),
            use_cuda=bool(getattr(args, "mamba3_use_cuda", False)),
            tp_size=tp_size,
            dtype=_compute_dtype(args),
        )
        carry = recurrent_carry_for_width({
            # Always "mamba3": the recurrent positions of a hybrid stack build
            # exactly the same carry, and SequenceLayer dispatches on this name.
            "ssm_type": "mamba3",
            "n_heads": 0,
            "headdim": headdim,
            "d_state": d_state,
            "num_rope_angles": int(d_state * rope_fraction / 2),
            "expand": expand,
            "tp_size": tp_size,
            "requested_headdim": headdim,
        }, H)

        if name == "hybrid_mamba3":
            from base_model.models.transformer import init_TransformerBlock

            n_layers = int(getattr(args, "n_layers", 6))
            override = getattr(args, "hybrid_attn_layers", None)
            attn_layers = (
                tuple(int(v) for v in str(override).split(",") if v.strip() != "")
                if override else nemotron_attention_indices(n_layers)
            )
            bad = [i for i in attn_layers if not 0 <= i < n_layers]
            if bad:
                raise ValueError(
                    f"hybrid_attn_layers={attn_layers} out of range for n_layers={n_layers}"
                )
            # Stamp the resolved positions back into args so they travel in the
            # checkpointed config. A checkpoint has to describe itself: if the
            # Nemotron placement rule above is ever edited, existing hybrids
            # must keep loading with the positions they were trained at, and
            # re-derivation stays reserved for builds that start from nothing.
            args.hybrid_attn_layers = ",".join(str(i) for i in attn_layers)

            n_heads = int(getattr(args, "hybrid_attn_heads", 10))
            while n_heads > 1 and H % n_heads:
                n_heads -= 1
            head_dim = H // n_heads
            # The Pallas causal kernel needs head_dim <= 256 and divisible by 8;
            # anything else silently falls back to a materialised LxL matrix,
            # which at 13k tokens is not a fallback but an out-of-memory crash.
            use_flash = bool(getattr(args, "hybrid_attn_flash", True))
            if use_flash and (head_dim > 256 or head_dim % 8):
                raise ValueError(
                    f"hybrid attention head_dim={head_dim} (d_model={H}/n_heads={n_heads}) "
                    "cannot use the Pallas kernel; choose n_heads so head_dim is <=256 "
                    "and a multiple of 8, or set hybrid_attn_flash=False"
                )
            d_ff = int(getattr(args, "hybrid_attn_d_ff", 0)) or 4 * H
            max_cache_len = int(getattr(args, "attention_max_cache_len", 26000))
            attn_layer_factory = init_TransformerBlock(
                H=H,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=float(args.p_dropout),
                dtype=_compute_dtype(args),
                use_flash=use_flash,
                remat=bool(getattr(args, "remat", False)),
                max_cache_len=max_cache_len,
                # Nemotron runs its attention layers with no positional encoding
                # at all; here the recurrent layers below already rotate their
                # state with RoPE, so injecting a sinusoid into the residual
                # stream would add a second, unrelated notion of position.
                use_positional_encoding=bool(
                    getattr(args, "hybrid_attn_positional_encoding", False)
                ),
            )
            attention_config = {
                "model_type": "transformer",
                "n_heads": n_heads,
                "head_dim": head_dim,
                "max_cache_len": max_cache_len,
            }

    elif name in {"gdn", "kda"}:
        from base_model.models.gdn import init_GDN_SSM

        head_dim = int(getattr(args, "gdn_head_dim", 128))
        num_heads = getattr(args, "gdn_num_heads", None) or max(1, H // head_dim)
        expand_v = int(getattr(args, "gdn_expand_v", 2))
        use_conv = bool(getattr(args, "gdn_use_conv", True))
        layer_factory = init_GDN_SSM(
            H=H,
            num_heads=int(num_heads),
            head_dim=head_dim,
            expand_v=expand_v,
            chunk_size=int(getattr(args, "gdn_chunk_size", 64)),
            use_conv=use_conv,
            use_kda=name == "kda",
        )
        carry = recurrent_carry_for_width({
            "ssm_type": name,
            "num_heads": int(num_heads),
            "head_dim": head_dim,
            "head_v_dim": head_dim * expand_v,
            "expand_v": expand_v,
            "use_conv": use_conv,
            "requested_num_heads": int(num_heads),
            "requested_head_dim": head_dim,
        }, H)

    elif name == "s5":
        import numpy as np
        from jax.scipy.linalg import block_diag
        from base_model.models.ssm import init_S5SSM
        from base_model.models.ssm_init import make_DPLR_HiPPO

        state_size = int(args.ssm_size_base)
        blocks = int(args.blocks)
        if state_size % blocks:
            raise ValueError("ssm_size_base must be divisible by blocks for S5")
        block_size = state_size // blocks
        Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)
        if bool(args.conj_sym):
            block_size //= 2
            state_size //= 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T
        Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))
        layer_factory = init_S5SSM(
            H=H,
            P=state_size,
            Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=Vinv,
            C_init=args.C_init,
            discretization=args.discretization,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
            conj_sym=args.conj_sym,
            clip_eigs=args.clip_eigs,
            bidirectional=args.bidirectional,
        )
        carry = {"ssm_type": name}

    else:
        import jax.numpy as jnp

        n_heads = int(getattr(args, "n_heads", 16))
        if H % n_heads:
            raise ValueError(f"d_model={H} must be divisible by n_heads={n_heads}")
        d_ff = int(getattr(args, "d_ff", 0)) or 4 * H
        dtype = {
            "float32": jnp.float32,
            "bfloat16": jnp.bfloat16,
        }.get(str(getattr(args, "dtype", "float32")), jnp.float32)
        max_cache_len = int(getattr(args, "attention_max_cache_len", 26000))
        attention = {
            "model_type": name,
            "n_heads": n_heads,
            "head_dim": H // n_heads,
            "max_cache_len": max_cache_len,
            "dtype": dtype,
        }
        if name == "transformer":
            from base_model.models.transformer import init_TransformerBlock

            layer_factory = init_TransformerBlock(
                H=H,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=float(args.p_dropout),
                dtype=dtype,
                use_flash=bool(getattr(args, "use_flash", False)),
                remat=bool(getattr(args, "remat", False)),
                max_cache_len=max_cache_len,
            )
        else:
            from base_model.models.nsa import init_NSABlock

            from base_model.encoding.encoding import Message_Tokenizer as _MT
            _msg_len = int(_MT.MSG_LEN)
            block_size = int(getattr(args, "block_size", _msg_len))
            if block_size != _msg_len:
                raise ValueError(
                    f"NSA block_size must equal the message length of the active "
                    f"encoding ({_msg_len}); got {block_size}"
                )
            layer_factory = init_NSABlock(
                H=H,
                n_heads=n_heads,
                d_ff=d_ff,
                block_size=block_size,
                block_counts=int(getattr(args, "block_counts", 16)),
                window_size=int(getattr(args, "window_size", 512)),
                dropout=float(args.p_dropout),
                dtype=dtype,
                remat=bool(getattr(args, "remat", False)),
                max_cache_len=max_cache_len,
                pe_type=str(getattr(args, "pe_type", "rope")),
                rope_base=float(getattr(args, "rope_base", 10000.0)),
            )
            attention = {**attention, "block_size": block_size}
        carry = {
            "ssm_type": name,
            "is_transformer": True,
            "transformer_config": attention,
        }
        return BackboneBuild(definition, layer_factory, carry, attention)

    if print_shapes:
        print(f"[backbone] architecture={name} wrapper={definition.wrapper_policy}")
        if attn_layers:
            print(f"[backbone] hybrid attention at trunk layers {attn_layers}")
    return BackboneBuild(
        definition, layer_factory, carry,
        attention_config=attention_config,
        attn_layer_factory=attn_layer_factory,
        attn_layers=attn_layers,
    )


def stamp_architecture(args: Any, architecture: str) -> None:
    """Normalize legacy consumers after resolving the canonical selector."""
    args.architecture = architecture
    args.ssm_type = architecture
    args.model_type = architecture if architecture in {"transformer", "nsa"} else "ssm"
