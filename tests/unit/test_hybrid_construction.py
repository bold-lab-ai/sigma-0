"""A hybrid must be buildable by the registry, and must describe itself.

Nothing in the suite ever constructed a hybrid; the per-position carry bug this
branch fixes was found in production, by hand, at generation time. The
construction below is the cheap test that would have caught it. The stamping
tests pin the companion rule from the same review: resolved attention placement
is written back into args at build time, so a checkpointed config keeps the
positions it was trained at even if the placement rule is edited afterwards --
the same defect class as every name-keyed check this registry campaigns
against, just wearing load-time clothes.
"""
import types

from base_model.models import registry
from base_model.models.registry import (
    attention_config_for_width,
    build_backbone,
    nemotron_attention_indices,
)
from base_model.models.seq_model import StackedEncoderModel

N_LAYERS, D_MODEL, SSM_SIZE, CACHE_LEN = 12, 64, 64, 96


def _hybrid_args(**over):
    base = dict(
        architecture="hybrid_mamba3",
        model_type="hybrid_mamba3",
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        ssm_size_base=SSM_SIZE,
        p_dropout=0.0,
        attention_max_cache_len=CACHE_LEN,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_nemotron_rule_places_two_layers_at_depth_12():
    assert nemotron_attention_indices(N_LAYERS) == (3, 9)


def test_nemotron_rule_reproduces_the_papers_positions_at_its_own_depth():
    # The only self-check the rule admits: at Nemotron-H's own depth it must
    # reproduce Nemotron-H's own placement. This pins the implementation half;
    # it says nothing about transfer across a 270x scale gap, and the PR body
    # says so explicitly.
    assert nemotron_attention_indices(31) == (8, 13, 17, 22)


def test_registry_builds_a_hybrid_and_stamps_the_placement():
    args = _hybrid_args()
    build = build_backbone(args)
    assert build.is_hybrid
    assert not build.is_attention  # externally wrapped: most layers recurrent
    assert build.definition.has_attention  # ...but attention is present
    assert build.attn_layers == (3, 9)
    assert build.attn_layer_factory is not None
    # The resolved placement is stamped back into args in the same format the
    # override accepts, so the checkpointed config round-trips through vars().
    assert args.hybrid_attn_layers == "3,9"


def test_a_stamped_config_survives_an_edited_placement_rule(monkeypatch):
    trained = _hybrid_args()
    build_backbone(trained)  # stamps "3,9" into the config that gets saved
    # The rule changes after the checkpoint was written...
    monkeypatch.setattr(registry, "NEMOTRON_ATTENTION_FRACTION", 8 / 31)
    moved = build_backbone(_hybrid_args()).attn_layers
    assert moved != (3, 9), "the edited rule should place differently"
    # ...and a load from the stamped config must keep the trained positions.
    loaded = _hybrid_args(hybrid_attn_layers=trained.hybrid_attn_layers)
    assert build_backbone(loaded).attn_layers == (3, 9)


def test_carries_are_kv_at_attention_positions_and_state_elsewhere():
    from jax.tree_util import tree_leaves, tree_structure

    args = _hybrid_args()
    build = build_backbone(args)
    carry_kwargs = dict(build.carry_kwargs)
    carry_kwargs["attn_layers"] = build.attn_layers
    carry_kwargs["attn_config"] = attention_config_for_width(
        build, D_MODEL, max_cache_len=CACHE_LEN)
    carries = StackedEncoderModel.initialize_carry(
        1, SSM_SIZE, N_LAYERS, **carry_kwargs)

    assert len(carries) == N_LAYERS
    attn, recurrent = carries[3], carries[0]
    # The two carry kinds must not be interchangeable: that is the exact
    # confusion that unpacked a four-tuple into three names in production.
    assert tree_structure(attn) != tree_structure(recurrent)
    assert tree_structure(carries[9]) == tree_structure(attn)
    for i in range(N_LAYERS):
        if i not in (3, 9):
            assert tree_structure(carries[i]) == tree_structure(recurrent), i
    # Attention positions carry a cache sized to the request; recurrent
    # positions carry fixed-size state with no cache axis anywhere.
    assert any(CACHE_LEN in getattr(leaf, "shape", ()) for leaf in tree_leaves(attn))
    for i in range(N_LAYERS):
        if i not in (3, 9):
            assert all(CACHE_LEN not in getattr(leaf, "shape", ())
                       for leaf in tree_leaves(carries[i])), i
