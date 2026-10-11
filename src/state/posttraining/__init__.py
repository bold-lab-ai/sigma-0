"""Post-training for the state model: four ways to teach "the cup must not shatter".

All four consume the same pretrained velocity model through ``flow.ModelFn``
and differ only in where the preference signal enters:

    refl.py       differentiable reward, gradient through the one-jump x0_hat
    flow_grpo.py  black-box reward, group advantage on per-step SDE likelihoods
    awm.py        same group data as GRPO, advantage-WEIGHTED pretraining loss
    ram.py        same group data, reward-ADJUSTED regression target
    dmd.py        no reward: distribution matching against a frozen teacher
                  (DMDRTrainer combines it with a reward term)

Picking one (the summary table lives in README.md next to this file):
a working differentiable reward model -> ReFL; only rules or black-box scores
-> Flow-GRPO if the per-step machinery is worth it, AWM/RAM to stay on the
pretraining objective; a strong teacher and a few-step student -> DMD, plus a
reward term (DMDR) because matching alone cannot create a constraint the
teacher does not carry.
"""

from .awm import AWMTrainer
from .dmd import DMDRTrainer, DMDTrainer
from .flow_grpo import FlowGRPOTrainer
from .ram import RAMTrainer
from .refl import ReFLTrainer

__all__ = [
    "AWMTrainer",
    "DMDRTrainer",
    "DMDTrainer",
    "FlowGRPOTrainer",
    "RAMTrainer",
    "ReFLTrainer",
]
