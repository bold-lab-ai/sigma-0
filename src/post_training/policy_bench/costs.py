"""Exchange fees per share, in LOBSTER units x shares like the cash of a Trace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNITS_PER_DOLLAR = 10_000


@dataclass(frozen=True)
class MakerTaker:
    """A rebate for each share filled as maker, a fee for each share taken."""
    rebate: float = 0.002     # $ per share
    fee: float = 0.003        # $ per share

    def __call__(self, fills: np.ndarray, taker_shares: float = 0.0) -> float:
        """Fees paid on fills (Trace.fills rows) plus `taker_shares` more taken, e.g. by a flatten."""
        maker = fills[:, 4] > 0
        shares_maker = float(fills[maker, 3].sum())
        shares_taker = float(fills[~maker, 3].sum()) + taker_shares
        return (shares_taker * self.fee - shares_maker * self.rebate) * UNITS_PER_DOLLAR


NO_COST = MakerTaker(rebate=0.0, fee=0.0)
