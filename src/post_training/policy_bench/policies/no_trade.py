"""Never trades: the zero line of every comparison."""

from __future__ import annotations


class NoTrade:
    def reset(self, episode) -> None:
        pass

    def step(self, obs) -> list:
        return []
