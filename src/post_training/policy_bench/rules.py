"""What a background message may do to the book, and what it does.

`check` is the legality test the sampler rejects on. A message is legal when
the book can execute it and it looks like something LOBSTER records:

    new order      size >= 1; does not cross the opposite best
    cancel/delete  QSLOT is not NA and names a background order at that
                   price; delete removes exactly the remaining size, cancel
                   removes less than it. QSLOT 99 means "slot 99 or later":
                   legal if any order there qualifies
    execution      levels walk the side from its best price outward without
                   skipping; inner levels are taken whole, the last one at
                   most whole

Agent orders take part in queue numbering and in executions (a background
execution fills whatever is at the front), but a background cancel or delete
can never name one.

Time is not checked here. The sampler writes t = t_prev + dt itself, and
`time_consistent` measures how often real data obeys that.
"""

from __future__ import annotations

from post_training.policy_bench.book import AGENT, BG, Book, side_of_direction
from post_training.policy_bench.codec import MAX_SIZE, SLOT_NA, Message
from post_training.policy_bench.nanosigma.newtok import N_QSLOT

LAST_SLOT = N_QSLOT - 1    # the token for "slot 99 or later"

REASONS = (
    "new_size_zero", "new_crosses",
    "no_level", "slot_na", "slot_out_of_range", "slot_agent", "delete_size", "cancel_size",
    "no_candidate",
    "exec_size_zero", "exec_not_best", "exec_skips_level", "exec_partial_inner", "exec_over_volume",
)


def _size_ok(evt: int, size: int, remaining: int) -> bool:
    if evt == 3:
        return size == min(remaining, MAX_SIZE)
    return 1 <= size < remaining


def slot_candidates(m: Message, book: Book) -> list[int]:
    """Slots a cancel or delete could mean: the named slot, or for QSLOT 99
    every slot from 99 on. Only background orders whose size fits qualify."""
    side = side_of_direction(m.d)
    price, size = m.levels[0]
    q = book.queue(side, price)
    slots = [m.qslot] if m.qslot < LAST_SLOT else range(LAST_SLOT, len(q))
    return [s for s in slots if s < len(q) and q[s].owner == BG and _size_ok(m.evt, size, q[s].size)]


def check(m: Message, book: Book) -> str | None:
    """None if `m` is legal in `book`, else the first rule it breaks."""
    side = side_of_direction(m.d)
    if m.evt == 1:
        price, size = m.levels[0]
        if size < 1:
            return "new_size_zero"
        opp = book.best(-side)
        if opp is not None and (price >= opp if side < 0 else price <= opp):
            return "new_crosses"
        return None

    if m.evt in (2, 3):
        price, size = m.levels[0]
        q = book.queue(side, price)
        if not q:
            return "no_level"
        if m.qslot == SLOT_NA:
            return "slot_na"
        if m.qslot < LAST_SLOT:
            if m.qslot >= len(q):
                return "slot_out_of_range"
            if q[m.qslot].owner == AGENT:
                return "slot_agent"
            if not _size_ok(m.evt, size, q[m.qslot].size):
                return "delete_size" if m.evt == 3 else "cancel_size"
            return None
        return None if slot_candidates(m, book) else "no_candidate"

    prices = book.prices(side)
    for k, (price, size) in enumerate(m.levels):
        if size < 1:
            return "exec_size_zero"
        if k >= len(prices) or price != prices[k]:
            return "exec_not_best" if k == 0 else "exec_skips_level"
        vol = book.volume(side, price)
        last = k == len(m.levels) - 1
        if not last and size != min(vol, MAX_SIZE):
            return "exec_partial_inner"
        if size > vol:
            return "exec_over_volume"
    return None


def apply(m: Message, book: Book, oid: int, slot: int | None = None):
    """Execute a legal message. `oid` names a new order; `slot` resolves a
    cancel or delete (required when m.qslot is 99). Returns the fills of an
    execution as [(order, filled), ...], else []."""
    side = side_of_direction(m.d)
    if m.evt == 1:
        price, size = m.levels[0]
        book.add(side, price, oid, size, BG)
        return []
    if m.evt in (2, 3):
        price, size = m.levels[0]
        s = m.qslot if slot is None else slot
        if m.evt == 3:
            book.remove_at(side, price, s)
        else:
            book.reduce_at(side, price, s, size)
        return []
    fills = []
    for price, size in m.levels:
        fills += book.execute(side, price, size)
    return fills


def time_consistent(m: Message, t_prev: int) -> bool:
    """Does the message's absolute time equal the previous one plus its dt?"""
    return m.time == t_prev + m.dt
