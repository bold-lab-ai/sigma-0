# Environment

Closed-loop market replay and trading-environment code lives here.

This package owns simulator-facing environment logic. It can call into
`matching_engine` for order-book state transitions, but matching-engine code
should not import this package.

