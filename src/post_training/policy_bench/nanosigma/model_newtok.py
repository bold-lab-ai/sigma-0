"""Scenario (a) model: the r4 baseline (NanoSigma) with the new variable-length message tokens.

Same blocks, widths and book branch as NanoSigma. Only two things change:
- vocabulary 2216 (newtok.VOCAB_NEW) for the embedding and the output head;
- the book vector of message i is spread over that message's token positions with a gather on
  msg_idx (messages now have 16, 17 or 12+5k tokens instead of a fixed 26).
"""
import torch
import torch.nn as nn

from .model import NanoSigma
from .newtok import VOCAB_NEW


class NanoSigmaNewTok(NanoSigma):
    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.d
        self.embed = nn.Embedding(VOCAB_NEW, d)
        self.head = nn.Linear(d, VOCAB_NEW, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def hidden(self, tokens, msg_idx=None, book=None):
        """tokens (B, T) input ids (shifted), msg_idx (B, T) message index of each TARGET position,
        book (B, M, 503) one vector per message."""
        from .model import rope_cache
        B, T = tokens.shape
        cos, sin = rope_cache(T, self.d // self.n_heads, tokens.device)
        x = self._run(self.msg_blocks, self.embed(tokens), cos, sin)
        if self.use_book:
            b = book.to(x.dtype)
            b = b + self.book_pre(self.book_pre_norm(b))
            b = self.book_proj(b)                                            # (B, M, d)
            b = torch.gather(b, 1, msg_idx.long().unsqueeze(-1).expand(-1, -1, b.size(-1)))
            b = self._run(self.book_blocks, b, cos, sin)
            x = self.fuse(torch.cat([x, b], dim=-1))
        x = self._run(self.blocks, x, cos, sin)
        return self.norm(x)

    def forward(self, tokens, msg_idx=None, book=None):
        return self.head(self.hidden(tokens, msg_idx, book))
