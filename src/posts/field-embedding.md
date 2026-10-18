---
title: Why Per-Field Embeddings Often Beat a Learned Encoder
date: 2026-10-18
description: Four mathematical reasons why a separate embedding table per field beats a learned encoder for structured order-book events, and when it does not.
tags: [embedding, transformer, representation learning, financial sequence modelling]
---

# Why Per-Field Embeddings Often Beat a Learned Encoder

When you feed structured events into a Transformer, for example an order-book action made of side, type, price level and size, there are two common designs:

- give each field its own embedding table, then add or concatenate the vectors you look up;
- pass the whole action through an encoder (for example an MLP) that outputs one embedding.

In practice the first design usually works better. This post explains why from four angles: expressiveness, optimisation, statistical efficiency and geometry. It ends with the cases where the conclusion flips.

## 0. Setup

An action has $K$ discrete fields, such as side, type, price level and size bucket. Field $i$ takes a value $x_i \in \{1,\dots,V_i\}$.

**Option A (one embedding per field):**

$$
h = \sum_{i=1}^K E_i[x_i] \quad\text{or}\quad h=[E_1[x_1];\dots;E_K[x_K]],\qquad E_i\in\mathbb{R}^{V_i\times d}
$$

**Option B (encoder):** $h = f_\theta(x)$, where $f_\theta$ is an MLP or another network. Its input is either the raw numbers or the concatenated field features.

## 1. With a linear first layer, the two are the same

Concatenate the one-hot vectors of all fields into $u=[e_{x_1};\dots;e_{x_K}]$ and apply one linear layer:

$$
Wu = \sum_i W_i\, e_{x_i} = \sum_i W_i[:,x_i]
$$

This is exactly the sum embedding of Option A.

So with one-hot input, the first layer of the encoder already is "one embedding per field, then add". All the encoder adds is the nonlinear layers after it. The real differences come from the three places below.

## 2. When the encoder reads raw numbers: spectral bias of MLPs

Suppose price and size go into the MLP as **scalars**, so $h=f_\theta(p)$. Compare two ways to map a value to a vector:

- **Lookup**: every value gets its own free vector $E[p]$. This can represent any function $g:\{1,\dots,V\}\to\mathbb{R}^d$ with $Vd$ parameters and assumes nothing about the function's shape.
- **ReLU MLP**: the output is piecewise linear in $p$. If $g$ has $m$ "kinks" (non-monotone points or jumps) over the $V$ values, you need about $m$ hidden units to fit it. MLPs also have a known spectral bias: a component with frequency $k$ is learned slowly, roughly $1/k^2$ times as fast.

Prices and sizes in financial data are exactly **high-frequency and non-smooth**. Round prices, tick boundaries and round lots (100, 500, 1000 shares) each carry their own meaning. For example, size = 100 and size = 101 mean very different things, but to an MLP the two inputs are nearly identical, so its outputs are nearly identical too. A lookup table does not have this problem: neighbouring values can learn completely different vectors.

Heavy-tailed values such as size make it worse. The scale of $w\cdot p$ is dominated by extreme values, and so are the gradients. A lookup over buckets does not depend on scale at all.

## 3. Optimisation: lookup gradients are decoupled

In Option A, the gradient only touches the rows that actually appear in the batch:

$$
\frac{\partial L}{\partial E_i[v]} = \sum_{n:\,x_i^{(n)}=v} \frac{\partial L}{\partial h^{(n)}}
$$

Parameters for different rows and different fields are not coupled. This means:

- the Hessian is roughly **block-diagonal** and well conditioned, so Adam's per-parameter scaling is almost the right preconditioner;
- rare values have their own parameters, so their gradients are not drowned out by frequent values.

In Option B all inputs share $\theta$, and the gradient is a sum over all samples:

$$
\nabla_\theta L=\sum_n J_\theta\big(x^{(n)}\big)^\top \frac{\partial L}{\partial h^{(n)}}
$$

Frequent values dominate the update direction, fields interfere with each other, and the Hessian is dense, so optimisation is harder.

## 4. Statistics: additive decomposition and sample complexity

If you view the encoder as modelling the whole combination of fields, its most flexible form gives every combination its own token, with $\prod_i V_i\, d$ parameters. Option A has $\sum_i V_i\, d$.

| | Parameters | Training samples per parameter (total $N$) |
|---|---|---|
| Joint token / tightly coupled encoder | $\prod_i V_i\,d$ | $\sim N/\prod_i V_i$ |
| One embedding per field | $\sum_i V_i\,d$ | $\sim N/V_i$ |

Example: with $V=(2,4,20,30)$, $\prod_i V_i=4800$ and $\sum_i V_i=56$. Write $P$ for the parameter count. Using the rough estimation error $\sim\sqrt{P/N}$, the error differs by about $\sqrt{4800/56}\approx 9$ times.

The key point is that Option A only models first-order (additive) effects at the **input layer**. Interactions between fields (for example side × price level) are left to the Transformer that follows. The Transformer learns those interactions with attention and FFN layers, and it learns them **in context**, which is more efficient than learning them for each action in isolation inside an encoder. This is an "additive first, interactions later" decomposition:

$$
p(y\mid x)\approx F\Big(\sum_i E_i[x_i],\ \mathrm{context}\Big)
$$

The assumption behind it is that most of the signal is a sum of per-field main effects and interactions matter less. This usually holds on real data, so you get small bias and much smaller variance.

## 5. Geometry: each field lives in a nearly orthogonal subspace

With **concatenation**, each field sits in its own exactly orthogonal subspace. The attention score

$$
q^\top k = \sum_{i,j} E_i[x_i]^\top W_Q^\top W_K\, E_j[x'_j]
$$

splits into field-by-field blocks. The model can attend to just one field, for example compare only price levels, by learning zero weights for the other blocks.

With **summation**, randomly initialised vectors in high dimension $d$ are nearly orthogonal, with inner products of order $O(1/\sqrt d)$. The fields therefore stay approximately separable, and a downstream linear layer can "read them back out".

An encoder's nonlinearity **entangles** the fields, so the downstream layers have to disentangle them before they can use a single field.

## 6. When an encoder is better (caveats)

1. A numeric field has very many values and really is smooth (for example continuous prices, with unseen values at test time). A lookup cannot extrapolate, so an MLP or "bucket embedding + numeric feature" works better.
2. The encoder's $d$ is smaller than the number of fields times the dimension each field needs, i.e. there is a bottleneck. The encoder may then lose because of capacity, not because of the method.
3. When comparing the two, make sure parameter counts and training steps are comparable. Otherwise Option A may win only because it has more parameters.

## Summary

A lookup is a non-parametric estimate for each discrete value: no smoothness assumption, decoupled gradients. Splitting by field is an additive decomposition at the input layer that cuts parameters from $\prod_i V_i$ to $\sum_i V_i$ and leaves interactions to the Transformer, which sees context. An encoder adds both a smoothness bias (bad for the discrete, high-frequency meaning of financial fields) and parameter coupling (harder to optimise), so it usually does worse.
