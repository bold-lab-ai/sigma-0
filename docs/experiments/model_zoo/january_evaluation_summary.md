# January Evaluation Summary

## Status: Fully Completed ✅

When you sent your message, evaluation job `5848062` (submitted by another session) was actually already running (an independent short job, rather than attaching to job `5827830` which was fully occupied by `backfill124`). Instead of launching a duplicate job (which would trigger lock conflicts), I took over monitoring, built and smoke-tested the entire downstream pipeline during the wait time, and seamlessly executed the full end-to-end chain as soon as the evaluation finished.

| Stage | Result |
| :--- | :--- |
| **Evaluation** | 132/132 JSON + NPY files, job `COMPLETED 0:0`, 1h 25m |
| **Main Table** | 132/132 rows, comparing three metrics side-by-side |
| **Fitting** | 500 draws / 2000 runs, 0 failures |
| **Delivery Verification** | 285/285 SHA256 OK |
| **Report / Notion** | Written back / Appended 7 blocks |

---

## Core Conclusion

**The $\alpha$ ceiling pathology stems from the sampling design, not a special data distribution.**

| Axis (last25) | $\alpha$ Point Estimate | $\alpha$ Ceiling Ratio | $\beta$ | $E$ |
| :--- | :--- | :--- | :--- | :--- |
| **Jan-2026 (Current standard)** | 2.000 (Ceiling) | 100% | 0.801 | 0.5552 |
| **Jan-shuffle (New)** | 1.893 (Interior point) | 35.6% | 1.405 | 0.5234 |
| **valset** | 1.922 (Interior point) | 34.0% | 1.337 | 0.5957 |

**Jan-shuffle** shares the exact same month and batch of raw data with the current standard protocol, whereas **valset** is an entirely separate dataset. Yet, the fitting behavior of **Jan-shuffle** shifted over to align with **valset**. 

This sharpens the conclusion from *"changing the evaluation metric changes the result"* to *"the sampling design of the original metric itself creates the ceiling pathology."* The 100% ceiling hit rate in the current standard protocol can no longer be defended by claiming that *"Jan-2026 is a special month."* This directly addresses Reviewer **pXiP**'s concern regarding unconstrained refitting—something the **valset** axis alone could not accomplish.

---

## Optimal Scale

- **valset**: 120M
- **Current standard**: 23M
- **Jan-shuffle**: 120M

What shifts the optimal point is the shape of the bias: `valset − shuffle` exhibits a constant offset (all falling within +0.0720 to +0.0754 across 1M–350M), and a constant shift does not alter the location of the minimum point. In contrast, `current − shuffle` monotonically increases on larger model sizes from +0.0361 at 23M up to +0.0669 at 350M, bending the tail upwards.

---

## Three Items Proactively Fixed / Corrected

1. *(Details to be expanded)*
2. *(Details to be expanded)*
3. *(Details to be expanded)*
