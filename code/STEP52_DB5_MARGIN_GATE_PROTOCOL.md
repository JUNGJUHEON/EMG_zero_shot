# Step52 DB5 Assignment-Margin Gate Protocol

Status: locked before primary prediction

Date: 2026-08-12

## Question

Can a truth-free assignment-confidence gate retain raw pose naming when it is
confident and use dynamic-pose naming only when the raw assignment is ambiguous?

## Scope

- NinaPro DB5 s1-s10 only; DB1 is forbidden.
- Fit exposure is A1-A12+B5/B6 and evaluated classes are B1/B2/B3/B4/B7.
- Step48 stable clusters and other-nine source pose anchors are frozen.
- Selection uses the 60 five-class mixed A/B5/B6 legal proxies per subject.

## Gate Family

For raw all-66 naming and dynamic `[std, delta]` naming, compute the difference
between the best and second-best Hungarian total assignment scores. Candidate gates
are:

1. always raw;
2. choose the larger-margin assignment;
3. choose dynamic only when mappings differ, dynamic margin exceeds raw margin, and
   raw margin is below one of `{0.005, 0.01, 0.02, 0.04, 0.08}`.

For outer target `s_i`, only the other nine subjects select a gate. An alternative
is eligible only when mixed-proxy held-B recall improves by at least `0.02` and
overall mixed five-class BA decreases by no more than `0.005`. Rank eligible gates
by held-B recall, overall BA, lower dynamic-switch rate, then lexical ID. If none is
eligible, retain raw naming.

## Firewall And Boundary

Phase 1 cannot open primary query caches, truth seals, endpoint scores, or DB1 paths.
The gate is hash-sealed before primary prediction, and a separate evaluator joins
truth only after prediction lock. Step52 is post hoc DB5-only internal evidence.
