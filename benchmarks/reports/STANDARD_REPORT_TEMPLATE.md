# Standard Benchmark Report

Use this template for scheduled JAX benchmark reporting.

## Run Identity

- benchmark id:
- title:
- repo sha:
- report date:
- workflow:

## Hardware Contract

- runner labels:
- platform:
- stable-hardware note:

## Runtime Contract

- default rollout lane: `native_cpu`
- benchmark lane:
- JAX / jaxlib:
- x64 enabled:
- compile behavior:
- compilation cache policy:

## Fixture Summary

- fixture:
- stage 2 grid:
- single-stage grid:
- optimizer backend:

## Timing Summary

| rung | passed | outer elapsed s | cpu elapsed s | lane elapsed s | speedup vs cpu |
| --- | --- | --- | --- | --- | --- |
| tier1b_real_stage2 |  |  |  |  |  |
| tier2_stage2_e2e |  |  |  |  |  |
| tier3_single_stage_init |  |  |  |  |  |
| tier4_adjoint_fd |  |  |  |  |  |

## Memory Summary

- peak RSS MB:
- GPU memory MB:

## Honest Interpretation

- cold compile notes:
- warm timing notes:
- parity-vs-fast note:
- memory caveats:

## Follow-up

- regressions to investigate:
- doc/report consumer updates:
