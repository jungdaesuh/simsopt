# Clean design-B compile matrix

Candidate: `975f6b722d0f9374b2980c7682506821b39fcad5`
Device: strict CPU, FP64
Watchdog: 120 seconds or 8 GiB RSS, direct child PID

| Cell | Elapsed (s) | Peak RSS (KiB) | StableHLO bytes | Lower (s) | Solver (s) | Iterations | Executables | Recompiled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| quadratic, `maxcor=10`, warm | 4.605 | 612,272 | — | — | 0.001959 | 2 | 1 | false |
| coil47, `maxcor=10`, compile | 5.607 | 853,928 | 1,571,730 | 0.929297 | — | — | — | — |
| coil47, `maxcor=10`, warm | 12.611 | 1,024,812 | — | — | 0.054607 | 11 | 1 | false |
| coil47, `maxcor=300`, compile | 5.603 | 858,632 | 1,572,317 | 0.917188 | — | — | — | — |

All four cells completed. The raw matrix is
`raw/matrix.json` (SHA-256
`836b913ddd86b17c2337326a5c4f0904e372e45f64300bb019aa479f9c600878`).

This is clean design-B evidence only. The legacy design-A transition timed out
in the earlier bounded comparison, so a complete promotion-grade A/B result is
still open; these measurements must not be presented as an A/B certification.
