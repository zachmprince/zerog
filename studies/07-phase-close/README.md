# Study 07: Phase 1 gate runs

Status: done (milestone M9). The results are in [decision 012](../../docs/decisions/012-phase-1-close.md).

The three gate cases on study 04's grid, with the code at the phase-closing commit: study 03's SFR
and PWR mixes, and study 05's SFR mix with 64 probability-table bands. Each also times one cold
fixed-source solve from its converged fission source, the speed gate's measure.

```bash
cd studies/03-k-infinity/scripts
```

```bash
PYTHONPATH=. pixi run python -u run_k.py sfr 1e-3 2 > ../../07-phase-close/results/sfr.txt
```

```bash
PYTHONPATH=. pixi run python -u run_k.py pwr 1e-3 2 > ../../07-phase-close/results/pwr.txt
```

```bash
cd ../../05-unresolved-range/scripts && pixi run python -u run_bands.py 64 1e-3 2 > ../../07-phase-close/results/sfr-bands.txt
```

The twins are in [`benchmarks/05-fast-k-infinity`](../../benchmarks/05-fast-k-infinity). `results/`
is gitignored.
