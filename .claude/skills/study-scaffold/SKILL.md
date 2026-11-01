---
name: study-scaffold
description: Create a numbered study under studies/ (README with question, method, result, decision; scripts; figures) and a matching decision-record stub in docs/decisions/
disable-model-invocation: true
argument-hint: <short-name> "<the one question this study answers>"
---

Scaffold a new ZeroG study. Arguments: `$ARGUMENTS` (short kebab-case name, then the question).

1. List `studies/` and `docs/decisions/` and pick the next free two-digit number `NN` (start at `00` if empty).
2. Create `studies/NN-<name>/` with:
   - `README.md` using the template below.
   - `scripts/` and `figures/` (add a `.gitkeep` to each).
3. Create `docs/decisions/NNN-<name>.md` as a short stub: Context, Decision (pending), Consequences. Link it from the study README.
4. If the study compares against OpenMC, note in the README that its twin lives in `benchmarks/NN-<name>/` with the same geometry, data and settings, unresolved-range probability tables off, and tally uncertainty at most a fifth of the comparison tolerance. Do not create the twin here.
5. Do not commit anything and never `git add -A`.

README template:

```markdown
# Study NN: <title>

## Question
<one question>

## Method
<geometry, data (library, temperatures), solver settings, OpenMC twin>

## Result
<pending: one-paragraph finding, figures in figures/>

## Decision
<pending: go / no-go against the phase gate; link to docs/decisions/NNN-<name>.md>
```

Follow `CLAUDE.md` conventions: eV, cm, barns, atoms/barn-cm, kelvin; energy arrays high to low.
