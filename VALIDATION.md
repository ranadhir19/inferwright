# Validation

Run the uniform harness from the repo root:

```bash
python3 validate.py
```

It is deterministic and offline, prints PASS/FAIL per stage, and exits 0
only when the final line reads `HARNESS GREEN`. Any failure exits non-zero
with `HARNESS RED: <stages>`.

## Stages and what they cover

1. **Import / compile** - parses `pyproject.toml`, byte-compiles every
   `.py` file in the repo (71 files), and imports the
   `llm_inference_engine` package plus every submodule (39 modules: the
   top-level package and its 10 system subpackages) with
   `PYTHONPATH=src`.
2. **Full test suite** - runs the complete existing pytest suite over the
   whole repo in a subprocess. The per-system suites all share the
   basename `tests/test_smoke.py`, so pytest is invoked with
   `--import-mode=importlib`; under the default import mode collection
   fails with "import file mismatch" before a single test runs. Importlib
   mode runs every test file - nothing is skipped, renamed, or weakened.
   Current result: **83 passed, 1 warning** (a pre-existing
   `PytestReturnNotNoneWarning` for `test_texts_labels` in
   `10-llm-mesh/tests/test_smoke.py`, which returns a tuple; warning
   only, the test passes).
3. **Forbidden-reference scan** - scans the whole tree with the standing
   external scanner kept outside this repository (an embedded copy of the
   same patterns is the fallback). Every candidate is judged in context:
   the harness's own pattern definitions are benign meta-references (a
   scanner must name its patterns); literals inside banned-term guard
   tests, if any appear, are judged by the same rule. Anything else that
   matches is a real reference and fails the stage; the harness never
   auto-removes such content, because whether it may be changed is the
   owner's decision. Current result: 13 candidates (all 13 in
   `validate.py` itself), **0 real references**.
4. **Results / docs consistency (existence only)** - every relative
   markdown link in every README resolves; every `results/...` and
   `experiments/...` file a README names exists; each of the 10 system
   folders ships an entry point (`experiments/run_experiments.py`),
   `tests/test_smoke.py`, and `results/results.md`.
   Stage 4 also includes **deployment-config validation**: the tree is
   scanned for `render.yaml`/`render.yml`, `vercel.json`, `Procfile`,
   `Dockerfile*`, `docker-compose*.yml`/`yaml`, `netlify.toml`,
   `fly.toml`, and `.github/workflows/*.yml`/`yaml`. Any file found is
   parsed and schema-checked (e.g. a Render service with `type: static`
   fails — a static site is `type: web` with `env: static`); a broken
   deploy config fails the stage. Current result: **no deployment
   configs present as of this run** — the step passes with nothing to
   validate.

## What this harness does NOT cover

- It does **not** re-run the experiments, and does not re-verify the
  measured numbers in `results/results.md` files or the README results
  showcase against fresh runs. Results files are checked for existence
  only, never for numeric correctness.
- It does not validate the *content* of READMEs or of the `docs/` site
  (claims, headline numbers, prose, rendered HTML) - only that README
  links and named files resolve. `docs/build.py` is compiled but not run.
- The pytest suites are the systems' own smoke tests, authored alongside
  the code. Passing them shows the code behaves as its authors asserted;
  it is not an independent domain-coverage audit, and no claim is made
  that uncovered behaviour is correct.
- The forbidden-reference scan judges candidates by the documented
  meta-reference rules and fails closed on anything else. A novel benign
  context would fail the stage until a human judges it and, if
  appropriate, extends the rules with the justification recorded.
- No GPU, network, installation-from-PyPI, or packaging/release checks
  are performed (see `PYPI_RELEASE.md` for the separate manual release
  checklist); the package is exercised from the source tree only, on CPU.
