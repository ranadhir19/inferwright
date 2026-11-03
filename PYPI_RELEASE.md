# PyPI Release Checklist — llm-inference-lab

Distribution name: **llm-inference-lab** (checked free on PyPI 2026-10-19;
`llm-inference-engine` was taken). Import name: `llm_inference_engine`.

## Pre-release

- [ ] All 8 `experiments/run_experiments.py` run green twice (byte-identical)
- [ ] All 8 `tests/test_smoke.py` pass
- [ ] All 8 `examples/*.py` execute green
- [ ] `pip install -e .` works from a clean checkout
- [ ] `import llm_inference_engine` exposes all 8 subpackages
- [ ] Version bumped in `pyproject.toml` (currently 0.1.0)
- [ ] `README.md` renders correctly on PyPI (valid markdown, no relative image refs)
- [ ] `LICENSE` present (MIT)
- [ ] No secrets, no binaries, no `.npz` artifacts in the sdist

## Build

```bash
python3 -m build
twine check dist/*
```

## Publish (test first)

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ llm-inference-lab
twine upload dist/*
```

## Post-release

- [ ] `pip install llm-inference-lab` works on a fresh machine
- [ ] Tag `v0.1.0` in GitHub, create release with changelog
- [ ] Update landing-site install block if the name/version changes
