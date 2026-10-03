# LLM inference: build, measure, explain

A laptop-friendly learning project for an experienced ML/software engineer: build an inference runtime, measure its behavior, and explain the results to technical and general audiences.

## Start here

- [Detailed learning roadmap](outputs/learning-roadmap.md): 12 milestones over approximately 16–20 weeks.
- [Visual learning companion](outputs/visual-learning-companion.md): data-flow explanations and visual exercises.
- [Blog writing kit](outputs/blog-writing-kit.md): series plan and first-post draft.
- [Tiny inference engine](outputs/tiny-inference/README.md): runnable NumPy decoder, correctness tests, and a local benchmark.
- [Animated walkthrough](outputs/visualizations/inference-data-flow.html): generation, causal attention, KV reuse, and scheduling. Open the file locally after cloning; GitHub does not run HTML in repository views.

## Run the first lab

From the repository root, with Python and NumPy installed:

```bash
cd outputs/tiny-inference
python3 -m unittest -v test_engine.py
python3 engine.py --prompt 'Hello inference' --tokens 8
python3 benchmark.py --prompt-tokens 64 --new-tokens 32 --repeats 5
```

See the engine README for virtual-environment setup.

## Scope and evidence

The starter uses untrained weights to teach transformer mechanics. It is not a pretrained language model or production server. Nine correctness tests pass. The saved benchmark is a small CPU mechanics measurement, not a production performance claim.

The four animated scenes are illustrative, not measured execution traces. Their state transitions were checked programmatically; browser appearance has not been verified. The standalone HTML contains the styling and runtime needed outside this chat. Its editable visualization fragment is kept alongside it.

No cloud deployment, paid infrastructure, or public website is required. Scratch work, local environments, credentials, and downloaded model weights are excluded from version control.
