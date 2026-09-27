# Verification

Verified locally on CPU with Python 3.12, PyTorch 2.14.0+cpu and NumPy 2.5.2.

- All 9 tests in `tests/test_relative.py` passed.
- Three training epochs completed using `--quick --subcarriers 2 --batch-size 2 --device cpu` (8 training, 4 validation, 4 extrapolation environments).
- Best validation checkpoint was saved, reloaded and evaluated successfully; configuration, history and evaluation files were produced.
- `tests.diagnose_relative` completed for two independent environments at b=0 and b=0.1, including shuffled geometry, disabled modulation and mean-reference comparisons. CSV and JSON output were verified.

The smoke run is deliberately too small to assess accuracy: validation improvement over averaging was 0.07 dB, and extrapolation improvement was -0.04 dB. These are wiring checks, not evidence of a performance advantage. No full training experiment or GPU memory benchmark was performed. Smoke-test weights are not distributed as a trained model.
