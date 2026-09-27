# Relative antenna-domain channel estimator

Separate experimental version based on the saved mean-reference project from the earlier chat. This is not a pull of the latest GitHub repository. Requires a fresh training run; old beamspace checkpoints are incompatible.

## Run in Kaggle

Extract the project, change into its directory, then run:

```python
!python -m unittest discover -s tests -p "test_*.py" -v
!python -m main.train --quick --device cuda --batch-size 4
```

For a fixed-amplitude experiment matching the previous b=0.1 discussion:

```python
!python -m main.train --device cuda --batch-size 4 --b-min 0.1 --b-max 0.1
```

For the default mixed training range (0.05–0.30 wavelengths):

```python
!python -m main.train --device cuda --batch-size 4
```

Validation uses the training deformation range and independent environments. The extrapolation split remains 0.35–0.50 wavelengths. Dataset size, seeds, and other defaults are inherited. Batch size 4 is recommended initially because retaining features for each observation uses more memory; the inherited command-line default is 16. Use the same batch size across comparisons.

After training:

```python
!python -m tests.diagnose_relative --run-dir artifacts/YOUR_RUN --device cuda
```

This writes a matched-seed deformation sweep comparing normal predictions, geometry from a different environment, disabled modulation, and the mean baseline. These interventions diagnose an already-trained model; disabling modulation is not a substitute for training an unconditioned baseline.

## Architecture and data contract

The dataset continues returning `(observations, geometry, residual_targets)`:

- Observations: `[B,4*M,NB,NU]`, with real/imaginary channels for two adjacent subcarriers.
- Geometry: `[B,M,NB,NU,6]`, BS xyz followed by UE xyz, in wavelength units relative to the fixed origin.
- Target: `[B,4,NB,NU]`, normalized undeformed H0 minus the mean observation.

The first measured observation is the input reference. Each state receives eight real channels `[Yr, Ym-Yr]`. A shared CNN processes states separately. Geometry descriptors are `[G0, Gr-G0, Gm-Gr]` (18 values); an MLP and pointwise convolutions produce spatial gamma/beta for every state and antenna pair. Conditioned features are averaged across states, decoded into an antenna-domain residual, and added directly to the observation mean. There is no beamspace transform or inverse transform.

All channel quantities retain the existing common observation-derived normalization. Relative preprocessing is lossless when the reference is retained, but it does not eliminate measurement noise or guarantee improved NMSE. Reference subtraction creates correlated noise among the differences.

Nominal geometry G0 is known hardware geometry, never the true H0 channel. Defaults match the supplied simulator's facing arrays and wavelength/8 spacing. For another geometry, pass the exact nominal `[NB,NU,6]` coordinates in the same units/frame as the dataset. Both model configuration and checkpoint buffer preserve these coordinates. The simulator displaces along array normals, which are not necessarily the global z axis; all three displacement components are retained.

The CNN operates over flattened BS/UE indices as in the previous implementation. It is not a graph convolution or a full treatment of the two physical 2D array topologies. The chosen reference state is special; only permutations that keep it fixed are guaranteed to preserve predictions.

## Comparison option

`--variant raw` uses the same shared antenna-domain network, geometry conditioning and fusion, but supplies four raw channels per state instead of eight relative channels. This isolates the relative-channel representation within this new architecture, apart from the differing input stem size. It is not an exact reproduction of the old beamspace baseline. Keep the previous project for that comparison.

## Changed files

New: `model/relative_preprocessing.py`, `tests/test_relative.py`, `tests/diagnose_relative.py`.

Replaced: `model/estimator.py`, `model/dilated_cnn.py`, `model/encoders.py`, `model/geometry_modulation.py`, `model/decoder.py`, `model/residual_reconstruction.py`.

Updated: `main/train.py` (variant and deformation-range options; full architecture configuration saved).

Preserved: channel physics, pilot estimation, dataset generation, geometry generation, normalization, training objective and checkpoint selection. Old architecture-specific tests and diagnostics are not bundled; use the new tests above.
