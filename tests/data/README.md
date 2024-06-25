# Test datasets

## mkt_ecdfs25_nano

This dataset is a small portion of a MIGHTEE survey observation which targets an off-Galactic plane field called E-CDFS 2.5, see Figure C1 of the [reference paper](https://arxiv.org/abs/2110.00347).

The data have been pre-processed by the [oxkat pipeline](https://github.com/IanHeywood/oxkat), and gone through the following steps:
- Averaging from 4096 to 1024 channels, and from 2 seconds sampling time to 8 seconds.
- 1GC: initial calibration and transfer of those calibration solutions to the science integrations.
- Flagging
- 2GC: direction-INdependent self-calibration

After that, we extracted the first 5 minutes of data, and channel indices 124 (inclusive) to 128 (exclusive).

The zip archive contains a directory called `mkt_ecdfs25_nano.ms`. This measurement set contains:
- 38 time samples
- 62 antennas
- 1953 baselines
- 4 frequency channels
- 4 correlation products: XX, XY, YX, YY
