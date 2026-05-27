# External Validation Feasibility

External validation is not implemented in the current release. This is a claim
discipline decision, not a compute objection.

The minimum useful validation would be deliberately narrow:

- Pacific Northwest only.
- ONDJFM only.
- Variables: precipitation, SST, and 500 hPa geopotential/height.
- Dataset-specific 90th-percentile `tp_extreme`.
- Fixed adjustment set `{z500}`.
- Same 1979-1999 train and 2000-2023 evaluation split.

The bottleneck is not the causal estimator. It is the adapter layer:

- acquiring MERRA-2 or JRA-55 with usable credentials and license terms;
- mapping each product's variable names, units, accumulation conventions,
  calendar, vertical coordinate, and grid to the existing panel schema;
- deciding whether the external SST field is semantically comparable to ERA5
  sea-surface temperature;
- verifying that the resulting panel reproduces the expected monthly count,
  ONDJFM mask, and extreme-event fraction before estimating ACE.

For the paper, the correct statement is:

> External reanalysis replication remains future work. The current release
> reports ERA5-only temporal holdout validation and documents a minimum viable
> MERRA-2/JRA-55 validation scope, but it does not claim cross-product
> replication.
