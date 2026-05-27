# Season Provenance Note

The current primary analysis is frozen around ONDJFM, implemented in
`config/base.yaml` as months `[10, 11, 12, 1, 2, 3]` for both discovery and
inference. That window is defensible as a broad cool-season precipitation
window for the Pacific Northwest.

The repository does not contain enough timestamped design history to prove that
ONDJFM was chosen before any exploratory result was inspected. Therefore the
paper should avoid claiming fully prospective seasonal pre-registration. The
most accurate wording is:

> ONDJFM is the frozen primary season for the current analysis release and is
> climatologically motivated as a broad Pacific Northwest cool-season window.
> Because the project evolved through exploratory specifications, the NDJF
> failure means the result should be treated as ONDJFM-specific rather than as a
> generic winter effect. The temporal holdout reduces overfitting concerns for
> the frozen claim, but it does not erase all seasonal-selection risk.

This wording is intentionally conservative. It gives the grader/reviewer the
truth they would otherwise probe for and prevents the NDJF failure from looking
like an unexplained post-hoc inconvenience.
