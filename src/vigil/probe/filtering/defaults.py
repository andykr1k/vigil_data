"""Starting noise values shared by the probe pose filters (from DataCollection's config)."""

POSITION_MEASUREMENT_NOISE_METERS = 0.01
LINEAR_ACCELERATION_NOISE_METERS_PER_SECOND2 = 1.0
# Longest tag dropout a filter's history survives; longer gaps restart it.
MAX_MISSING_SECONDS = 0.5
