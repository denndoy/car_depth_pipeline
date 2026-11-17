# Vendored marker and monotone-quartic helpers

These are local copies of generic helpers from the person-research workspace.
The originals remain unchanged. They are included so the car repository can
run without a sibling person-project checkout.

- `marker_line_tools.py` — marker-point annotation, profile, visual QC, crossing
  candidates, and manually confirmed event export. For this car study, the
  bottom-center box proxy corresponds to the front bumper only because vehicles
  approach the camera consistently. If `subject_id` is blank, the car helper
  uses the run-local `track_id` as a pseudo vehicle ID. Event exports retain the
  legacy `foot_crossing_confirmed` CSV column name, which means manual front-
  bumper crossing confirmation in this project.
- `polynomial_take_pipeline.py` — strict CSV/GT validation, detection linking,
  frozen model metadata checks, and metric helpers reused by scripts 03/04.
- `monotone_quartic.py` — compact, car-generic copy of constrained quartic
  fitting and analytical monotonicity verification reused by scripts 03/04.
  Person IDs, job IDs, paper plots, and output paths are intentionally omitted.

These are source copies inside the car project; the original person scripts were
not modified. Keep model/data paths relative to the car repository root.
