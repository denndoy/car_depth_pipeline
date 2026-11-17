# Vendored track-review helpers

These are local copies of generic, manually reviewed helper scripts from the
person-research workspace. The originals remain unchanged. They are included
here so the car repository does not require a sibling person-project checkout.

- `prepare_track_review.py` — offline review of tracker segments and manual
  vehicle-ID assignment.
- `apply_subject_crosswalk.py` — applies confirmed mappings to a copy of the
  detections table; the legacy `subject_id` column is used as `vehicle_id`.

The review CSV is tied to the exact detection file by SHA256. Track IDs are
local to a video/run and are not vehicle identity.
