# other/

This folder is a **safety net** for repository items that don't cleanly map to a single processing category, or that we temporarily park during re-organization to avoid accidental loss.

## Why files end up here
A file should be in `other/` if it is:
- Experimental / ad-hoc
- Helper-only or not clearly a data pipeline stage
- Legacy
- Or, during refactor, a top-level repo artifact we want to preserve without changing behavior yet

## Contents
- `README_original.md`
  - The original repository README before refactor.
  - Kept here so we can replace it later with a new top-level README that reflects the new folder layout.

- `requirements.txt`
  - Dependency list preserved as-is.
  - We can move it back to repo root later (optional) once the structure stabilizes.

