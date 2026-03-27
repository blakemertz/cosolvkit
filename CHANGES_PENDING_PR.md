# Pending Pull Request — Changes on `update_cosolvkit`

This document records all changes committed locally on the `update_cosolvkit` branch
that have not yet been pushed to `origin` due to missing collaborator access.
Once you are added as a collaborator, follow the instructions at the bottom to push
and open the pull request.

---

## Branch information

| Item | Value |
|---|---|
| Local branch | `update_cosolvkit` |
| Remote target | `origin/update_cosolvkit` → PR against `master` |
| Commits ahead of `origin/update_cosolvkit` | 3 |

---

## Commit 1 — GPU selection (`8418c91`)

**Date:** 2026-03-23
**Files changed:** `cosolvkit/simulation.py`, `cosolvkit/cli/create_cosolvent_system.py`

### Problem

`CudaDeviceIndex` was hardcoded to `'0'` in `run_simulation()`, making it impossible
to target a specific GPU on multi-GPU workstations or run two simulations simultaneously
on separate devices.

### Changes

**`cosolvkit/simulation.py`**
- Added `gpu_id: int = 0` parameter to `run_simulation()`.
- `platform.setPropertyDefaultValue('CudaDeviceIndex', str(gpu_id))` replaces the
  hardcoded `'0'`.
- Added docstring entry for the new parameter.

**`cosolvkit/cli/create_cosolvent_system.py`**
- Added `--gpu_id` argument (type `int`, default `0`) to the CLI argument parser.
- Passes `gpu_id=args.gpu_id` through to `run_simulation()`.

### Usage

```bash
# Run two simulations simultaneously on separate GPUs
create_cosolvent_system -c config_run1.json --gpu_id 0
create_cosolvent_system -c config_run2.json --gpu_id 1   # separate terminal
```

> **Note:** `--gpu_id` only applies to the CUDA platform. If the code falls back to
> OpenCL or CPU, the argument has no effect.

---

## Commit 2 — mmCIF output for large systems (`9ea962b`)

**Date:** 2026-03-27
**Files changed:** `cosolvkit/utils.py`, `cosolvkit/cosolvent_system.py`,
`cosolvkit/simulation.py`, `cosolvkit/cli/create_cosolvent_system.py`

### Problem

Standard PDB format has fixed-width columns that cap atom serial numbers at 99,999
and residue sequence numbers at 9,999. OpenMM's `PDBFile.writeFile()` overflows these
limits using hybrid-36 / hexadecimal encoding for larger systems. MDAnalysis does not
decode this encoding, causing the warning:

```
PDB file is missing resid information. Defaulted to '1'
```

and corrupting all residue-based atom selections.

### Changes

**`cosolvkit/utils.py`**
- `MD_FORMAT_EXTENSIONS['OPENMM']['position']`: `.pdb` → `.cif`

**`cosolvkit/cosolvent_system.py`**
- `save_pdb()`: switched from `app.PDBFile.writeFile(..., keepIds=True)` to
  `app.PDBxFile.writeFile()`. mmCIF has no column-width limitations.
- Updated docstring to document the format change and rationale.

**`cosolvkit/simulation.py`**
- `run_simulation()`: detects file extension at load time — uses `app.PDBxFile` for
  `.cif` files and falls back to `app.PDBFile` for legacy `.pdb` files.

**`cosolvkit/cli/create_cosolvent_system.py`**
- Replaced the hardcoded string `"system.pdb"` with a dynamic lookup:
  ```python
  pos_ext = MD_FORMAT_EXTENSIONS['OPENMM']['position']
  pdb_fname = os.path.join(config.output_dir, f"system{pos_ext}")
  ```

### Compatibility note

- New runs produce `system.cif`. Existing `system.pdb` files from previous runs still
  load correctly because `run_simulation()` detects the extension.
- mmCIF is readable by MDAnalysis, PyMOL, and VMD.
- Users with existing `system.pdb` topology files should update their `analysis.py`
  scripts to point `top_file` at their `.pdb` file — the old file is still valid for
  those runs.

---

## Commit 3 — Analysis bug fixes and multi-trajectory support (`3d8e2c5`)

**Date:** 2026-03-27
**Files changed:** `cosolvkit/analysis.py`

### Bug fixes

#### 1. `cosolvent_names` string coercion (`Report.__init__`)

**Problem:** Passing `cosolvent_names="P1R"` (a string) caused the code to iterate
over individual characters `'P'`, `'1'`, `'R'` instead of treating `"P1R"` as a single
residue name. The resulting empty `AtomGroup` caused:
```
IndexError: index 0 is out of bounds for axis 0 with size 0
```
**Fix:** Added `if isinstance(cosolvent_names, str): cosolvent_names = [cosolvent_names]`
before the existing None/empty check.

#### 2. `generate_pymol_session` — `TypeError` when passing a list (`density_files`)

**Problem:** `os.path.isfile(density_files)` was called before `isinstance(density_files, list)`,
crashing with:
```
TypeError: stat: path should be string, bytes, os.PathLike or integer, not list
```
**Fix:** Reordered the checks so `isinstance(list)` is tested first.

#### 3. `generate_pymol_session` — auto-discovery of `.dx` files

**Problem:** `density_files=None` fell through to the `else` error branch, requiring
the caller to know the generated filenames (`map_rawdensity_P1R.dx`,
`map_agfe_P1R.dx`) before they existed.
**Fix:** Added a `None` branch that scans `self.out_path` for `*.dx` files, matching
where `generate_density_maps()` writes them. Correct usage is now simply:

```python
report.generate_pymol_session(reference_pdb="system.pdb", selection_string="")
```

#### 4. `cmd.spectrum` — empty selection string crashes PyMOL

**Problem:** Passing `selection_string=""` to `cmd.spectrum("b", "blue_white_red", "")`
raised:
```
pymol.CmdException: Error: Empty expression
```
**Fix:** Default to `"all"` when `selection_string` is empty:
```python
spectrum_selection = selection_string if selection_string else "all"
cmd.spectrum("b", "blue_white_red", spectrum_selection)
```

#### 5. VMD script — wrong trajectory type

**Problem:** The generated VMD script used `type netcdf` for the trajectory, but
cosolvkit produces DCD files.
**Fix:** Changed to `type dcd`.

### New feature — multi-trajectory support

**Problem:** `Report` accepted only a single trajectory and statistics file,
preventing combined analysis of replicate runs.

**Fix:** `traj_file` and `statistics_file` now accept either a `str` or a `list`:

- `Universe(topology, list_of_trajs)` — MDAnalysis `ChainReader` concatenates
  trajectories transparently; no changes needed downstream.
- `_get_temp_vol_pot`: uses `pd.concat` when a list is provided.
- VMD script: emits a `mol addfile` line for each trajectory in the list.

**Usage:**

```python
report = Report(
    statistics_file=["../results-p1r-run1/statistics.csv",
                     "../results-p1r-run2/statistics.csv"],
    traj_file=["../results-p1r-run1/aligned-pdb.dcd",
               "../results-p1r-run2/aligned-pdb.dcd"],
    top_file="system.pdb",
    cosolvent_names=["P1R"],
    out_path="results-combined"
)
```

Single-trajectory usage is unchanged — a bare string is wrapped in a list internally.

---

## How to push and open the pull request

Once you have been added as a collaborator to `forlilab/cosolvkit`:

### 1. Verify the branch is clean

```bash
git status          # should show only environment.yml as modified (pre-existing)
git log --oneline origin/update_cosolvkit..HEAD   # should show 0 commits if already pushed
# OR if not yet pushed:
git log --oneline origin/master..HEAD
```

### 2. Push the branch

```bash
git push origin update_cosolvkit
```

If you get an authentication error with HTTPS, either:
- Use a GitHub personal access token (Settings → Developer settings → Personal access tokens)
  as your password, or
- Switch the remote to SSH:
  ```bash
  git remote set-url origin git@github.com:forlilab/cosolvkit.git
  git push origin update_cosolvkit
  ```

### 3. Open the pull request

```bash
gh pr create \
  --base master \
  --head update_cosolvkit \
  --title "Bug fixes and enhancements: GPU selection, mmCIF output, multi-trajectory analysis" \
  --body "$(cat <<'EOF'
## Summary

- **GPU selection**: add \`--gpu_id\` CLI flag and \`gpu_id\` parameter to
  \`run_simulation()\` so CUDA simulations can be directed to a specific
  device, enabling concurrent runs on multi-GPU workstations.

- **mmCIF output**: switch OpenMM position output from PDB to mmCIF
  (\`PDBxFile\`) to correctly handle large systems (>99,999 atoms or
  >9,999 residues) that overflow standard PDB column widths. MDAnalysis,
  PyMOL, and VMD all support mmCIF natively.

- **Multi-trajectory analysis**: \`Report\` now accepts a list for
  \`traj_file\` and \`statistics_file\`, concatenating trajectories via
  MDAnalysis ChainReader and \`pd.concat\` respectively.

- **Analysis bug fixes**:
  - \`cosolvent_names\` coerced from bare string to list, preventing
    character-by-character iteration and downstream \`IndexError\`
  - \`generate_pymol_session\`: \`isinstance(list)\` check moved before
    \`os.path.isfile()\` to prevent \`TypeError\`; \`density_files=None\`
    now auto-discovers \`.dx\` files from the output directory
  - \`cmd.spectrum\` defaults to \`"all"\` when \`selection_string\` is
    empty, fixing PyMOL \`Empty expression\` error
  - VMD script: trajectory type corrected from \`netcdf\` to \`dcd\`;
    multiple trajectories emit repeated \`mol addfile\` lines

## Test plan

- [ ] Run \`create_cosolvent_system\` with \`--gpu_id 1\` on a multi-GPU
      machine and confirm the process binds to the correct device (\`nvidia-smi\`)
- [ ] Build a system with >99,999 atoms and confirm \`system.cif\` is
      written and loads cleanly in MDAnalysis without resid warnings
- [ ] Run \`Report\` with a list of two trajectory files and confirm
      frame counts and statistics are concatenated correctly
- [ ] Confirm single-trajectory usage is unaffected (string still
      accepted for both \`traj_file\` and \`statistics_file\`)
- [ ] Run \`generate_pymol_session()\` with no arguments after
      \`generate_density_maps()\` and confirm \`.dx\` files are found
      automatically

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### 4. Verify

```bash
gh pr list   # confirm the PR appears
```

---

## Commit hashes (for reference)

| Hash | Description |
|---|---|
| `8418c91` | Add gpu_id parameter to enable per-GPU simulation assignment |
| `9ea962b` | Switch OpenMM position output from PDB to mmCIF for large system support |
| `3d8e2c5` | Fix several analysis.py bugs and add multi-trajectory support |
