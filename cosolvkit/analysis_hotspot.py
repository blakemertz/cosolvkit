#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Hotspot detection and scoring from cosolvent occupancy maps
#
# Method (see Ghanakota & Carlson, J. Phys. Chem. B 2016, 120, 8685-8695 and
# Ghanakota et al., J. Chem. Inf. Model. 2018, 58, 784-793):
#   1. Each probe's occupancy map is converted to Z-scores, (x - <x>) / sigma,
#      over the entire grid.
#   2. Grid points with Z >= z_cutoff are clustered per probe by single-linkage
#      clustering with a distance cutoff (default 3.0 A). Each cluster is a spot.
#   3. Spots that share grid points (across probes) are merged into hotspots.
#      A hotspot consists of at least two overlapping spots.
#   4. Hotspot score = sum over probes, spots and grid points of Z.
#      Points shared by overlapping spots therefore contribute once per spot,
#      whereas the hotspot volume counts every grid point once.
#
# Alternative grouping (method="dbscan", see Graham, Leja & Carlson, J. Chem. Inf.
# Model. 2018, 58, 1426-1433): the grid points above the cutoff of ALL probes are
# pooled and clustered together with DBSCAN (eps, min_samples). Each DBSCAN
# cluster is a hotspot, scored by the same sum of Z over its points; the
# contribution of every probe is reported separately. Points that DBSCAN labels
# as noise are discarded.
#
# A "probe" is one occupancy map: a whole cosolvent (map_rawdensity_<RES>.dx,
# use_atomtypes=False, or map_com_<RES>.dx, the occupancy of the molecules'
# centers of mass, com_density=True) or one functional group of a cosolvent
# (map_density_<ATYPE>_<RES>.dx, use_atomtypes=True).
#
# All maps must be in the same Cartesian frame: trajectories of every probe
# simulation have to be aligned (rotation + translation) onto one common
# reference structure before the maps are generated. Centering/imaging alone is
# not enough. Use check_frame_consistency() to verify this.
#

import os
import re
import sys
import glob
import logging
import argparse
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN

from cosolvkit.analysis import _read_dx, _grid_density


# Named explicitly: under "python -m cosolvkit.analysis_hotspot", __name__ is "__main__"
# and records would bypass the "cosolvkit" logger configured by setup_logging().
logger = logging.getLogger("cosolvkit.analysis_hotspot")

_DENSITY_ATYPE_RE = re.compile(r"^map_density_(?P<atype>.+)_(?P<resname>[^_]+)\.dx$")
_DENSITY_RAW_RE = re.compile(r"^map_rawdensity_(?P<resname>[^_]+)\.dx$")
_DENSITY_COM_RE = re.compile(r"^map_com_(?P<resname>[^_]+)\.dx$")
MAP_KINDS = ("atomtype", "raw", "com")


def _map_kind(fname: str):
    """Returns 'atomtype', 'raw' or 'com' for a cosolvkit occupancy map file name, None otherwise."""
    basename = os.path.basename(fname)
    if _DENSITY_COM_RE.match(basename):
        return "com"
    if _DENSITY_RAW_RE.match(basename):
        return "raw"
    if _DENSITY_ATYPE_RE.match(basename):
        return "atomtype"
    return None


def _probe_from_filename(fname: str):
    """Returns (probe_name, cosolvent_resname) for a cosolvkit density map file name,
    or None if the file is not an occupancy map (e.g. map_agfe_*.dx, solvent_accessible_map.dx)."""
    basename = os.path.basename(fname)
    match = _DENSITY_COM_RE.match(basename)
    if match:
        return f"{match['resname']}_COM", match['resname']
    match = _DENSITY_ATYPE_RE.match(basename)
    if match:
        return f"{match['resname']}_{match['atype']}", match['resname']
    match = _DENSITY_RAW_RE.match(basename)
    if match:
        return match['resname'], match['resname']
    return None


def find_density_maps(directories: Union[str, List[str]], kinds: Union[str, List[str]] = ("atomtype", "raw")) -> List[str]:
    """Collects the occupancy maps written by Report.generate_density_maps() in one or more directories.
    Free energy maps (map_agfe_*) and the solvent accessible map are skipped, they are not occupancies.

    :param directories: directory or list of directories to search.
    :type directories: Union[str, list]
    :param kinds: kinds of occupancy maps to collect: 'atomtype' (map_density_<ATYPE>_<RES>.dx, one map per
        functional group), 'raw' (map_rawdensity_<RES>.dx, all atoms) and/or 'com' (map_com_<RES>.dx, molecule
        centers of mass). Mixing per-atom and center of mass maps of the same cosolvent counts it twice,
        defaults to ("atomtype", "raw")
    :type kinds: Union[str, list], optional
    :return: sorted list of density map paths.
    :rtype: list
    """
    if isinstance(kinds, str):
        kinds = [kinds]
    unknown = set(kinds) - set(MAP_KINDS)
    if unknown:
        raise ValueError(f"Unknown map kinds {unknown}, choose from {MAP_KINDS}")

    if isinstance(directories, str):
        directories = [directories]

    density_files = []
    for directory in directories:
        for fname in sorted(glob.glob(os.path.join(directory, "*.dx"))):
            if _map_kind(fname) in kinds:
                density_files.append(fname)

    return density_files


def check_frame_consistency(structure_files: List[str], selection: str = "name CA", tolerance: float = 2.0) -> pd.DataFrame:
    """Checks that structures (e.g. the averaged_trajectory.pdb of each probe simulation) are in the
    same Cartesian frame, by computing the RMSD between them WITHOUT superposition.
    Maps coming from simulations whose structures differ by more than a few Angstrom cannot be
    merged into hotspots: the trajectories must be aligned to a common reference first.

    :param structure_files: list of structure files, one per probe simulation.
    :type structure_files: list
    :param selection: MDAnalysis selection used for the comparison, defaults to "name CA"
    :type selection: str, optional
    :param tolerance: RMSD (A) above which a warning is emitted, defaults to 2.0
    :type tolerance: float, optional
    :return: matrix of no-fit RMSD values (A).
    :rtype: pd.DataFrame
    """
    from MDAnalysis import Universe

    positions = []
    for structure_file in structure_files:
        atoms = Universe(structure_file).select_atoms(selection)
        positions.append(atoms.positions.astype(float))

    n_atoms = {len(p) for p in positions}
    if len(n_atoms) != 1:
        raise ValueError(f"Selection '{selection}' matched a different number of atoms in each structure: {n_atoms}")

    names = [os.path.basename(os.path.dirname(os.path.abspath(f))) or f for f in structure_files]
    rmsd = np.zeros((len(positions), len(positions)))
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            rmsd[i, j] = rmsd[j, i] = np.sqrt(np.mean(np.sum((positions[i] - positions[j]) ** 2, axis=1)))
            if rmsd[i, j] > tolerance:
                logger.warning(f"No-fit RMSD between {structure_files[i]} and {structure_files[j]} is {rmsd[i, j]:.2f} A "
                               f"(> {tolerance} A). The maps are not in the same frame, hotspots will not be meaningful.")

    return pd.DataFrame(rmsd, index=names, columns=names)


def _cluster_points(xyz: np.ndarray, cutoff: float) -> np.ndarray:
    """Single-linkage clustering with a distance cutoff.
    Equivalent to scipy.cluster.hierarchy.fcluster(linkage(xyz, 'single'), cutoff, 'distance'):
    the clusters are the connected components of the graph linking points closer than the cutoff.
    This avoids the O(n^2) memory of the full linkage matrix.

    :return: cluster label for each point.
    :rtype: np.ndarray
    """
    if len(xyz) == 0:
        return np.zeros(0, dtype=int)

    pairs = cKDTree(xyz).query_pairs(r=cutoff, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz)))
    _, labels = connected_components(graph, directed=False)

    return labels


def _normalize_occupancy(grid: np.ndarray, normalize: str = "zscore") -> np.ndarray:
    """Z-scores, (x - mean) / sigma over the whole grid, or fraction of the maximum, x / max."""
    if normalize == "zscore":
        return _grid_density(grid)
    if normalize == "fraction":
        grid_max = grid.max()
        return grid / grid_max if grid_max > 0 else np.zeros_like(grid, dtype=float)
    raise ValueError(f"Unknown normalization {normalize}, use 'zscore' or 'fraction'.")


class HotspotAnalysis:
    """Detects and scores hotspots from the occupancy maps of one or more cosolvent probes.

    Example:

        hotspots = HotspotAnalysis()
        for fname in find_density_maps(["analysis/piperazine_GT", "analysis/pyrazolopyrimidine_VQ"]):
            hotspots.add_map(fname)
        hotspots.run(z_cutoff=20.0, cluster_cutoff=3.0)
        hotspots.export("hotspots_out", reference_structure="analysis/piperazine_GT/averaged_trajectory.pdb")
    """
    def __init__(self):
        self.logger = logger
        self._probes = {}
        self.spots = None
        self.hotspots = None
        self._hotspot_points = None

    def add_map(self, fname: str, probe_name: str = None, cosolvent: str = None):
        """Adds one occupancy map (raw bin counts or Z-scores, the Z-score transformation is idempotent).

        :param fname: path to the .dx occupancy map.
        :type fname: str
        :param probe_name: name of the probe, defaults to <RES>_<ATYPE> or <RES> parsed from the cosolvkit file name.
        :type probe_name: str, optional
        :param cosolvent: resname of the cosolvent the probe belongs to, defaults to the one parsed from the file name.
        :type cosolvent: str, optional
        """
        if os.path.basename(fname).startswith("map_agfe"):
            raise ValueError(f"{fname} is a free energy map. Hotspots are detected on occupancy (density) maps.")

        parsed = _probe_from_filename(fname)
        if parsed is None and probe_name is None:
            raise ValueError(f"Cannot infer the probe name from {fname}, please pass probe_name.")
        if probe_name is None:
            probe_name = parsed[0]
        if cosolvent is None:
            cosolvent = parsed[1] if parsed is not None else probe_name
        if probe_name in self._probes:
            raise ValueError(f"Probe {probe_name} was already added.")

        grid = _read_dx(fname)
        self._probes[probe_name] = {'fname': fname, 'cosolvent': cosolvent, 'grid': grid}
        self.logger.info(f"Added probe {probe_name} ({cosolvent}) from {fname}")
        return

    def _common_lattice(self):
        """Returns the origin and spacing of the lattice shared by all maps, used to detect overlapping spots."""
        grids = [probe['grid'] for probe in self._probes.values()]
        delta = np.asarray(grids[0].delta, dtype=float)
        origin = np.asarray(grids[0].origin, dtype=float)

        for name, probe in self._probes.items():
            if not np.allclose(probe['grid'].delta, delta):
                raise ValueError(f"Grid spacing of {name} ({probe['grid'].delta}) differs from {delta}. "
                                 "All maps must be generated with the same gridsize.")
            shift = (np.asarray(probe['grid'].origin) - origin) / delta
            misalignment = np.max(np.abs(shift - np.rint(shift)))
            if misalignment > 0.25:
                self.logger.warning(f"Grid of {name} is offset by {misalignment:.2f} grid points from the common lattice. "
                                    "Overlaps between spots are approximated to the nearest grid point.")
        return origin, delta

    def _points_above_cutoff(self, cutoff: float, normalize: str, origin: np.ndarray, delta: np.ndarray) -> pd.DataFrame:
        """Returns the grid points of every probe whose normalized occupancy is >= cutoff.
        The normalized value is stored in the 'zscore' column (a Z-score unless normalize='fraction')."""
        points = []
        for probe_name, probe in self._probes.items():
            grid = probe['grid']
            values = _normalize_occupancy(grid.grid, normalize)
            idx = np.argwhere(values >= cutoff)

            if len(idx) == 0:
                self.logger.warning(f"No grid points of probe {probe_name} have a {normalize} value >= {cutoff}.")
                continue

            xyz = np.column_stack([grid.midpoints[axis][idx[:, axis]] for axis in range(3)])
            lattice = np.rint((xyz - origin) / delta).astype(np.int64)
            points.append(pd.DataFrame({'probe': probe_name, 'cosolvent': probe['cosolvent'],
                                        'i': lattice[:, 0], 'j': lattice[:, 1], 'k': lattice[:, 2],
                                        'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2],
                                        'zscore': values[tuple(idx.T)]}))

        return pd.concat(points, ignore_index=True) if points else pd.DataFrame()

    @staticmethod
    def _spot_record(spot_id: int, probe_name: str, cosolvent: str, spot_points: pd.DataFrame, voxel_volume: float) -> dict:
        z = spot_points['zscore']
        return {'spot_id': spot_id,
                'probe': probe_name,
                'cosolvent': cosolvent,
                'score': z.sum(),
                'volume': len(spot_points) * voxel_volume,
                'n_points': len(spot_points),
                'max_z': z.max(),
                'x': np.average(spot_points['x'], weights=z),
                'y': np.average(spot_points['y'], weights=z),
                'z': np.average(spot_points['z'], weights=z)}

    def _find_spots(self, points: pd.DataFrame, cluster_cutoff: float, voxel_volume: float):
        """Clusters the grid points of every probe separately into spots (single linkage).
        Spots sharing at least one lattice point are then linked; the components are the hotspot candidates."""
        spots = []
        spot_ids = np.zeros(len(points), dtype=int)

        for probe_name, probe_points in points.groupby('probe', sort=False):
            labels = _cluster_points(probe_points[['x', 'y', 'z']].to_numpy(), cluster_cutoff)
            self.logger.info(f"Probe {probe_name}: {len(probe_points)} grid points above cutoff in {labels.max() + 1} spots")
            for label in np.unique(labels):
                mask = labels == label
                spot_id = len(spots)
                spots.append(self._spot_record(spot_id, probe_name, probe_points['cosolvent'].iloc[0],
                                               probe_points[mask], voxel_volume))
                spot_ids[points.index.get_indexer(probe_points.index[mask])] = spot_id

        spots = pd.DataFrame(spots)
        points = points.assign(spot_id=spot_ids)

        # Spots sharing at least one lattice point are linked, hotspots are the connected components
        sorted_points = points.sort_values(['i', 'j', 'k', 'spot_id'], kind='stable')
        keys = sorted_points[['i', 'j', 'k']].to_numpy()
        same_point = np.all(keys[1:] == keys[:-1], axis=1)
        ids = sorted_points['spot_id'].to_numpy()
        graph = coo_matrix((np.ones(same_point.sum(), dtype=bool), (ids[:-1][same_point], ids[1:][same_point])),
                           shape=(len(spots), len(spots)))
        _, component = connected_components(graph, directed=False)
        spots['component'] = component
        return spots, points

    def _find_pooled_clusters(self, points: pd.DataFrame, eps: float, min_samples: int, voxel_volume: float):
        """Clusters the grid points of all probes together with DBSCAN. Each DBSCAN cluster is a hotspot
        candidate; its points of each probe form one spot (the contribution of that probe). Noise points are dropped."""
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points[['x', 'y', 'z']].to_numpy()).labels_
        n_noise = int(np.sum(labels < 0))
        self.logger.info(f"DBSCAN (eps {eps} A, min_samples {min_samples}) on {len(points)} pooled grid points: "
                         f"{labels.max() + 1} clusters, {n_noise} noise points discarded")

        points = points.assign(cluster=labels)
        points = points[points['cluster'] >= 0].copy()
        spots = []
        spot_ids = np.zeros(len(points), dtype=int)
        for (cluster, probe_name), spot_points in points.groupby(['cluster', 'probe'], sort=True):
            spot_id = len(spots)
            record = self._spot_record(spot_id, probe_name, spot_points['cosolvent'].iloc[0], spot_points, voxel_volume)
            record['component'] = cluster
            spots.append(record)
            spot_ids[points.index.get_indexer(spot_points.index)] = spot_id

        points['spot_id'] = spot_ids
        return pd.DataFrame(spots), points.drop(columns='cluster')

    def run(self, z_cutoff: float = 20.0, cluster_cutoff: float = 3.0, min_spots: int = None, min_cosolvents: int = 1,
            method: str = "overlap", min_samples: int = 10, normalize: str = "zscore"):
        """Detects hotspots and ranks them by score (sum of the normalized occupancy over their points).

        Two ways of grouping the grid points of the different probes are available:

        - method="overlap": the points of each probe are clustered separately into spots (single linkage,
          cluster_cutoff); spots of any probes that share a grid point are merged into a hotspot.
        - method="dbscan": the points of all probes are pooled and clustered together with DBSCAN
          (eps=cluster_cutoff, min_samples); every cluster is a hotspot and the points of each probe in it
          are reported as that probe's spot. Points labeled as noise are discarded.

        :param z_cutoff: only grid points with a normalized occupancy >= z_cutoff are used, defaults to 20.0
            (a Z-score; use a fraction of the maximum, e.g. 0.1, with normalize="fraction")
        :type z_cutoff: float, optional
        :param cluster_cutoff: single-linkage distance cutoff (overlap) or DBSCAN eps (dbscan) in A, defaults to 3.0
        :type cluster_cutoff: float, optional
        :param min_spots: minimum number of spots per hotspot, defaults to 2 for overlap and 1 for dbscan
        :type min_spots: int, optional
        :param min_cosolvents: minimum number of distinct cosolvents mapping a hotspot, defaults to 1.
            Spots of different functional groups of the same cosolvent can form a hotspot with the default.
            Use 2 to require that the site is mapped by more than one probe molecule (Ghanakota & Carlson 2016).
        :type min_cosolvents: int, optional
        :param method: "overlap" (probes clustered separately) or "dbscan" (probes pooled), defaults to "overlap"
        :type method: str, optional
        :param min_samples: DBSCAN minimum number of points in the eps neighborhood of a core point, defaults to 10
        :type min_samples: int, optional
        :param normalize: "zscore", (x - mean) / sigma over the whole grid, or "fraction", x / max, defaults to "zscore"
        :type normalize: str, optional
        :return: hotspots table, ranked by score.
        :rtype: pd.DataFrame
        """
        if len(self._probes) == 0:
            self.logger.error("No occupancy maps were added. Use add_map() first.")
            sys.exit(1)
        if method not in ("overlap", "dbscan"):
            raise ValueError(f"Unknown method {method}, use 'overlap' or 'dbscan'.")
        if min_spots is None:
            min_spots = 2 if method == "overlap" else 1

        origin, delta = self._common_lattice()
        voxel_volume = np.prod(delta)
        points = self._points_above_cutoff(z_cutoff, normalize, origin, delta)

        self.spots = pd.DataFrame()
        self.hotspots = pd.DataFrame()
        self._hotspot_points = pd.DataFrame()
        if points.empty:
            self.logger.warning("No grid points above the cutoff, no hotspots to report.")
            return self.hotspots

        if method == "overlap":
            spots, points = self._find_spots(points, cluster_cutoff, voxel_volume)
        else:
            spots, points = self._find_pooled_clusters(points, cluster_cutoff, min_samples, voxel_volume)
            if spots.empty:
                self.logger.warning("DBSCAN found no clusters, no hotspots to report.")
                return self.hotspots
        self.spots = spots

        probe_names = list(self._probes)
        hotspots = []
        for comp_id, comp_spots in spots.groupby('component'):
            n_cosolvents = comp_spots['cosolvent'].nunique()
            if len(comp_spots) < min_spots or n_cosolvents < min_cosolvents:
                continue

            comp_points = points[points['spot_id'].isin(comp_spots['spot_id'])]
            unique_points = comp_points.groupby(['i', 'j', 'k']).agg(x=('x', 'mean'), y=('y', 'mean'), z=('z', 'mean'),
                                                                     zscore=('zscore', 'sum')).reset_index()
            weights = unique_points['zscore']
            record = {'component': comp_id,
                      'score': comp_spots['score'].sum(),
                      'volume': len(unique_points) * voxel_volume,
                      'n_spots': len(comp_spots),
                      'n_probes': comp_spots['probe'].nunique(),
                      'n_cosolvents': n_cosolvents,
                      'probes': ','.join(sorted(comp_spots['probe'].unique())),
                      'max_z': comp_spots['max_z'].max(),
                      'x': np.average(unique_points['x'], weights=weights),
                      'y': np.average(unique_points['y'], weights=weights),
                      'z': np.average(unique_points['z'], weights=weights)}
            # contribution of each probe to the score, and its highest value in the hotspot
            by_probe = comp_spots.groupby('probe').agg(score=('score', 'sum'), max_z=('max_z', 'max'))
            for probe_name in probe_names:
                record[f'score_{probe_name}'] = by_probe['score'].get(probe_name, 0.0)
                record[f'max_{probe_name}'] = by_probe['max_z'].get(probe_name, 0.0)
            hotspots.append(record)

        if not hotspots:
            self.logger.warning(f"Found {len(spots)} spots but none of them form hotspots.")
            spots['hotspot'] = -1
            spots.drop(columns='component', inplace=True)
            return self.hotspots

        hotspots = pd.DataFrame(hotspots).sort_values('score', ascending=False).reset_index(drop=True)
        hotspots.insert(0, 'hotspot', hotspots.index)

        rank_by_component = dict(zip(hotspots['component'], hotspots['hotspot']))
        spots['hotspot'] = spots['component'].map(rank_by_component).fillna(-1).astype(int)
        spots.drop(columns='component', inplace=True)
        hotspots.drop(columns='component', inplace=True)

        points = points.merge(spots[['spot_id', 'hotspot']], on='spot_id')
        self._hotspot_points = points[points['hotspot'] >= 0] \
            .groupby(['hotspot', 'i', 'j', 'k']).agg(x=('x', 'mean'), y=('y', 'mean'), z=('z', 'mean'),
                                                     zscore=('zscore', 'sum')).reset_index()

        self.spots = spots
        self.hotspots = hotspots
        self.logger.info(f"Method {method}: {len(spots)} spots and {len(hotspots)} hotspots.")
        for _, hotspot in hotspots.head(10).iterrows():
            self.logger.info(f"Hotspot {hotspot['hotspot']}: score {hotspot['score']:.0f}, volume {hotspot['volume']:.2f} A^3, "
                             f"probes {hotspot['probes']}")

        return self.hotspots

    def _residues_near_hotspots(self, reference_structure: str, distance: float = 4.0, selection: str = "protein"):
        """Lists the residues of the reference structure with any atom within distance of each hotspot grid point."""
        from MDAnalysis import Universe

        atoms = Universe(reference_structure).select_atoms(selection)
        tree = cKDTree(atoms.positions)
        residues = {}

        for hotspot_id, hotspot_points in self._hotspot_points.groupby('hotspot'):
            neighbors = tree.query_ball_point(hotspot_points[['x', 'y', 'z']].to_numpy(), r=distance)
            atom_idx = np.unique(np.concatenate([np.asarray(n, dtype=int) for n in neighbors]))
            hotspot_residues = atoms[atom_idx].residues
            residues[hotspot_id] = ' '.join(f"{res.resname}{res.resid}" for res in hotspot_residues)

        return residues

    def _write_pdb(self, fname: str):
        """Writes one pseudo-atom per hotspot grid point. The residue number is the hotspot rank and
        the B-factor is the summed Z-score of the overlapping spots at that point."""
        lines = []
        for serial, point in enumerate(self._hotspot_points.itertuples(index=False), start=1):
            lines.append(f"HETATM{serial % 100000:5d}  C   HSP H{point.hotspot % 10000:4d}    "
                         f"{point.x:8.3f}{point.y:8.3f}{point.z:8.3f}{1.0:6.2f}{min(point.zscore, 999.99):6.2f}           C")
        lines.append("END")
        with open(fname, "w") as fo:
            fo.write("\n".join(lines) + "\n")
        return

    def _write_pymol_script(self, fname: str, pdb_fname: str, reference_structure: str = None, n_hotspots: int = 10):
        """Writes a PyMOL script showing the top hotspots as surfaces, numbered by rank."""
        colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta', 'salmon', 'wheat']
        cmd_string = ""
        if reference_structure is not None:
            cmd_string += f"load {os.path.abspath(reference_structure)}, reference\n"
            cmd_string += "hide everything, reference\nshow cartoon, reference\ncolor grey80, reference\n"
        cmd_string += f"load {os.path.abspath(pdb_fname)}, hotspots\n"
        cmd_string += "hide everything, hotspots\nalter hotspots, vdw=0.35\nrebuild\n"

        for hotspot_id in self.hotspots['hotspot'].head(n_hotspots):
            color = colors[hotspot_id % len(colors)]
            cmd_string += f"create hotspot_{hotspot_id}, hotspots and resi {hotspot_id}\n"
            cmd_string += f"show surface, hotspot_{hotspot_id}\ncolor {color}, hotspot_{hotspot_id}\n"
        cmd_string += "set transparency, 0.3\nbg_color white\n"

        with open(fname, "w") as fo:
            fo.write(cmd_string)
        return

    def export(self, out_path: str, reference_structure: str = None, residue_distance: float = 4.0, n_hotspots_pymol: int = 10):
        """Writes the hotspot and spot tables, a PDB file of the hotspot grid points and a PyMOL script.

        :param out_path: output directory.
        :type out_path: str
        :param reference_structure: structure in the same frame as the maps (e.g. averaged_trajectory.pdb),
            used to list the residues lining each hotspot and in the PyMOL script, defaults to None
        :type reference_structure: str, optional
        :param residue_distance: distance (A) from the hotspot grid points to list residues, defaults to 4.0
        :type residue_distance: float, optional
        :param n_hotspots_pymol: number of top-ranked hotspots shown in the PyMOL script, defaults to 10
        :type n_hotspots_pymol: int, optional
        """
        if self.spots is None:
            self.logger.error("No results to export. Use run() first.")
            sys.exit(1)

        os.makedirs(out_path, exist_ok=True)
        if not self.spots.empty:
            self.spots.to_csv(os.path.join(out_path, "spots.csv"), index=False, float_format="%.3f")
        if self.hotspots.empty:
            self.logger.warning("No hotspots to export.")
            return

        hotspots = self.hotspots.copy()
        if reference_structure is not None:
            residues = self._residues_near_hotspots(reference_structure, residue_distance)
            hotspots['residues'] = hotspots['hotspot'].map(residues)

        hotspots.to_csv(os.path.join(out_path, "hotspots.csv"), index=False, float_format="%.3f")

        pdb_fname = os.path.join(out_path, "hotspots.pdb")
        self._write_pdb(pdb_fname)
        self._write_pymol_script(os.path.join(out_path, "hotspots.pml"), pdb_fname, reference_structure, n_hotspots_pymol)
        self.logger.info(f"Hotspot results written to {out_path}")
        return


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Detects and scores hotspots from cosolvkit occupancy maps")
    parser.add_argument('-m', '--maps', dest='maps', nargs='+', default=[],
                        action='store', help='occupancy maps (map_density_*.dx, map_rawdensity_*.dx or map_com_*.dx)')
    parser.add_argument('-d', '--dirs', dest='dirs', nargs='+', default=[],
                        action='store', help='directories with the occupancy maps generated by cosolvkit')
    parser.add_argument('-o', '--out_path', dest='out_path', required=True,
                        action='store', help='path where to store output files')
    parser.add_argument('-r', '--reference', dest='reference', default=None,
                        action='store', help='reference structure in the same frame as the maps')
    parser.add_argument('-s', '--structures', dest='structures', nargs='+', default=[],
                        action='store', help='one structure per probe simulation to check that the maps share the same frame')
    parser.add_argument('-k', '--map_kinds', dest='map_kinds', nargs='+', default=['atomtype', 'raw'], choices=MAP_KINDS,
                        action='store', help='kinds of maps collected from --dirs: atomtype (map_density_<ATYPE>_<RES>), '
                                             'raw (map_rawdensity_<RES>), com (map_com_<RES>) (default: atomtype raw)')
    parser.add_argument('--method', dest='method', default='overlap', choices=('overlap', 'dbscan'),
                        action='store', help='overlap: cluster each probe separately and merge overlapping spots; '
                                             'dbscan: pool the grid points of all probes and cluster them together (default: overlap)')
    parser.add_argument('--normalize', dest='normalize', default='zscore', choices=('zscore', 'fraction'),
                        action='store', help='occupancy normalization: Z-score or fraction of the maximum (default: zscore)')
    parser.add_argument('--z_cutoff', dest='z_cutoff', type=float, default=20.0,
                        action='store', help='cutoff on the normalized occupancy of grid points (default: 20)')
    parser.add_argument('--cluster_cutoff', dest='cluster_cutoff', type=float, default=3.0,
                        action='store', help='single-linkage cutoff (overlap) or DBSCAN eps (dbscan) in A (default: 3.0)')
    parser.add_argument('--min_samples', dest='min_samples', type=int, default=10,
                        action='store', help='DBSCAN min_samples (default: 10)')
    parser.add_argument('--min_spots', dest='min_spots', type=int, default=None,
                        action='store', help='minimum number of spots per hotspot (default: 2 for overlap, 1 for dbscan)')
    parser.add_argument('--min_cosolvents', dest='min_cosolvents', type=int, default=1,
                        action='store', help='minimum number of distinct cosolvents per hotspot (default: 1)')
    return parser.parse_args()


def main():
    from cosolvkit.utils import setup_logging

    args = cmd_lineparser()
    os.makedirs(args.out_path, exist_ok=True)
    setup_logging(level="INFO", filepath=os.path.join(args.out_path, "cosolvkit.log"))

    if len(args.structures) > 1:
        rmsd = check_frame_consistency(args.structures)
        logger.info(f"No-fit RMSD between structures (A):\n{rmsd.round(2)}")

    maps = args.maps + (find_density_maps(args.dirs, kinds=args.map_kinds) if args.dirs else [])
    if not maps:
        logger.error("No occupancy maps provided or found.")
        return 1

    hotspots = HotspotAnalysis()
    for fname in maps:
        hotspots.add_map(fname)
    hotspots.run(z_cutoff=args.z_cutoff, cluster_cutoff=args.cluster_cutoff, min_spots=args.min_spots,
                 min_cosolvents=args.min_cosolvents, method=args.method, min_samples=args.min_samples,
                 normalize=args.normalize)
    hotspots.export(args.out_path, reference_structure=args.reference)

    return 0


if __name__ == "__main__":
    sys.exit(main())
