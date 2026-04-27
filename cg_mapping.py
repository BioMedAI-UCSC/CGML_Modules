import re
import warnings

import mdtraj
import numpy as np
from moleculekit.molecule import Molecule
from aggforce import LinearMap, project_forces #type: ignore

def extend_objects(objects, bonds):
    """Forms either angles (if given a list of bonds) or dihedrals (if given a list of angles)
       by looking for a bond that can extend each object. It makes no assumptions about the
       ordering of elements in either list."""

    result = set() # There will be duplicates

    def sort_key(k):
        if k[0] > k[-1]:
            return tuple(reversed(k))
        return tuple(k)

    # loops thru list of bonds/angles
    for a in objects:
        a = list(a)
        for b in bonds:
            b = list(b)
            if a == b:
                continue
            k = None
            if a[0] == b[-1]:
                k = b[:] + a[1:]
            elif a[0] == b[0]:
                k = list(reversed(b[:])) + a[1:]
            elif a[-1] == b[0]:
                k = a[:] + b[1:]
            elif a[-1] == b[-1]:
                k = b[:] + list(reversed(a))[1:]

            if k is not None:
                # An atom can't occur twice in an object
                if max([sum([i == j for j in k]) for i in k]) > 1:
                    continue
                result.add(sort_key(k))

    return list(result)

class CGMapping:
    def __init__(self, topology, map_def):
        """Generate a CG mapping from an all atom MDTraj topology and a CGMappingDef definition"""
        self.src_idx = []
        self.pos_weights = []
        self.force_weights = []

        self.embeddings = []

        self.bead_atom_names = []
        self.bead_types = []
        self.bead_mass = []

        self.cg_topology = mdtraj.Topology()

        mappable_residues = set(map_def.bead_atom_selection.keys())
        dna_mods = getattr(map_def, "dna_modifications", None) or {}
        dna_residue_names = frozenset(["DA", "DT", "DG", "DC"])

        def strip_resname(rname: str) -> str:
            return re.sub(r"\d+$", "", rname)

        def clean_atom_name(name: str) -> str:
            s = str(name).split("-")[-1]
            return s.replace("*", "'")

        def _bead_indices(imap: dict, atom_names) -> list | None:
            out: list = []
            for an in atom_names:
                if an not in imap:
                    return None
                out.append(imap[an])
            return out

        for chain in topology.chains:
            if not any(
                (strip_resname(r.name) in mappable_residues) or (strip_resname(r.name) in dna_mods)
                for r in chain.residues
            ):
                continue

            result_chain = self.cg_topology.add_chain()
            last_backbone_idx = None
            chain_protein_mode = 0

            for res in chain.residues:
                res_name = strip_resname(res.name)
                mapped_resname: str | None
                if res_name in mappable_residues:
                    mapped_resname = res_name
                elif res_name in dna_mods:
                    parent = dna_mods[res_name]
                    if parent in mappable_residues:
                        mapped_resname = str(parent)
                    else:
                        mapped_resname = None
                else:
                    mapped_resname = None

                if mapped_resname is None:
                    if chain_protein_mode == 1:
                        chain_protein_mode = 2
                    continue
                if chain_protein_mode == 0:
                    chain_protein_mode = 1
                elif chain_protein_mode == 2:
                    warnings.warn(
                        f"Non-contiguous chain {chain.index}: residue {res.name} after a gap of "
                        f"unmappable residues",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                idx_mapping = {clean_atom_name(a.name): a.index for a in res.atoms}
                bead_mapping = map_def.bead_atom_selection[mapped_resname]
                backbone_idx = self.cg_topology.n_atoms
                all_bead_indices: list
                bbcand = getattr(map_def, "dna_backbone_atom_candidates", None)
                if (
                    bbcand
                    and mapped_resname in dna_residue_names
                    and len(bead_mapping) >= 1
                ):
                    first: list | None = None
                    for alt in bbcand:
                        first = _bead_indices(idx_mapping, alt)
                        if first is not None:
                            break
                    if first is None:
                        continue
                    rest: list = []
                    ok = True
                    for bead in bead_mapping[1:]:
                        bidx = _bead_indices(idx_mapping, bead)
                        if bidx is None:
                            ok = False
                            break
                        rest.append(bidx)
                    if not ok:
                        continue
                    all_bead_indices = [first] + rest
                else:
                    all_bead_indices = []
                    valid = True
                    for bead in bead_mapping:
                        bidx = _bead_indices(idx_mapping, bead)
                        if bidx is None:
                            valid = False
                            break
                        all_bead_indices.append(bidx)
                    if not valid or len(all_bead_indices) != len(bead_mapping):
                        continue

                result_res = self.cg_topology.add_residue(res.name, result_chain)
                for bead_name in map_def.bead_atom_names[mapped_resname]:
                    if bead_name == "DBB":
                        element = mdtraj.element.phosphorus
                    elif str(bead_name).startswith("DB"):
                        element = mdtraj.element.nitrogen
                    elif str(bead_name).startswith("CA"):
                        element = mdtraj.element.carbon
                    else:
                        fc = str(bead_name)[0].upper() if bead_name else "C"
                        elmap = {
                            "P": mdtraj.element.phosphorus,
                            "N": mdtraj.element.nitrogen,
                            "C": mdtraj.element.carbon,
                        }
                        element = elmap.get(fc, mdtraj.element.carbon)
                    self.cg_topology.add_atom(bead_name, element, result_res)
                self.bead_atom_names.extend(map_def.bead_atom_names[mapped_resname])
                self.bead_types.extend(map_def.bead_types[mapped_resname])
                self.bead_mass.extend(map_def.bead_masses[mapped_resname])
                self.embeddings.extend(map_def.bead_embeddings[mapped_resname])
                for bead_idx in all_bead_indices:
                    self.src_idx.append(bead_idx)
                    bead_w = np.array([topology.atom(i).element.mass for i in bead_idx])
                    bead_w = (bead_w / np.sum(bead_w)).tolist()
                    self.pos_weights.append(bead_w)
                    self.force_weights.append(bead_w)
                if last_backbone_idx is not None:
                    self.cg_topology.add_bond(
                        self.cg_topology.atom(last_backbone_idx), self.cg_topology.atom(backbone_idx)
                    )
                for i in range(len(all_bead_indices) - 1):
                    self.cg_topology.add_bond(
                        self.cg_topology.atom(backbone_idx + i), self.cg_topology.atom(backbone_idx + i + 1)
                    )
                last_backbone_idx = backbone_idx

    def to_mol(self, bonds=True, angles=True, dihedrals=True):
        """Generate a moleculekit Molecule object for the CG topology"""
        mol = Molecule()

        mol.serial = np.arange(self.cg_topology.n_atoms)+1                                                 #pyright: ignore[reportAttributeAccessIssue]
        mol.segid = np.array([str(a.residue.chain.index) for a in self.cg_topology.atoms], dtype=object)   #pyright: ignore[reportAttributeAccessIssue]
        mol.insertion = np.full((self.cg_topology.n_atoms,), '', dtype=object)                             #pyright: ignore[reportAttributeAccessIssue]
        mol.chain = np.copy(mol.segid) # Previously this was left as ''                                    #pyright: ignore[reportAttributeAccessIssue]
        mol.resid = np.array([a.residue.index+1 for a in self.cg_topology.atoms])                          #pyright: ignore[reportAttributeAccessIssue]
        mol.insertion = np.full((self.cg_topology.n_atoms,), '', dtype=object)                             #pyright: ignore[reportAttributeAccessIssue]

        # Requried to make pdbs write correctly
        mol.occupancy = np.full((self.cg_topology.n_atoms,), 1.0, dtype=np.float32)                       #pyright: ignore[reportAttributeAccessIssue]
        mol.beta = np.full((self.cg_topology.n_atoms,), 0.0, dtype=np.float32)                            #pyright: ignore[reportAttributeAccessIssue]
        mol.record = np.full((self.cg_topology.n_atoms,), 'ATOM', dtype=object)                           #pyright: ignore[reportAttributeAccessIssue]
        mol.altloc = np.full((self.cg_topology.n_atoms,), '', dtype=object)                               #pyright: ignore[reportAttributeAccessIssue]
        # PSF: CG bead names; DNA DBB/DB* use P/N for writer compatibility (cgschnet)
        mol.element = np.array(
            [
                "P" if n == "DBB" else "N" if str(n).startswith("DB") else "C" if str(n).startswith("CA") else "C"
                for n in self.bead_atom_names
            ],
            dtype=object,
        )  # pyright: ignore[reportAttributeAccessIssue]
        mol.atomicnumber = np.array(
            [15 if n == "DBB" else 7 if str(n).startswith("DB") else 6 for n in self.bead_atom_names],
            dtype=np.int32,
        )
        disp_masses: list[float] = []
        for n in self.bead_atom_names:
            if n == "DBB":
                disp_masses.append(30.97)
            elif str(n).startswith("DB"):
                disp_masses.append(14.01)
            else:
                disp_masses.append(12.01)
        mol.masses = np.array(disp_masses, dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]
        mol.formalcharge = np.full((self.cg_topology.n_atoms,), 0, dtype=np.int32)                        #pyright: ignore[reportAttributeAccessIssue]

        # The output psf contains resname=res_abbr, name=CA, atomtype=bead_type
        mol.name = np.array(self.bead_atom_names, dtype=object)                                           #pyright: ignore[reportAttributeAccessIssue]
        mol.atomtype = np.array(self.bead_types, dtype=object)                                            #pyright: ignore[reportAttributeAccessIssue]
        mol.resname = np.array([a.residue.name for a in self.cg_topology.atoms], dtype=object)            #pyright: ignore[reportAttributeAccessIssue]

        mol.charge = np.full((self.cg_topology.n_atoms,), 0)                                              #pyright: ignore[reportAttributeAccessIssue]
        mol._cg_bead_masses = np.array(self.bead_mass, dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]

        mol.box = np.zeros((3, 0), dtype=np.float32)

        mol.bonds = np.empty((0,2))
        mol.angles = np.empty((0,3))                                                                      #pyright: ignore[reportAttributeAccessIssue]
        mol.dihedrals = np.empty((0,4))                                                                   #pyright: ignore[reportAttributeAccessIssue]

        bonds_to_write = [[b.atom1.index, b.atom2.index] for b  in self.cg_topology.bonds]
        angles_to_write = extend_objects(bonds_to_write, bonds_to_write)
        angles_to_write.sort(key=lambda x: x[0])
        dihedrals_to_write = extend_objects(bonds_to_write, angles_to_write)
        dihedrals_to_write.sort(key=lambda x: x[0])

        if bonds:
            mol.bonds = np.array(bonds_to_write)
        if angles:
            mol.angles = np.array(angles_to_write)
        if dihedrals:
            mol.dihedrals = np.array(dihedrals_to_write)

        return mol

    def to_mdtraj(self):
        """Generate a MDTraj topology object for the CG topology"""
        return self.cg_topology.copy()

    def cg_forces(self, aa_forces):
        """Map all atom forces to CG forces"""
        return self._do_mapping(aa_forces, self.force_weights)

    def cg_positions(self, aa_positions):
        """Map all atom positions to CG forces"""
        return self._do_mapping(aa_positions, self.pos_weights)

    def _do_mapping(self, aa_input, mapping_weights):
        # Apply a weighted mapping to the aa_input (positions or forces)
        num_beads = len(self.src_idx)
        # For each input atom define the bead index it contributes to
        bead_targets = np.concatenate([[i]*len(self.src_idx[i]) for i in range(num_beads)])
        # Flatten the input indices and weights
        bead_idx = np.concatenate(self.src_idx)
        mapping_weights = np.concatenate(mapping_weights, dtype=np.float32)

        num_frames = len(aa_input)
        # Change the axis ordering to (atom, frame, xyz)
        aa_input = aa_input.swapaxes(0,1)[bead_idx]
        weighted_coords = aa_input*mapping_weights[:,None,None]

        bead_output = np.zeros((num_beads, num_frames, 3), dtype=np.float32)
        np.add.at(bead_output, bead_targets, weighted_coords)
        # Change the axis ordering back to (frame, atom, xyz)
        bead_output = bead_output.swapaxes(0,1)
        return bead_output

    def cg_optimal_forces(self, aa_trajectory, aa_forces):
        # get data
        coords = aa_trajectory.xyz

        # Create coordinate mapping and set up bond constraints from all Hydrogen bonds
        cmap = LinearMap(self.src_idx, n_fg_sites=coords.shape[1])
        bonds_array = np.array([(bond.atom1.index, bond.atom2.index)
                                for bond in aa_trajectory.topology.bonds
                                if bond.atom1.element.symbol == 'H'
                                or bond.atom2.element.symbol == 'H'])
        constraintsUnzip = np.array(bonds_array)
        constraints = {frozenset(v) for v in constraintsUnzip}

        # Basic mapping
        # basic_results = project_forces(
        #     forces=forces,
        #     constrained_inds=constraints,
        #     method=constraint_aware_uni_map,
        #     coords=coords,
        #     coord_map=cmap
        # )

        # Statistically optimal mapping
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=r"Converted [PA] to scipy\.sparse\.csc\.csc_matrix")
            optim_results = project_forces(
                forces=aa_forces,
                constrained_inds=constraints,
                coords=coords,
                coord_map=cmap
            )

        # Select only the forces from the results
        return optim_results['mapped_forces'].astype(np.float32)
