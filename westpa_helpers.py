import yaml
import json
import pickle
import os
import sys
import numpy as np
import mdtraj
import deeptime
import glob
import matplotlib.pyplot as plt
import tempfile
from tqdm import tqdm
import h5py
import pdbfixer
from openmm.app import PDBFile, ForceField, Modeller

def extract_simulation_config(cfg_path="west.cfg"):
    with open(cfg_path, "r") as file:
        yaml_content = file.read()
 
    config = yaml.load(yaml_content, Loader=yaml.UnsafeLoader)

    if not config or 'west' not in config:
        raise ValueError("YAML structure invalid or missing top-level 'west' key.")

    west_section = config['west']
    cg_prop = west_section.get('cg_prop', {})
    pcoord_calc = cg_prop.get('pcoord_calculator', {})

    extracted_data = {
        "model_path": cg_prop.get('model_path'),
        "cgschnet_path": cg_prop.get('cgschnet_path'),
        "topology_path": cg_prop.get('topology_path'),
        "components": pcoord_calc.get('components'),
        "tica_model_path": pcoord_calc.get('model_path')
    }

    return extracted_data

def extract_all_atom_simulation_config(cfg_path="west_openmm.cfg"):
    with open(cfg_path, "r") as file:
        yaml_content = file.read()
 
    config = yaml.load(yaml_content, Loader=yaml.UnsafeLoader)

    if not config or 'west' not in config:
        raise ValueError("YAML structure invalid or missing top-level 'west' key.")

    west_section = config['west']
    cg_prop = west_section.get('openmm', {})
    pcoord_calc = cg_prop.get('pcoord_calculator', {})

    extracted_data = {
        "model_path": cg_prop.get('model_path'),
        "cgschnet_path": cg_prop.get('cgschnet_path'),
        "topology_path": cg_prop.get('topology_path'),
        "components": pcoord_calc.get('components'),
        "tica_model_path": pcoord_calc.get('model_path')
    }

    return extracted_data

def convert_to_mdtraj_topology(cg_mol):
    with tempfile.TemporaryDirectory() as tmpdirname:
        topology_path = os.path.join(tmpdirname, "topology.pdb")
        cg_mol.write(topology_path)
        topology = mdtraj.load(topology_path).top
        return topology

def create_cg_topology_from_all_atom(simulation_config):
    cgschnet_path = simulation_config["cgschnet_path"] 
    if not cgschnet_path in sys.path:
        sys.path.append(cgschnet_path)
    import simulate  # pyright: ignore[reportMissingImports]

    checkpoint_path = simulation_config["model_path"] 

    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "checkpoint-best.pth")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    assert os.path.exists(checkpoint_path)

    prior_path = os.path.join(checkpoint_dir, "priors.yaml")
    assert os.path.exists(prior_path)
    prior_params_path = os.path.join(checkpoint_dir, "prior_params.json")

    with open(f"{prior_params_path}", 'r') as file:
        prior_params = json.load(file)

    mol, embeddings = simulate.load_molecule(
        prior_path, prior_params, simulation_config["topology_path"], use_box=False, verbose=False)

    return mol


def load_trajectories(coordinate_files, size_limit=None):
    coordinate_list = []
    label_list = []

    for cf in tqdm(coordinate_files):
        batch_label = os.path.basename(cf)
        batch_traj = []
        for subtraj in tqdm(glob.glob(cf)):
            if subtraj.endswith("npy"):
                coords = np.load(subtraj, allow_pickle=True)
                if type(coords) == dict: # One of Raz's benchmark archives
                    batch_label = os.path.join(*(cf.split(os.path.sep)[-2:]))
                    batch_traj.extend(coords["mdtraj_list"])
                else: # A preprocess.py output file
                    batch_label = os.path.join(*(cf.split(os.path.sep)[-3:]))
                    # Convert to NM to match mdtraj coordinates
                    coords = coords/10
                    psf_path = glob.glob(os.path.join(os.path.dirname(cf),"../processed/*_processed.psf"))[0]
                    traj = mdtraj.Trajectory(coords, topology=mdtraj.load_psf(psf_path))
                    batch_traj.append(traj)
            else: # Something mdtraj can open
                traj = mdtraj.load(subtraj)
                batch_traj.append(traj)
        if len(batch_traj) == 0:
            raise RuntimeError(f"{cf} did not match any files")
        
        batch_traj = mdtraj.join(batch_traj)

        # Select with a stride that brings the total number of frames down to the size_limit
        if size_limit and len(batch_traj) > size_limit:
            batch_traj = batch_traj[::(len(batch_traj)//size_limit)]

        label_list.append(batch_label)
        coordinate_list.append(batch_traj)

    assert len(coordinate_list) == len(label_list)
    return coordinate_list, label_list

def load_tica_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
        assert hasattr(model, "tica_model")
        return model

def calculate_component_values(model, coordinates, components):
    # Returns an object of type list(dict(array)) : [trajectory, component, component_values_for_frames]
    component_values = {k: [] for k in components}

    pairs = np.vstack(np.triu_indices(coordinates.n_atoms, k=1)).T
    distances = mdtraj.compute_distances(coordinates, pairs)
    tica_comps = model.tica_model.transform(distances)
    for k, v in component_values.items():
        v.extend(tica_comps[:, k])

    return component_values

def shorten_label(label, maxlen):
    if len(label) > maxlen:
        return label[:maxlen-3] + "..."
    return label

def get_traj(traj_loc, topology, cut):
    """
    Load a WESTPA segment (DCD or NPZ) and return an mdtraj.Trajectory.
    For NPZ, we assume the array stored under the key 'pos' is in Å.
    """
    ext = os.path.splitext(traj_loc)[1]
    if ext == ".dcd":
        traj = mdtraj.load_dcd(traj_loc, top=topology)
    elif ext == ".npz":
        with np.load(traj_loc) as d:
            xyz_nm = d["pos"] * 0.1          # Å → nm
            traj = mdtraj.Trajectory(xyz_nm, topology)
            if "box_vectors" in d:           # optional unit-cell info
                traj.unitcell_vectors = d["box_vectors"] * 0.1
    else:
        raise ValueError(f"Unsupported segment type: {ext}")

    if cut:
        traj = traj[traj.n_frames // 2 : traj.n_frames // 2 + 1]
    return traj
   

def iter_name(n):
    return f"iter_{n:08d}"

def iter_num_convert(n):
    return f"{n:06d}"

def get_weights_for_iteration(h5_file: h5py.File, iteration: int) -> np.ndarray:
    """Get segment weights from a specific iteration."""
    dataset = h5_file[f'iterations/{iter_name(iteration)}/seg_index']
    assert isinstance(dataset, h5py.Dataset)
    weights = dataset['weight'] 
    assert isinstance(weights, np.ndarray)
    return weights

def get_dcds_for_iteration(index_loc):
    seg_dcds = [os.path.join(index_loc, seg, "seg.dcd") for seg in os.listdir(index_loc)]
    return seg_dcds

def get_seg_files_for_iteration(index_loc: str, ext: str = "dcd") -> list[str]:
    """More generalized to work for both dcd and npz."""
    pattern = f"seg.{ext}"
    return [os.path.join(index_loc, seg, pattern)
            for seg in os.listdir(index_loc)]

def load_all_weights_and_trajs_flat(h5_path: str, root_path: str, ext: str = "dcd") -> tuple[np.ndarray, list[str]]:
    """
    Extract all segment weights from all iterations in order,
    and return them as a 1D numpy array along with
    (iteration, seg_id) tracking for debugging and mapping when integrating with benchmark code.
    """
    all_weights_flat = []
    ordered_traj_locs = []
    trajs_path = os.path.join(root_path, "traj_segs")
    

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found at: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        group = f['iterations']
        assert isinstance(group, h5py.Group)
        iterations = sorted(int(key.split('_')[1]) for key in group.keys())
        iterations.pop() #Remove last, usually its incomplete
        for iter_num in iterations:
            index_loc = os.path.join(trajs_path, iter_num_convert(iter_num))
            if not os.path.isdir(index_loc):
                print(f"Iteration {iter_num} directory {index_loc} missing; skipping.")
                continue

            seg_files = get_seg_files_for_iteration(index_loc, ext)
            iter_weights = get_weights_for_iteration(f, iter_num)

            if len(seg_files) != len(iter_weights):
                print(f"Mismatch in number of segment files ({len(seg_files)}) "
                    f"and weights ({len(iter_weights)}) for iteration {iter_num}; skipping.")
                continue
            all_weights_flat.extend(iter_weights)
            ordered_traj_locs.extend(seg_files)
        
    assert len(ordered_traj_locs) == len(all_weights_flat), \
        f"Total number of segment files ({len(ordered_traj_locs)}) does not match total number of weights ({len(all_weights_flat)})"
    return np.array(all_weights_flat), ordered_traj_locs

def get_traj_locs_inorder(root_path: str, ext: str = "dcd") -> list[str]:
    ordered_traj_locs = []
    trajs_path = os.path.join(root_path, "traj_segs")
    for index in os.listdir(trajs_path):
        index_loc = os.path.join(trajs_path, index)
        for seg in os.listdir(index_loc):
            seg_loc = os.path.join(index_loc, seg)
            ordered_traj_locs.append(os.path.join(seg_loc, f"seg.{ext}")) #changed to be more general to file type
            
    return ordered_traj_locs

def get_topology_from_westpa(root_path: str, ext: str = "dcd") -> mdtraj.Topology:
    cfg_path = os.path.join(root_path, "west.cfg")
    with open(cfg_path, "r") as file:
        yaml_content = file.read()

    config = yaml.load(yaml_content, Loader=yaml.UnsafeLoader)
    west = config["west"]

    # First check CG block, then OpenMM (AA) as a fallback
    topology_path = (
        west.get("cg_prop", {}).get("topology_path") or
        west.get("openmm", {}).get("topology_path")
    )
    if topology_path is None or not os.path.exists(topology_path):
        raise FileNotFoundError(f"Topology file not found in west.cfg ({cfg_path})")

    topology = mdtraj.load(topology_path).topology

    # Optional CA‑only reduction
    # do this only if the file is all‑atom *and* the npz coords we read later are CA‑only
    if ext == "npz":
        topology = mdtraj.load(topology_path).topology
        ca_inds = [a.index for a in topology.atoms if a.name == "CA"]
        print(f"Original topology atoms: {topology.n_atoms}")
        print(f"CA atoms found: {len(ca_inds)}")
        if ca_inds: 
            topology = topology.subset(ca_inds)
            print(f"Final topology atoms: {topology.n_atoms}")

    return topology

def get_implicit_topology_from_westpa(root_path: str) -> mdtraj.Topology:
    cfg_path = os.path.join(root_path, "west_openmm.cfg")
    with open(cfg_path, "r") as file:
        yaml_content = file.read()

    config = yaml.load(yaml_content, Loader=yaml.UnsafeLoader)
    topology_path = config['west']['openmm']['topology_path']
    forcefield_files = config['west']['openmm']['forcefield']


    topology_path = os.path.expandvars(topology_path)
    pdb = PDBFile(topology_path)
    forcefield = ForceField(*forcefield_files)

    fixer = pdbfixer.PDBFixer(topology_path)

    # find missing residues and atoms
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    print(f"Missing residues: {fixer.missingResidues}")
    print(f"Missing terminals: {fixer.missingTerminals}")
    print(f"Missing atoms: {fixer.missingAtoms}")

    # remove missing residues at the terminal
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    # check if the terminal residues are removed
    for key in list(keys):
        chain = chains[key[0]]
        assert key[1] != 0 and key[1] != len(list(chain.residues())), "Terminal residues are not removed."

    # find and replace nonstandard residues
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    # add missing atoms and hydrogens
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    # use modeller to clean up
    modeller = Modeller(fixer.topology, fixer.positions)

    modeller.deleteWater()
    ions_to_delete = [res for res in modeller.topology.residues() if res.name in ('NA', 'CL')]
    modeller.delete(ions_to_delete)

    pdb.topology = modeller.getTopology()
    pdb.positions = modeller.getPositions()

    topo_mdtraj = mdtraj.Topology.from_openmm(pdb.topology)
    return topo_mdtraj



# def extend_weights(westpa_weights, frames_per_traj):
#     weights_extended = []
#     for i in range(len(westpa_weights)):
#         weights_extended.extend([westpa_weights[i]] * frames_per_traj)
#     return np.array(weights_extended)

def extend_weights(westpa_weights, per_traj_objs):
    """
    westpa_weights: 1D array, len = #traj
    per_traj_objs : list of arrays (e.g. tica outputs) or list of frame counts
    """
    try:
        # list of arrays
        lengths = [x.shape[0] for x in per_traj_objs]
    except AttributeError:
        # list of ints
        lengths = per_traj_objs
    return np.repeat(westpa_weights, lengths)



if __name__ == "__main__":
    general_path = "/media/DATA_18_TB_1/awaghili/WESTPA_MM/mabbin_chignolin_2/westpa_prop"

