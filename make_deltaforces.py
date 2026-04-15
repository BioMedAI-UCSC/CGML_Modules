import multiprocessing as mp
import time

import numpy as np
import torch
from tqdm import tqdm
from moleculekit.molecule import Molecule
from torchmd.forces import Forces
from torchmd.parameters import Parameters
from torchmd.systems import System

# from torchmd.forcefields.forcefield import ForceField
from module.external_nn import ExternalNN, ParametersNN
from module.torchmd import tagged_forcefield

# from simulate import CalcWrapper

# adapted from https://github.com/torchmd/torchmd-cg/blob/master/torchmd_cg/utils/make_deltaforces.py


def _split_frame_indices(frames: list[int], n_workers: int) -> list[list[int]]:
    n = len(frames)
    if n_workers <= 1 or n <= 1:
        return [frames]
    n_workers = min(n_workers, n)
    q, r = divmod(n, n_workers)
    out: list[list[int]] = []
    idx = 0
    for w in range(n_workers):
        take = q + (1 if w < r else 0)
        out.append(frames[idx : idx + take])
        idx += take
    return out


def _classical_prior_chunk_worker(
    psf: str,
    coords_npz: str,
    box_npz: str | None,
    forcefield: str,
    exclusions: tuple,
    forceterms: list,
    frame_indices: list[int],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    torch.set_num_threads(1)
    try:
        import mkl  # type: ignore[import-untyped]

        mkl.set_num_threads(1)
    except Exception:
        pass

    precision = torch.float32
    device = torch.device("cpu")
    mol = Molecule(psf)
    natoms = mol.numAtoms
    coords = np.load(coords_npz)
    box = np.load(box_npz) if box_npz else None
    coords_t = torch.tensor(coords, dtype=precision)
    if box is not None:
        linearized = box.reshape(-1, 9).take([0, 4, 8], axis=1)
        box_full = linearized.reshape(linearized.shape[0], 3, 1)
    else:
        box_full = torch.zeros(coords.shape[0], 3, 1)

    ff = tagged_forcefield.create(mol, forcefield)
    parameters = Parameters(ff, mol, forceterms, precision=precision, device=device)  # pyright: ignore[reportArgumentType]
    system = System(natoms, 1, precision, device)
    system.set_positions(np.zeros((natoms, 3, 1)))
    system.set_velocities(torch.zeros(1, natoms, 3))
    forces = Forces(parameters, terms=forceterms, exclusions=exclusions)

    n_fr = len(frame_indices)
    out_f = np.zeros((n_fr, natoms, 3), dtype=np.float32)
    out_e = np.zeros((n_fr,), dtype=np.float32)
    for k, i in enumerate(frame_indices):
        co = coords_t[i]
        system.set_box(box_full[i])
        pot = forces.compute(co.reshape([1, natoms, 3]), system.box, system.forces)
        out_f[k] = system.forces.detach().cpu().reshape(natoms, 3).numpy()
        p0 = pot[0]
        out_e[k] = p0.item() if hasattr(p0, "item") else float(p0)
    return frame_indices, out_f, out_e


class DeltaForces:
    def __init__(self, device, psf, coords_npz, box_npz):
        self.device = torch.device(device)
        self.precision = torch.float32
        self.replicas = 1

        self._psf_path = psf
        self._coords_npz_path = coords_npz
        self._box_npz_path = box_npz

        self.mol = Molecule(psf)
        self.natoms = self.mol.numAtoms

        self.coords = np.load(coords_npz)
        self.box = None
        if box_npz:
            self.box = np.load(box_npz)

        self.coords = torch.tensor(self.coords, dtype=self.precision).to(device)

        if self.box is not None:
            # Reshape box to be rectangle, then format to be given to set_box
            linearized = self.box.reshape(-1,9).take([0,4,8],axis=1)
            self.box_full = linearized.reshape(linearized.shape[0], 3, 1)
        else:
            self.box_full = torch.zeros(self.coords.shape[0], 3, 1)

        self.prior_forces = torch.zeros((self.coords.shape[0], self.natoms, 3), dtype=self.precision).to('cpu') # store these on CPU
        self.prior_energies = torch.zeros(self.coords.shape[0], dtype=self.precision).to('cpu')
        self.parameters = None
        

    def computePriorForces(
        self,
        forcefield,
        exclusions=("bonds"),
        forceterms=["Bonds", "Angles", "RepulsionCG"],
        bar_position=0,
        frames=None,
        num_parallel_workers: int = 1,
    ):
        # if forceterms is empty list, then we exit
        if forceterms == []:
            return

        if frames is None:
            frames = range(0, self.coords.shape[0])
        frames_list = list(frames)
        in_daemon = mp.current_process().daemon
        effective_workers = num_parallel_workers
        if in_daemon and num_parallel_workers > 1:
            tqdm.write(
                "Delta forces - Classical: running single-process because daemon workers cannot spawn child pools."
            )
            effective_workers = 1

        start_time = time.time()
        if effective_workers <= 1 or len(frames_list) <= 1:
            ff = tagged_forcefield.create(self.mol, forcefield)
            parameters = Parameters(
                ff, self.mol, forceterms, precision=self.precision, device=self.device
            )  # pyright: ignore[reportArgumentType]

            system = System(self.natoms, self.replicas, self.precision, self.device)
            system.set_positions(np.zeros((self.natoms, 3, self.replicas)))
            system.set_velocities(torch.zeros(self.replicas, self.natoms, 3))

            forces = Forces(parameters, terms=forceterms, exclusions=exclusions)
            for i in tqdm(
                frames_list,
                position=bar_position,
                dynamic_ncols=True,
                desc="Delta forces - Classical",
                leave=(bar_position == 0),
            ):
                co = self.coords[i]
                system.set_box(self.box_full[i])
                pot = forces.compute(co.reshape([1, self.natoms, 3]), system.box, system.forces)
                fr = system.forces.detach().cpu().reshape([self.natoms, 3])
                self.prior_forces[i, :, :] += fr
                assert len(pot) == 1
                self.prior_energies[i] += pot[0]
        else:
            chunks = _split_frame_indices(frames_list, effective_workers)
            box_arg = self._box_npz_path
            tasks = [
                (
                    self._psf_path,
                    self._coords_npz_path,
                    box_arg,
                    forcefield,
                    tuple(exclusions),
                    list(forceterms),
                    chunk,
                )
                for chunk in chunks
            ]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(chunks)) as pool:
                results = pool.starmap(_classical_prior_chunk_worker, tasks)
            for idxs, f_blk, e_blk in results:
                idx_t = torch.tensor(idxs, dtype=torch.long)
                self.prior_forces[idx_t, :, :] += torch.tensor(f_blk, dtype=self.precision)
                self.prior_energies[idx_t] += torch.tensor(e_blk, dtype=self.precision)
            tqdm.write(
                f"Time taken for classical forces (parallel, {len(chunks)} workers) {time.time() - start_time:.2f}"
            )
            return

        tqdm.write(f"Time taken for classical forces {time.time() - start_time:.2f}")
        

    def makeAndSaveDeltaForces(self, forces_npz, delta_forces_npz, prior_energy_npz):
        all_forces = np.load(forces_npz)
        prior_forces_npy = np.array(self.prior_forces.detach().cpu())
        delta_forces = all_forces - prior_forces_npy
        np.save(delta_forces_npz, delta_forces)
        np.save(prior_energy_npz, self.prior_energies.detach().cpu())

    def addExternalForces(self, forcefield, nnetsBonds, nnetsAngles, nnetsDihedrals, forceterms, bar_position=0, frames=None):
        # if forceterms is empty list, then we exit
        if forceterms == []:
            return

        parameters = ParametersNN(self.mol, forceterms, precision=self.precision, device=self.device) #pyright: ignore[reportArgumentType]

        # for adding the neural network priors. ExternalNN is molecule-agnostic
        calc = ExternalNN(parameters, nnetsBonds, nnetsAngles, nnetsDihedrals, forceterms, self.device)
        tensorbox = torch.tensor(self.box, dtype=self.precision).to(self.device)

        if frames is None: # if None, then process all frames
            frames = range(0, self.coords.shape[0])

        start_time = time.time()
        pot, forces, _ = calc.calculate(self.coords, tensorbox)
        self.prior_forces += forces.detach().cpu()
        self.prior_energies += sum(pot)
        tqdm.write(f"Time taken for neural network forces {time.time() - start_time:.2f}")
