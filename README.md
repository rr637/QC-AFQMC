# GPU-Accelerated QC-AFQMC 

This repository provides a **GPU-accelerated implementation of phaseless auxiliary-field quantum Monte Carlo (ph-AFQMC)** using **quantum-generated trial wavefunctions** for quantum chemistry.

It accompanies our paper:  
**"Benchmarking quantum trial Wavefunctions for phaseless auxiliary-field quantum Monte Carlo"** — https://arxiv.org/abs/2605.02056

The goal of this project is to investigate how **quantum-prepared trial states (via VQE)** influence the **accuracy, bias, and efficiency of AFQMC simulations**.

---

##  Workflow Overview
The workflow integrates:
- **[PySCF](https://github.com/pyscf/pyscf)** for molecular integrals and Hamiltonian construction  
- **[CUDA-Q](https://github.com/NVIDIA/cuda-quantum)** for preparing quantum trial wavefunctions via VQE  
- **[ipie](https://github.com/JoonhoLee-Group/ipie)** for performing ph-AFQMC simulations  


```text
Molecular system 
        ↓
Electronic structure data (.chk, FCIDUMP)
        ↓
VQE (CUDA-Q) → Quantum trial wavefunction
        ↓
ph-AFQMC (ipie) → Ground state energy & observables
```

---

## Environment Setup

This repository is designed for HPC environments (e.g., NERSC Perlmutter).

To set up the environment, run:

```bash
bash perlmutter_environment_setup.sh
```

This script configures:
- Python environment (conda / modules)
- CUDA / GPU dependencies

---

## Usage

### 1. Build Molecular System (PySCF)

`System.BuildMoleculeProblem` is a wrapper around PySCF for generating molecular Hamiltonians.

Customize your PySCF molecule in `build_sys_files.py` and run. 

This will generate:
- `.chk` file (PySCF checkpoint)
- `FCIDUMP` file (integrals for AFQMC)

Set a unique identifier:

```python
BuildMoleculeProblem.mol_identifier = "your_system_name"
```

This identifier is used consistently across:
- VQE simulations  
- AFQMC simulations  

---

### 2. Run VQE with CUDA-Q

Use `vqe_cuda.py` to generate quantum trial wavefunctions


#### Supported Ansatz Types
- Unitary Coupled Cluster (UCC) 
- ADAPT-VQE on UCCSD operator pool (ADAPT-UCCSD)  
- Local Unitary Cluster Jastrow (LUCJ)  
- Hamiltonian Variational Ansatz (HVA)  

- Gradient evaluations are parallelized across multiple GPUs  

Example job script in `vqe_cuda.sh`

Optimized VQE trial states are saved in {OUTPUT_DIR}/


These are used as inputs for AFQMC.

---

### 3. Run ph-AFQMC

Use `afqmc_grid.py` to perform AFQMC simulations using saved VQE trial wavefunctions:

- Walkers are distributed across CPU ranks using MPI

Example job scirpt in `afqmc_grid.sh`


---

## License

This project is licensed under the Apache License 2.0.

---

##  Acknowledgements

This repository includes modified third-party code from:

- **[Qiskit Nature](https://github.com/qiskit-community/qiskit-nature)** (Apache 2.0)  
- **[CUDA-Q](https://github.com/NVIDIA/cuda-quantum)** (Apache 2.0)  

---
 