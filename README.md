### FermionDSph

Dwarf spheroidal galaxies are excellent systems to probe the nature of fermionic dark matter due to their high observed dark matter phase-space density. This repository is in companion to the recent paper *New Constraints on the Mass of Fermionic Dark Matter from Dwarf Spheroidal Galaxies (Alvey, Sabti et al.)*, where the methodology is presented. Here, the reader may find:

- A tabulated version of the bounds for all dwarfs presented in Fig. 6 of the companion paper. These can be found in the `Bounds/` subfolder for the following cases:
  1. Pauli Exclusion principle (`mdeg.csv`)
  2. Relativistically decoupled thermal fermions for both choices of coarse-graining (`mFD_Gaussian.csv`, `mFD_Maximal.csv`)
  3. Non-resonantly produced sterile neutrinos for both choices of coarse-graining (`mNRP_Gaussian.csv`, `mNRP_Maximal.csv`)
  4. Resonantly-produced sterile neutrinos - contained in the folder `RPSterileNeutrinos` including phase-space bounds based on Gaussian coarse-graining, BBN bounds and overproduction bounds
 
- In the `Code/` subfolder, full code and data for reproducing all figures found in the paper as well as the intermediate quantities such as the escape velocity, can be found.
