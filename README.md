# single_stage_optimization
 Optimization of stellarator devices using a single stage approach
---
 # Requisites
 - SIMSOPT: install branch single_stage from https://github.com/hiddenSymmetries/simsopt
 - VMEC2000: https://github.com/hiddenSymmetries/VMEC2000
 - booz_xform: https://github.com/hiddenSymmetries/booz_xform
 - (for finite beta cases) virtual-casing: https://github.com/hiddenSymmetries/virtual-casing
  
Other python requisites available in requirements.txt

Note: aditional plotting routines available if python library `mayavi` is installed

# Run the code
Launch directly the main.py file using

`./main.py`

or launch it using python (or python3)

`python3 main.py`

# Input parameters
The input parameters are sent via the command line.

- Quasisymmetry flavour: choose `./main.py QA`, QH or QI to choose the vmec input file and objective function.
If the two letters are not present, defaults to QA
- Stage: add --stage1, --stage2 or --single_stage when running `main.py` to select which optimization stage to use
- All other inputs can be seen in file `src/inputs.py`. Their default values can be changed in the input terminal. For examples, to change the number of iterations of the single stage optimization to 20, one just needs to write `--MAXITER_single_stage 10` in the terminal. 

# Simulation results
The output of the code is stored in a folder called results which contains, among other things, the coils and the boundary surface, which can be visualized using the Paraview software. 

To see live the results of the optimization check the file `output.txt` with, for example, `tail -f output.txt`.

# Examples

- `./main.py QI --stage_1` runs only stage 1 for quasi-isodynamic symmetry with the default parameters that are in the file `src/inputs.py` at that time.

- `./main.py QA --single_stage` does the same but for quasi-axisymmetry and for a single-stage optimization. 

- `./main.py QI --single_stage --ncoils 10 --max_modes 4 ----MAXITER_stage_2 1000` runs single-stage with quasi-isodynamic symmetry, 10 coils per half field period, changes to 4 the number of Fourier mode resolution for the plasma surface and to 1000 the number of iterations of stage 2 performed in the beginning. This last parameter may seem mute for single-stage optimization but is in fact crucial due to the initial stage 2 that is performed in order to obtain the starting coils.

# Plots

- After running the code, it is possible to obtain certain plots of the results which are not immediately created. They require the installation of VMEC2000 and booz_xform (# Requisites).

- These are created and obtained by running `plot.py` file. 
