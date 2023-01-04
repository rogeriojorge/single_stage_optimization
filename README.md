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

- Quasisymmetry flavour: choose `./main.py QA` or `./main.py QH` to use QA or QH vmec input file. If not present, defaults to QA
- Stage: add --stage1, --stage2 or --single_stage when running `main.py` to select which optimization stage to use
- All other inputs can be seen in file `src/inputs.py`. Their default values can be changed in the input terminal

# Simulation results
The output of the code is stored in a folder called results.

To see live the results of the optimization check the file `output.txt` with, for example, `tail -f output.txt`.