## Inputs Parameters overwritten with command line arguments
use_half_period = True # If not optimizing a CNT-like stellarator, set this to True for efficiency
finite_beta = False # finite beta optimization not implemented yet
output_interval = 50 # Number of single stage steps in between coil and surface outputs to file
use_initial_coils_if_available = True
# QA
vmec_input_start_QA = 'input.nfp3_QA_optimized' # VMEC input file that serves as a start for the optimization when there are no previous results
LENGTHBOUND_QA = 12 # Threshold for the coil length (each coil is constrained to LENGTHBOUND/ncoils)
CC_THRESHOLD_QA = 0.1 # Threshold for the coil-to-coil distance penalty in the objective function
CURVATURE_THRESHOLD_QA = 5 # Threshold for the curvature penalty in the objective function
MSC_THRESHOLD_QA = 5 # Threshold for the mean squared curvature penalty in the objective function
ncoils_QA = 4 # Number of coils per half field period
quasisymmetry_helicity_n_QA = 0 # Toroidal quasisymmetry integer N in |B|
include_iota_target_QA = True # Specify if iota should be added to the objective function
iota_target_QA = 0.42 # Target rotational transform iota
aspect_ratio_target_QA = 6  # Target aspect ratio
# QH
vmec_input_start_QH = 'input.nfp4_QH_optimized' # VMEC input file that serves as a start for the optimization when there are no previous results
LENGTHBOUND_QH = 12 # Threshold for the sum of coil lengths (each coil is constrained to LENGTHBOUND/ncoils)
CC_THRESHOLD_QH = 0.07 # Threshold for the coil-to-coil distance penalty in the objective function
CURVATURE_THRESHOLD_QH = 14 # Threshold for the curvature penalty in the objective function
MSC_THRESHOLD_QH = 14 # Threshold for the mean squared curvature penalty in the objective function
ncoils_QH = 5 # Number of coils per half field period
quasisymmetry_helicity_n_QH = -1 # Toroidal quasisymmetry integer N in |B|
include_iota_target_QH = False # Specify if iota should be added to the objective function
aspect_ratio_target_QH = 7  # Target aspect ratio
# QI
vmec_input_start_QI = 'input.nfp1_QI' # VMEC input file that serves as a start for the optimization when there are no previous results
LENGTHBOUND_QI = 14 # Threshold for the sum of coil lengths (each coil is constrained to LENGTHBOUND/ncoils)
CC_THRESHOLD_QI = 0.06 # Threshold for the coil-to-coil distance penalty in the objective function
CURVATURE_THRESHOLD_QI = 16 # Threshold for the curvature penalty in the objective function
MSC_THRESHOLD_QI = 16 # Threshold for the mean squared curvature penalty in the objective function
ncoils_QI = 5 # Number of coils per half field period
quasisymmetry_helicity_n_QI = -1 # Toroidal quasisymmetry integer N in |B|
include_iota_target_QI = False # Specify if iota should be added to the objective function
iota_target_QI = 0.61 # Target rotational transform iota
aspect_ratio_target_QI = 7  # Target aspect ratio
elongation_weight = 5e-2
snorms = [0.2,0.4,0.6,0.8] # Flux surfaces at which the penalty will be calculated
nphi_QI=251 # Number of points along measured along each well
nalpha_QI=51 # Number of wells measured
nBj_QI=151 # Number of bounce points measured
mpol_QI=20 # Poloidal modes in Boozer transformation
ntor_QI=20 # Toroidal modes in Boozer transformation
nphi_out_QI=2000 # size of return array if arr_out_QI = True
arr_out_QI=True # If True, returns (nphi_out*nalpha) values, each of which is the difference
# Generic
stage_1 = False # Perform a stage-1 optimization
stage_2 = False # Perform a stage-2 optimization
single_stage = False # Perform a single stage optimization
order = 16 # Number of Fourier modes describing each Cartesian component of each coil
## General input parameters
# Frequent use
max_modes = [1] # Fourier mode resolution for the plasma surface
MAXITER_stage_1 = 10 # Number of iterations to perform in the stage 1 optimization
MAXITER_stage_2_simple = 5 # Number of iterations to perform in the stage 2 optimization (squared flux and length only)
MAXITER_stage_2 = 5 # Number of iterations to perform in the stage 2 optimization
MAXITER_single_stage = 5 # Number of iterations to perform in the main optimization loop
# Flags
remove_previous_results = False # If exists, move folder with the same name to a backup place
remove_previous_debug_output = False # Chose if previous debug_output_file should be deleted
vmec_verbose = False # Print vmec output if True
create_wout_final = True # Create a VMEC wout file entitled "final"
vmec_plot_result = True # Plot the optimized VMEC result
find_QFM_surface = False # Find QFM surface and resulting VMEC equilibrium
vmec_plot_QFM_result = False # Plot the found QFM equilibrium
create_Poincare_plot = False # Create Poincare plot from coils
booz_xform_plot_result = False
booz_xform_plot_QFM_result = False
debug_coils_outputtxt = True
# Diagnostics for coils and plasma
boozxform_nsurfaces = 14
nzeta_Poincare = 4
degree_Interpolated_field = 4
tmax_fl_Poincare = 800
tol_tracing_Poincare = 1e-12
nradius_Poincare = 3
tol_qfm = 1e-14
maxiter_qfm = 2000
constraint_weight_qfm = 1e-0
ntheta_vmec_qfm = 300
mpol_qfm = 16
ntor_qfm = 16
nphi_qfm = 25
ntheta_qfm = 40
# Stage 1 optimization parameters
JACOBIAN_THRESHOLD = 850 # Value of the objective function if VMEC does not converge
aspect_ratio_weight = 1 # Weight for the aspect ratio objective
quasisymmetry_target_surfaces = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ] # Normalized toroidal flux surfaces to target quasisymmetry
qsqi_weight = 1e1 # Weight for the quasisymmetry residuals objective
quasisymmetry_helicity_m = 1 # Poloidal quasisymmetry integer M in |B|
iota_weight = 1e2 # Weight for the rotational transform objective
finite_difference_rel_step = 1e-3 # Relative step size for the stage 1 finite difference
finite_difference_abs_step = 1e-5 # Absolute step size for the stage 1 finite difference
diff_method = "forward" # "forward" # "centered"
# Stage 2 optimization parameters
coils_objective_weight = 1e+3 # Scale of stage 2 objective with respect to stage 1 objective
R0 = 1.0 # Major radius for the initial circular coils
R1 = 0.35 # Minor radius for the initial circular coils
initial_current = 1e5 # Initial current for all coils
nphi = 50 # Toroidal resolution for the Biot-Savart magnetic field
ntheta = 30 # Poloidal resolution for the Biot-Savart magnetic field
LENGTH_CON_WEIGHT = 0.01 # Weight on the quadratic penalty for the curve length
LENGTH_WEIGHT = 1e-7 # Weight on the curve lengths in the objective function
CC_WEIGHT = 5e-1 # Weight for the coil-to-coil distance penalty in the objective function
CS_THRESHOLD = 0.3 # Threshold for the coil-to-surface distance penalty in the objective function
CS_WEIGHT = 3e-1 # Weight for the coil-to-surface distance penalty in the objective function
CURVATURE_WEIGHT = 1e-4 # Weight for the curvature penalty in the objective function
MSC_WEIGHT = 1e-4 # Weight for the mean squared curvature penalty in the objective function
ARCLENGTH_WEIGHT = 5e-8 # Weight for the arclength variation penalty in the objective function
# Files and Folders
executables_folder = "executables" # Directory where VMEC, BOOZ_XFORM, NEO executables are stored
plotting_folder = "plotting" # Directory where the figures will be stored
figures_folder = "figures" # Directory where the figures will be stored
coils_folder = "coils" # Directory where the coils will be stored
vmec_folder = "vmec" # Directory where the vmec input/output files will be stored
debug_output_file = "output.txt" # Text file written at each optimization step
resulting_field_json = "biot_savart_opt.json" # JSON file with resulting coils and Biot-Savart field
initial_surface = "surf_init"
initial_coils = "curves_init"
resulting_surface = "surf_opt" # VTK file with resulting optimized surface and B·n on that surface
resulting_coils = "curves_opt" # VTK file with resulting coils
resulting_coils_inner_loop_json = "biot_savart_inner_loop.json" # JSON file with coils and Biot-Savart field from the inner stage 2 optimization
curves_before_inner_loop_file = "curves_before_inner_loop" # VTK file with optimized surface and B·n on that surface from the inner stage 2 optimization
curves_after_inner_loop_file = "curves_after_inner_loop" # VTK file with coils from the inner stage 2 optimization
