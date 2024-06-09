#!/usr/bin/env python

import os
import shutil
import numpy as np
from pathlib import Path
from simsopt import make_optimizable
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)

"""
Optimize a VMEC equilibrium for quasi-helical symmetry (M=1, N=1)
throughout the volume.
"""

# This problem has 24 degrees of freedom, so we can use 24 + 1 = 25
# concurrent function evaluations for 1-sided finite difference
# gradients.
proc0_print("Running 2_Intermediate/QH_fixed_resolution.py")
proc0_print("=============================================")

mpi = MpiPartition()

filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.QA')

nfp = 7
qs_surfaces    = np.linspace(0,1,10)
max_mode_array = [1,2,3,4]*2
rel_step_array = [1e-4,1e-5]*4
abs_step_array = [1e-6,1e-7]*4
max_nfev = 25
maxmodes_mpol_mapping = {1: 5, 2: 5, 3: 5, 4: 5, 5: 6}
ftol = 1e-5

aspect_ratio_target = 5
iota_target = 0.21
# min_iota = 0.2
# min_average_iota = 0.3

this_path = os.path.join(parent_path, f'QA_nfp{nfp}')#+('_circularaxis' if circular_axis else ''))
os.makedirs(this_path, exist_ok=True)
shutil.copy(os.path.join(parent_path, 'QA_fixed_resolution.py'), os.path.join(this_path, 'QA_fixed_resolution.py'))
os.chdir(this_path)

vmec = Vmec(filename, mpi=mpi, verbose=False)
vmec.indata.nfp = nfp

# Define parameter space:
for i, max_mode in enumerate(max_mode_array):
    proc0_print(f"###### Optimizing for max_mode={max_mode}, rel_step={rel_step_array[i]} and nfp={nfp}.")
    surf = vmec.boundary
    vmec.indata.mpol = np.max((maxmodes_mpol_mapping[max_mode], vmec.indata.mpol))
    vmec.indata.ntor = np.max((maxmodes_mpol_mapping[max_mode], vmec.indata.ntor))
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    vmec.run()
    
    # def aspect_ratio_max_objective(vmec):
    #     vmec.run()
    #     return np.max((vmec.aspect()-aspect_ratio_target,0))
    # def iota_min_objective(vmec):
    #     vmec.run()
    #     return np.min((np.min(np.abs(vmec.wout.iotaf))-min_iota,0))
    # def iota_mean_min_objective(vmec):
    #     vmec.run()
    #     return np.min((np.abs(vmec.mean_iota())-min_average_iota,0))
    # aspect_ratio_max_optimizable = make_optimizable(aspect_ratio_max_objective, vmec)
    # iota_min_optimizable         = make_optimizable(iota_min_objective, vmec)
    # iota_mean_min_optimizable    = make_optimizable(iota_mean_min_objective, vmec)

    qs = QuasisymmetryRatioResidual(vmec, qs_surfaces, helicity_m=1, helicity_n=0)

    prob = LeastSquaresProblem.from_tuples([(qs.residuals, 0, 1),(vmec.aspect,aspect_ratio_target,1),(vmec.mean_iota,iota_target,1e2)])
                                            # (aspect_ratio_max_optimizable.J, 0, 1),
                                            # (iota_mean_min_optimizable.J, 0, 1e1)])
                                            # (iota_min_optimizable.J, 0, 1e1),
    prob.objective()
    proc0_print("  Quasisymmetry objective before optimization:", qs.total())
    proc0_print("  Total objective before optimization:", prob.objective())
    proc0_print("  Aspect ratio:", vmec.aspect())
    proc0_print("  Mean iota:", np.abs(vmec.mean_iota()))
    # proc0_print("  Min iota:", np.min(np.abs(vmec.wout.iotaf)))
    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=rel_step_array[i], abs_step=abs_step_array[i], max_nfev=max_nfev, ftol=ftol, xtol=ftol)
    prob.objective()
proc0_print("##### Optimization complete.")
proc0_print("  Quasisymmetry objective after optimization:", qs.total())
proc0_print("  Total objective after optimization:", prob.objective())
proc0_print("  Final aspect ratio:", vmec.aspect())
proc0_print("  Final iota:", vmec.wout.iotaf)
proc0_print("  Final mean iota:", np.abs(vmec.mean_iota()))
proc0_print("  Final min iota:", np.min(np.abs(vmec.wout.iotaf)))

if mpi.proc0_world:
    vmec.write_input(os.path.join(this_path, 'input.final'))
    vmec_final = Vmec(os.path.join(this_path, f'input.final'), mpi=mpi, verbose=True)
    vmec_final.indata.ns_array[:3]    = [  16,    51,    101]
    vmec_final.indata.niter_array[:3] = [ 300,   500,  20000]
    vmec_final.indata.ftol_array[:3]  = [ 1e-9, 1e-10, 1e-14]
    vmec_final.write_input(os.path.join(this_path, 'input.final'))
    vmec_final.run()
    shutil.move(os.path.join(this_path, f"wout_final_000_000000.nc"), os.path.join(this_path, f"wout_final.nc"))
    os.remove(os.path.join(this_path, f'input.final_000_000000'))
