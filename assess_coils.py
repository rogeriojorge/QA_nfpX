#!/usr/bin/env python3
import os
import re
import time
import numpy as np
from pathlib import Path
from simsopt import load
from simsopt.mhd.vmec import Vmec
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk, 
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data,
                           coils_to_focus, coils_to_makegrid)
from simsopt.geo import SurfaceRZFourier
from simsopt.util import proc0_print, comm_world
this_path = os.path.dirname(os.path.abspath(__file__))

nfp=2
filename_wout = f'wout_final.nc'
filename_input = f'input.final'
results_folder = f'QA_nfp{nfp}'
coils_file = f'biot_savart_opt.json'
# ncoils = int(re.search(r'ncoils(\d+)', results_folder).group(1))
if nfp==1: ncoils = 5
if nfp==2: ncoils = 3
if nfp==3: ncoils = 3
if nfp==4: ncoils = 3
if nfp==5: ncoils = 2
if nfp==6: ncoils = 2
if nfp==7: ncoils = 2

nfieldlines = 12
tmax_fl = 5000 # 20000
degree = 4
extend_distance = 0.2 # 0.2
nfieldlines_to_plot = 12
print_surface = False

interpolate_field = True

nphi = 200
ntheta = 64
out_dir = os.path.join(this_path,results_folder)
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)
OUT_DIR = Path("coils")
OUT_DIR.mkdir(parents=True, exist_ok=True)
vmec_file_input = os.path.join(out_dir,filename_input)
surf_vmec = SurfaceRZFourier.from_vmec_input(vmec_file_input, nphi=nphi, ntheta=ntheta, range="full torus")
R_max_vmec = np.max(surf_vmec.gamma()[0,:,0])

surf = SurfaceRZFourier.from_vmec_input(vmec_file_input, nphi=nphi, ntheta=ntheta, range="full torus")
surf.extend_via_normal(extend_distance)
R_max = np.max(surf.gamma()[0,:,0])

vmec_file_wout = os.path.join(out_dir,filename_wout)
R_axis = np.sum(Vmec(vmec_file_wout).wout.raxis_cc)

proc0_print('Loading coils file')
coils_filename = os.path.join(OUT_DIR,coils_file)
bs = load(coils_filename)

coils = bs.coils
base_curves = [coils[i]._curve for i in range(ncoils)]
base_currents = [coils[i]._current for i in range(ncoils)]
coils_to_makegrid(os.path.join(OUT_DIR,"coils_makegrid_format.txt"),base_curves,base_currents,nfp=surf.nfp, stellsym=True)
# coils_to_focus(os.path.join(OUT_DIR,"coils_focus_format.txt"),curves=[coil._curve for coil in coils],currents=[coil._current for coil in coils],nfp=surf.nfp,stellsym=True)

proc0_print('Computing surface classifier')
# surf.to_vtk(os.path.join(OUT_DIR,'surface_for_Poincare'))
sc_fieldline = SurfaceClassifier(surf, h=0.03*R_axis, p=2)
# sc_fieldline.to_vtk(os.path.join(OUT_DIR,'levelset'), h=0.04*R_axis)

def trace_fieldlines(bfield, label):
    t1 = time.time()
    R0 = np.linspace(0.999*R_axis, 1.00*R_max, nfieldlines)
    proc0_print(f"R0={R0}", flush=True)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/surf.nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm_world,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    t2 = time.time()
    proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm_world is None or comm_world.rank == 0:
        for i, fieldline_tys in enumerate(fieldlines_tys[-nfieldlines_to_plot:]):
            particles_to_vtk([fieldline_tys], os.path.join(OUT_DIR,f'fieldlines_{label}_{i}'))
        # particles_to_vtk(fieldlines_tys[-6:], os.path.join(OUT_DIR,f'fieldlines_{label}'))
        plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(OUT_DIR,f'poincare_fieldline_{label}.png'), dpi=300, s=1.5, surf=surf_vmec)

if interpolate_field:
    n = 35
    rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
    zs = surf.gamma()[:, :, 2]
    rrange = (0.8*np.min(rs), 1.2*np.max(rs), n)
    phirange = (0, 2*np.pi/surf.nfp, n*2)
    zrange = (0, 1.2*np.max(np.abs(zs)), n//2)

    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05*R_axis*2.0).flatten())
        # skip = [False]*len(skip)
        proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip

    proc0_print('Initializing InterpolatedField')
    bsh = InterpolatedField(bs, degree, rrange, phirange, zrange, True, nfp=surf.nfp, stellsym=True, skip=skip)
    proc0_print('Done initializing InterpolatedField.')

    bsh.set_points(surf.gamma().reshape((-1, 3)))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    proc0_print("Mean(|B|) on plasma surface =", np.mean(bs.AbsB()))

    proc0_print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))
    
    if print_surface:
        bs.set_points(surf_vmec.gamma().reshape((-1, 3)))
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN_surf = np.sum(Bbs * surf_vmec.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
        pointData = {"B.n/B": BdotN_surf[:, :, None]}
        surf_vmec.to_vtk("surf_assess_coils", extra_data=pointData)
        proc0_print("Printed surface. Exiting.")
        exit()
        
    bsh.set_points(surf.gamma().reshape((-1, 3)))
    bs.set_points(surf.gamma().reshape((-1, 3)))

    proc0_print('Beginning field line tracing')
    trace_fieldlines(bsh, 'bsh')
else:
    trace_fieldlines(bs, 'bs')
