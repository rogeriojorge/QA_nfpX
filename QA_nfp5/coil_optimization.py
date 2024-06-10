#!/usr/bin/env python
import os
import shutil
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from simsopt.mhd import Vmec
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves, CurveSurfaceDistance,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature, CurveXYZFourier,
                         LpCurveCurvature, ArclengthVariation, LinkingNumber, CurveRZFourier)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)

nfp = 5

order = 12
length_target_initial = 4
length_weight = 1e-1
cc_weight = 1e-1
cs_threshold = 0.1
cs_weight = 1e+3
max_curvature_threshold = 10
max_curvature_weight = 1e-3
msc_threshold = 10
msc_weight = 1e-5
arclength_weight = 5e-8
R1 = 0.85
if nfp==1:
    ncoils = 5
    length_target_final = 6.5
    cc_threshold = 0.15
if nfp==2:
    ncoils = 3
    length_target_final = 5
    cc_threshold = 0.18
if nfp==3:
    ncoils = 3
    length_target_final = 6
    max_curvature_threshold = 6
    cc_threshold = 0.1
if nfp==4:
    ncoils = 3
    length_target_final = 5
    max_curvature_threshold = 6
    cc_threshold = 0.06
    cc_weight = 1e+1
if nfp==5:
    ncoils = 2
    R1 = 0.45
    length_target_initial = 3
    length_target_final = 5
    max_curvature_threshold = 10
    cc_threshold = 0.06
    cc_weight = 1e+1

MAXITER = 500

nphi = 32
ntheta = 32

this_path = os.path.join(parent_path, f'QA_nfp{nfp}')
os.makedirs(this_path, exist_ok=True)
shutil.copy(os.path.join(parent_path, 'coil_optimization.py'), os.path.join(this_path, 'coil_optimization.py'))
os.chdir(this_path)
coils_dir = os.path.join(this_path, "coils")
os.makedirs(coils_dir, exist_ok=True)

s = SurfaceRZFourier.from_vmec_input('input.final', range="half period", nphi=nphi, ntheta=ntheta)
# base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
vmec = Vmec('wout_final.nc', verbose=False)
ma = CurveRZFourier(np.linspace(0,1,(ncoils+1)*2*s.nfp), len(vmec.wout.raxis_cc)-1, s.nfp, False)
ma.rc[:] = vmec.wout.raxis_cc
ma.zs[:] = -vmec.wout.zaxis_cs[1:]
ma.x = ma.get_dofs()
gamma_curves = ma.gamma()
numquadpoints = order * 16
base_curves = []
for i in range(ncoils):
    curve = CurveXYZFourier(quadpoints=numquadpoints, order=order)
    angle = (i+0.5)*(2*np.pi)/((2)*s.nfp*ncoils)
    curve.set("xc(0)", gamma_curves[i+1,0])
    curve.set("xc(1)", np.cos(angle)*R1)
    curve.set("yc(0)", gamma_curves[i+1,1])
    curve.set("yc(1)", np.sin(angle)*R1)
    curve.set("zc(0)", gamma_curves[i+1,2])
    curve.set("zs(1)", R1)
    curve.x = curve.x  # need to do this to transfer data to C++
    base_curves.append(curve)
    
base_currents = [Current(1)*1e5 for i in range(ncoils)]

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, os.path.join(coils_dir,"curves_init"))
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN_surf = np.sum(Bbs * s.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
Bmod = bs.AbsB().reshape((nphi,ntheta,1))
pointData = {"B.n/B": BdotN_surf[:, :, None], "B": Bmod}
s.to_vtk(os.path.join(coils_dir, "surf_init"), extra_data=pointData)

Jf = SquaredFlux(s, bs, definition="local")
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
Jcs = [LpCurveCurvature(c, 2, max_curvature_threshold) for c in base_curves]
Jcsdist = CurveSurfaceDistance(curves, s, cs_threshold)
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]

JF = (Jf + length_weight * QuadraticPenalty(sum(Jls), length_target_initial * ncoils, "max") + cc_weight * Jccdist + max_curvature_weight * sum(Jcs)
     + msc_weight * sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs) + arclength_weight * sum(Jals) + cs_weight * Jcsdist)

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.max(np.abs(np.sum(Bbs * s.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, max B·n/B={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    # outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


dofs = JF.x
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 50, 'maxcor': 300}, tol=1e-15)

dofs = res.x
JF = (Jf + length_weight * QuadraticPenalty(sum(Jls), length_target_final * ncoils, "max") + cc_weight * Jccdist + max_curvature_weight * sum(Jcs)
     + msc_weight * sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs) + LinkingNumber(curves, 2) + arclength_weight * sum(Jals) + cs_weight * Jcsdist)
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)

bs.set_points(s.gamma().reshape((-1, 3)))
curves_to_vtk(curves, os.path.join(coils_dir,"curves_opt"))
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN_surf = np.sum(Bbs * s.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
Bmod = bs.AbsB().reshape((nphi,ntheta,1))
pointData = {"B.n/B": BdotN_surf[:, :, None], "B": Bmod}
s.to_vtk(os.path.join(coils_dir, "surf_opt"), extra_data=pointData)
bs.save(os.path.join(coils_dir, "biot_savart_opt.json"))

# Output bigger surfaces
nphi_big = nphi * 2 * nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi = np.linspace(0, 1, nphi_big)
surf_big = SurfaceRZFourier(dofs=s.dofs,nfp=nfp,mpol=s.mpol,ntor=s.ntor,quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta,)
bs.set_points(surf_big.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
BdotN_surf = np.sum(Bbs * surf_big.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
Bmod = bs.AbsB().reshape((nphi_big,ntheta_big,1))
pointData = {"B.n/B": BdotN_surf[:, :, None], "B": Bmod}
surf_big.to_vtk(os.path.join(coils_dir, "surf_opt_big"), extra_data=pointData)
surf_big.to_vtk(os.path.join(this_path, "surface"), extra_data=pointData)