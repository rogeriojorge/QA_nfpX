from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry, QuasisymmetryRatioResidual, vmec_splines
import numpy as np
from simsopt.mhd import Vmec
from matplotlib import pyplot as plt

nfp_array = [2,3,4,5,6,7]
s = 0.99
ntheta = 80
nphi = 100
alpha = [0]
    
max_u_dot_grad_phi_times_R = []
max_u_dot_grad_R = []
max_u_dot_grad_Z = []
for nfp in nfp_array:
    vmec = Vmec(f'/Users/rogeriojorge/local/QA_nfpX/QA_nfp{nfp}/wout_final.nc', ntheta=ntheta, nphi=nphi)
    
    # ## Plot gaussian curvature
    # surface = vmec.boundary
    # plt.imshow(surface.surface_curvatures()[:,:,1])
    # plt.clim(-10,10)
    # plt.colorbar()
    # plt.xlabel('phi')
    # plt.ylabel('theta')
    # plt.show()
    # exit()

    theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    fl = vmec_compute_geometry(vmec, s, theta1d, phi1d, phi_center=0)
    qs = QuasisymmetryRatioResidual(vmec, s, 1, 1, ntheta=ntheta, nphi=nphi)
    r = qs.compute()

    R = fl.R
    Z = fl.Z
    d_R_d_theta = fl.d_R_d_theta_vmec
    d_R_d_s = fl.d_R_d_s
    d_R_d_phi = fl.d_R_d_phi
    d_Z_d_theta = fl.d_Z_d_theta_vmec
    d_Z_d_s = fl.d_Z_d_s
    d_Z_d_phi = fl.d_Z_d_phi
    b_sub_theta = r.bsubu
    b_sub_phi = r.bsubv
    b_sup_phi = r.bsupv
    b_sup_theta = r.bsupu
    modB = r.modB
    iota = r.iota
    G = r.G
    sqrtg = r.sqrtg

    u_dot_grad_phi_times_R = R*(-b_sub_theta/sqrtg + G/iota*b_sup_phi)/modB**2
    u_dot_grad_R = (b_sub_phi*d_R_d_theta - b_sub_theta*d_R_d_phi)/(sqrtg*modB**2) + (b_sup_phi*d_R_d_phi+b_sup_theta*d_R_d_theta)*G/(iota*modB**2)
    u_dot_grad_Z = (b_sub_phi*d_Z_d_theta - b_sub_theta*d_Z_d_phi)/(sqrtg*modB**2) + (b_sup_phi*d_Z_d_phi+b_sup_theta*d_Z_d_theta)*G/(iota*modB**2)
    
    max_u_dot_grad_phi_times_R.append(np.mean(np.abs(u_dot_grad_phi_times_R)))
    max_u_dot_grad_R.append(np.mean(np.abs(u_dot_grad_R)))
    max_u_dot_grad_Z.append(np.mean(np.abs(u_dot_grad_Z)))

    # # ## Plot u dot grad phi times R
    # # plt.figure();plt.imshow(u_dot_grad_phi_times_R[0], origin='lower');plt.xlabel('theta');plt.ylabel('phi');plt.colorbar()
    # # plt.figure();plt.imshow(u_dot_grad_R[0], origin='lower');plt.xlabel('theta');plt.ylabel('phi');plt.colorbar()
    # plt.figure();plt.plot(u_dot_grad_R[0,20]);plt.xlabel('phi');plt.ylabel('u_dot_grad_R')
    # # plt.figure();plt.plot(u_dot_grad_R[0,:,0]);plt.xlabel('theta');plt.ylabel('u_dot_grad_R')
    # plt.show()
    # exit()

# # plot the fit to a linear function
# fit = np.polyfit(nfp_array, max_u_dot_grad_phi_times_R, 1)
# linear_fit = np.poly1d(fit)
# plt.plot(nfp_array, max_u_dot_grad_phi_times_R, label='u dot grad phi times R')
# plt.plot(nfp_array, linear_fit(nfp_array), label='Linear Fit')
# plt.legend()
# plt.show()
# exit()

plt.plot(nfp_array, max_u_dot_grad_phi_times_R, label='u dot grad phi times R')
plt.plot(nfp_array, max_u_dot_grad_R, label='u dot grad R')
plt.plot(nfp_array, max_u_dot_grad_Z, label='u dot grad Z')
plt.legend()
plt.show()
