from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry, QuasisymmetryRatioResidual, vmec_splines
import numpy as np
from simsopt.mhd import Vmec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

nfp_array_curvs = [2,3,4,6]
nfp_array = [2,3,4,5,6,7]
s = 0.99
ntheta = 120
nphi = 200
alpha = [0]
    
max_u_dot_grad_phi_times_R = []
max_u_dot_grad_R = []
max_u_dot_grad_Z = []

fig_curvs = plt.figure(0)
curv_fig, curv_axes = plt.subplots(4, 4, figsize=(6, 4))
curv_array = [1,2,3]
i=0
for nfp in nfp_array:
    vmec = Vmec(f'/Users/rogeriojorge/local/QA_nfpX/QA_nfp{nfp}/wout_final.nc', ntheta=ntheta, nphi=nphi, range_surface='field period')
    surface = vmec.boundary

    theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    fl = vmec_compute_geometry(vmec, s, theta1d, phi1d, phi_center=0)
    qs = QuasisymmetryRatioResidual(vmec, s, 1, 1, ntheta=ntheta, nphi=nphi)
    r = qs.compute()

    cmap='RdBu_r'
    ncontours = 15
    if nfp in nfp_array_curvs:
        for j, curv in enumerate(curv_array):
            if curv==1: 
                title=r'$\mathcal{K}$';plot_min=-50;plot_max=50
            elif curv==2:
                title=r'$\kappa_1$';plot_min=-5;plot_max=5
            elif curv==3:
                title=r'$\kappa_2$';plot_min=-50;plot_max=50
            ax = curv_axes[j,i].imshow(surface.surface_curvatures()[:,:,curv].transpose(), origin='lower', extent=[0,2/nfp,0,2], aspect='auto', cmap=cmap)
            # ax = curv_axes[j,i].contourf(surface.surface_curvatures()[:,:,curv].transpose(), origin='lower', extent=[0,2*np.pi/nfp,0,2*np.pi], cmap=cmap, levels=np.linspace(plot_min, plot_max, ncontours), vmin=plot_min, vmax=plot_max)
            ax.set_clim(plot_min,plot_max)
            if i==len(nfp_array_curvs)-1:
                divider = make_axes_locatable(curv_axes[j,i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                clb = curv_fig.colorbar(ax, cax=cax, orientation='vertical', cmap=cmap)
                clb.ax.set_title(title)
            if i==0: curv_axes[j,i].set_ylabel(r'$\theta/\pi$')
            else: curv_axes[j,i].set_yticklabels([])
            curv_axes[j,i].set_xticklabels([])
            if j==0: curv_axes[j,i].set_title(r'$\mathregular{n_{fp}}='+f'{nfp}$')

        j+=1
        grad_psi = np.sqrt(fl.grad_psi_X**2 + fl.grad_psi_Y**2 + fl.grad_psi_Z**2)
        ax = curv_axes[j,i].imshow(1/grad_psi[0,:,:], origin='lower', extent=[0,2/nfp,0,2], aspect='auto', cmap=cmap)
        ax.set_clim(0,20)
        if i==len(nfp_array_curvs)-1:
            divider = make_axes_locatable(curv_axes[j,i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = curv_fig.colorbar(ax, cax=cax, orientation='vertical', cmap=cmap)
            clb.ax.set_title(r'$1/|\nabla \psi|$')
        if i==0: curv_axes[j,i].set_ylabel(r'$\theta/\pi$')
        else: curv_axes[j,i].set_yticklabels([])
        curv_axes[j,i].set_xlabel(r'$\phi/\pi$')

        i+=1

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
    # plt.figure();plt.imshow(u_dot_grad_phi_times_R[0], origin='lower');plt.xlabel('theta');plt.ylabel('phi');plt.colorbar()
    # # plt.figure();plt.imshow(u_dot_grad_R[0], origin='lower');plt.xlabel('theta');plt.ylabel('phi');plt.colorbar()
    # plt.figure();plt.plot(u_dot_grad_R[0,20]);plt.xlabel('phi');plt.ylabel('u_dot_grad_R')
    # # plt.figure();plt.plot(u_dot_grad_R[0,:,0]);plt.xlabel('theta');plt.ylabel('u_dot_grad_R')
    # plt.show()
    # exit()

curv_fig.tight_layout()
curv_fig.subplots_adjust(wspace=0.3, hspace=0.45)
curv_fig.savefig(f'gaussian_curvature_nfp'+''.join(map(str,nfp_array_curvs))+'.pdf', dpi=500)
plt.close(curv_fig)

fig_udot = plt.figure(2, figsize=(5,3))
plt.plot(nfp_array, max_u_dot_grad_phi_times_R, 'ro-', label=r'max($R \boldsymbol{u} \cdot \nabla \phi$)')
plt.plot(nfp_array, np.poly1d(np.polyfit(nfp_array, max_u_dot_grad_phi_times_R, 1))(nfp_array), 'r--', label='Linear Fit')
plt.plot(nfp_array, max_u_dot_grad_R, 'bo-', label=r'max($\boldsymbol{u} \cdot \nabla R$)')
plt.plot(nfp_array, np.poly1d(np.polyfit(nfp_array, max_u_dot_grad_R, 1))(nfp_array), 'b--', label='Linear Fit')
plt.plot(nfp_array, max_u_dot_grad_Z, 'ko-', label=r'max($\boldsymbol{u} \cdot \nabla Z$)')
plt.plot(nfp_array, np.poly1d(np.polyfit(nfp_array, max_u_dot_grad_Z, 1))(nfp_array), 'k--', label='Linear Fit')
plt.xlabel(r'$n_{fp}$', fontsize=12)
plt.ylabel(r'max($\boldsymbol{u} \cdot \nabla \boldsymbol{x}$)', fontsize=12)
plt.legend()
plt.tight_layout()
fig_udot.savefig('u_dot_grad.pdf', dpi=500)
