# relationship of plasma and freqeuncy
import numpy as np
import torch

dev_u = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.enable_grad()
def omega_pe_r(ne_r,r,dev_u=dev_u):
    """Calculates the plasma frequency based on electron density.

    Args:
        ne_r (function): A function that returns the electron density at a given radius.
        r (torch.Tensor): The radius in solar radii.
        dev_u (torch.device, optional): The device to use for calculations. 
            Defaults to dev_u.

    Returns:
        torch.Tensor: The plasma frequency.
    """
    # plasma frequency density relationship
    PI = torch.acos(-torch.ones(1,device=dev_u))
    return 8.93e3* (ne_r(r))**(0.5) * 2 * PI


@torch.enable_grad()
def omega_pe_r_vec(ne_r_vec, r_vec,dev_u=dev_u):
    """Calculates the plasma frequency for a vector of radii.

    Args:
        ne_r_vec (function): A function that returns the electron density at a given radius vector.
        r_vec (torch.Tensor): The radius vector in solar radii.
        dev_u (torch.device, optional): The device to use for calculations. 
            Defaults to dev_u.

    Returns:
        torch.Tensor: The plasma frequency.
    """
    # plasma frequency density relationship
    PI = torch.acos(-torch.ones(1,device=dev_u))
    return 8.93e3* (ne_r_vec(r_vec))**(0.5) * 2 * PI

def omega_pe_r_np(ne_r,r):
    """Calculates the plasma frequency using numpy.

    Args:
        ne_r (function): A function that returns the electron density at a given radius.
        r (np.ndarray): The radius in solar radii.

    Returns:
        np.ndarray: The plasma frequency.
    """
    # plasma frequency density relationship
    return 8.93e3* (ne_r(r))**(0.5) * 2 * np.pi

@torch.enable_grad()
def domega_dxyz_1d(ne_r,r_vec,dev_u=dev_u):
    """Calculates the gradient of the plasma frequency.

    Args:
        ne_r (function): A function that returns the electron density at a given radius.
        r_vec (torch.Tensor): The radius vector in solar radii.
        dev_u (torch.device, optional): The device to use for calculations. 
            Defaults to dev_u.

    Returns:
        torch.Tensor: The gradient of the plasma frequency.
    """
    # differential of omegape
    r_vec.requires_grad_(True)
    rr = torch.sqrt(torch.sum(r_vec.pow(2),axis=0))
    omega_pe_xyz = omega_pe_r(ne_r,rr,dev_u=dev_u)#.repeat(3,1) # to be size:3*N 
    omega_pe_xyz.backward(torch.ones(omega_pe_xyz.shape,device=dev_u)) # for the gradient of omega
    diff_vec = r_vec.grad.data
    return diff_vec.detach()


@torch.enable_grad()
def domega_dxyz_vec(ne_r_vec,r_vec,dev_u=dev_u):
    """Calculates the gradient of the plasma frequency for a vector of radii.

    Args:
        ne_r_vec (function): A function that returns the electron density at a given radius vector.
        r_vec (torch.Tensor): The radius vector in solar radii.
        dev_u (torch.device, optional): The device to use for calculations. 
            Defaults to dev_u.

    Returns:
        torch.Tensor: The gradient of the plasma frequency.
    """
    # differential of omegape
    r_vec.requires_grad_(True)
    omega_pe_xyz = omega_pe_r_vec(ne_r_vec,r_vec,dev_u=dev_u)#.repeat(3,1) # to be size:3*N 
    omega_pe_xyz.backward(torch.ones(omega_pe_xyz.shape,device=dev_u)) # for the gradient of omega
    diff_vec = r_vec.grad.data
    return diff_vec.detach()

@torch.enable_grad()
def dNe_dxyz(ne_r,r_vec,dev_u=dev_u):
    """
    Calculates the gradient of the electron density with respect to x, y, and z.

    Args:
        ne_r (function): A function that returns the electron density at a given radius.
        r_vec (torch.Tensor): The radius vector in solar radii.
        dev_u (torch.device, optional): The device to use for calculations. 
            Defaults to dev_u.

    Returns:
        torch.Tensor: The gradient of the electron density.
    """
    # differential of omegape
    r_vec.requires_grad_(True)
    rr = torch.sqrt(torch.sum(r_vec.pow(2),axis=0))
    Ne_arr = ne_r(rr)#.repeat(3,1) # to be size:3*N 
    Ne_arr.backward(torch.ones(Ne_arr.shape,device=dev_u)) # for the gradient of omega
    diff_vec = r_vec.grad.data
    return diff_vec
