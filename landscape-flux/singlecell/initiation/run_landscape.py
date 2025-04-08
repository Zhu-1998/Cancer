import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import anndata
#import dynamo as dyn
from joblib import Parallel, delayed
#import seaborn as sns
import sys
import os
import time
#import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from typing import Callable, Dict, List, Optional, Tuple, Union
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict
#dyn.get_all_dependencies_version()


adata = anndata.read("/home/datadisk1/zlg/singlecell/EMT/PRJNA803321_2022cell/processed_normal_AT2like_pearson.h5ad")
VecFld = adata.uns['VecFld_umap']


class NormDict(TypedDict):
    xm: np.ndarray
    ym: np.ndarray
    xscale: float
    yscale: float
    fix_velocity: bool

class VecFldDict(TypedDict):
    X: np.ndarray
    valid_ind: float
    X_ctrl: np.ndarray
    ctrl_idx: float
    Y: np.ndarray
    beta: float
    V: np.ndarray
    C: np.ndarray
    P: np.ndarray
    VFCIndex: np.ndarray
    sigma2: float
    grid: np.ndarray
    grid_V: np.ndarray
    iteration: int
    tecr_traj: np.ndarray
    E_traj: np.ndarray
    norm_dict: NormDict

def is_outside_domain(x: np.ndarray, domain: Tuple[float, float]) -> np.ndarray:
    x = x[None, :] if x.ndim == 1 else x
    return np.any(np.logical_or(x < domain[0], x > domain[1]), axis=1)

def vector_field_function(
    x: np.ndarray,
    vf_dict=VecFld,
    dim: Optional[Union[int, np.ndarray]] = None,
    kernel: str = "full",
    X_ctrl_ind: Optional[List] = None,
    **kernel_kwargs,
) -> np.ndarray:
    """vector field function constructed by sparseVFC.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition

    Args:
        x: Set of cell expression state samples
        vf_dict: VecFldDict with stored parameters necessary for reconstruction
        dim: Index or indices of dimensions of the K gram matrix to return. Defaults to None.
        kernel: one of {"full", "df_kernel", "cf_kernel"}. Defaults to "full".
        X_ctrl_ind: Indices of control points at which kernels will be centered. Defaults to None.

    Raises:
        ValueError: If the kernel value specified is not one of "full", "df_kernel", or "cf_kernel"

    Returns:
        np.ndarray storing the `dim` dimensions of m x m gram matrix K storing the kernel evaluated at each pair of control points
    """
    # x=np.array(x).reshape((1, -1))
    if "div_cur_free_kernels" in vf_dict.keys():
        has_div_cur_free_kernels = True
    else:
        has_div_cur_free_kernels = False

    x = np.array(x)
    if x.ndim == 1:
        x = x[None, :]

    if has_div_cur_free_kernels:
        if kernel == "full":
            kernel_ind = 0
        elif kernel == "df_kernel":
            kernel_ind = 1
        elif kernel == "cf_kernel":
            kernel_ind = 2
        else:
            raise ValueError(f"the kernel can only be one of {'full', 'df_kernel', 'cf_kernel'}!")

        K = con_K_div_cur_free(
            x,
            vf_dict["X_ctrl"],
            vf_dict["sigma"],
            vf_dict["eta"],
            **kernel_kwargs,
        )[kernel_ind]
    else:
        Xc = vf_dict["X_ctrl"]
        K = con_K(x, Xc, vf_dict["beta"], **kernel_kwargs)

    if X_ctrl_ind is not None:
        C = np.zeros_like(vf_dict["C"])
        C[X_ctrl_ind, :] = vf_dict["C"][X_ctrl_ind, :]
    else:
        C = vf_dict["C"]

    K = K.dot(C)

    if dim is not None and not has_div_cur_free_kernels:
        if np.isscalar(dim):
            K = K[:, :dim]
        elif dim is not None:
            K = K[:, dim]

    return K

# def vector_field_function(x, VecFld=VecFld, dim=None, kernel="full", X_ctrl_ind=None):
    # """Learn an analytical function of vector field from sparse single cell samples on the entire space robustly.

		# Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
		# """

    # x = np.array(x).reshape((1, -1))
    # if np.size(x) == 1:
        # x = x[None, :]
    # K = dyn.vf.utils.con_K(x, VecFld["X_ctrl"], VecFld["beta"])
    
    # if X_ctrl_ind is not None:
        # C = np.zeros_like(VecFld["C"])
        # C[X_ctrl_ind, :] = VecFld["C"][X_ctrl_ind, :]
    # else:
        # C = VecFld["C"]

    # K = K.dot(C)
    # return K
#################################################################################
def con_K(
    x: np.ndarray, y: np.ndarray, beta: float = 0.1, method: str = "cdist", return_d: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """con_K constructs the kernel K, where K(i, j) = k(x, y) = exp(-beta * ||x - y||^2).

    Args:
        x: Original training data points.
        y: Control points used to build kernel basis functions.
        beta: Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2),
        return_d: If True the intermediate 3D matrix x - y will be returned for analytical Jacobian.

    Returns:
        Tuple(K: the kernel to represent the vector field function, D:
    """
    if method == "cdist" and not return_d:
        K = cdist(x, y, "sqeuclidean")
        if len(K) == 1:
            K = K.flatten()
    else:
        n = x.shape[0]
        m = y.shape[0]

        # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
        # https://stackoverflow.com/questions/12787475/matlabs-permute-in-python
        D = np.matlib.tile(x[:, :, None], [1, 1, m]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, n]), [2, 1, 0])
        K = np.squeeze(np.sum(D**2, 1))
    K = -beta * K
    K = np.exp(K)

    if return_d:
        return K, D
    else:
        return K
#####################################################################
def con_K_div_cur_free(
    x: np.ndarray, y: np.ndarray, sigma: int = 0.8, eta: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a convex combination of the divergence-free kernel T_df and curl-free kernel T_cf with a bandwidth sigma
    and a combination coefficient gamma.

    Args:
        x: Original training data points.
        y: Control points used to build kernel basis functions
        sigma: Bandwidth parameter.
        eta: Combination coefficient for the divergence-free or the curl-free kernels.

    Returns:
        A tuple of G (the combined kernel function), divergence-free kernel and curl-free kernel.

    See also: :func:`sparseVFC`.
    """
    m, d = x.shape
    n, d = y.shape
    sigma2 = sigma**2
    G_tmp = np.matlib.tile(x[:, :, None], [1, 1, n]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, m]), [2, 1, 0])
    G_tmp = np.squeeze(np.sum(G_tmp**2, 1))
    G_tmp3 = -G_tmp / sigma2
    G_tmp = -G_tmp / (2 * sigma2)
    G_tmp = np.exp(G_tmp) / sigma2
    G_tmp = np.kron(G_tmp, np.ones((d, d)))

    x_tmp = np.matlib.tile(x, [n, 1])
    y_tmp = np.matlib.tile(y, [1, m]).T
    y_tmp = y_tmp.reshape((d, m * n), order="F").T
    xminusy = x_tmp - y_tmp
    G_tmp2 = np.zeros((d * m, d * n))

    tmp4_ = np.zeros((d, d))
    for i in tqdm(range(d), desc="Iterating each dimension in con_K_div_cur_free:"):
        for j in np.arange(i, d):
            tmp1 = xminusy[:, i].reshape((m, n), order="F")
            tmp2 = xminusy[:, j].reshape((m, n), order="F")
            tmp3 = tmp1 * tmp2
            tmp4 = tmp4_.copy()
            tmp4[i, j] = 1
            tmp4[j, i] = 1
            G_tmp2 = G_tmp2 + np.kron(tmp3, tmp4)

    G_tmp2 = G_tmp2 / sigma2
    G_tmp3 = np.kron((G_tmp3 + d - 1), np.eye(d))
    G_tmp4 = np.kron(np.ones((m, n)), np.eye(d)) - G_tmp2
    df_kernel, cf_kernel = (1 - eta) * G_tmp * (G_tmp2 + G_tmp3), eta * G_tmp * G_tmp4
    G = df_kernel + cf_kernel

    return G, df_kernel, cf_kernel
#####################################################################
def LHSample( D,bounds,N):  

    result = np.empty([N, D]) 
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):

        for j in range(N):
            temp[j] = np.random.uniform(low=j*d, high=(j+1)*d, size = 1)[0]           

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]
    if np.any(lower_bounds > upper_bounds):
        print('error')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,(upper_bounds - lower_bounds),out=result),lower_bounds,out=result)
    return result
#########################################################################################

VecFnc = vector_field_function

#'xlim': [0.762267804145813, 3.9148308277130126],
#'ylim': [-4.437934923171997, 0.8370339453220368],
x_lim=[0.5, 4.0]
y_lim=[-4.5, 1.0]
Dim = 2      
bounds =[x_lim, y_lim]
N = 400
LHS_of_paras = LHSample(Dim, bounds, N)

numTimeSteps=50000000
starttime = 10000000
Tra_grid = 100


def path_function(D, dt, i):    
    x_path = []
    y_path = []
    num_tra = np.zeros((Tra_grid, Tra_grid))
    total_Fx = np.zeros((Tra_grid, Tra_grid))
    total_Fy = np.zeros((Tra_grid, Tra_grid))

    init_xy = LHS_of_paras[i, :]
    x0 = init_xy[0]
    y0 = init_xy[1]
 
    # Initialize "path" variables
    x_p = x0
    y_p = y0
    # Evaluate potential (Integrate) over trajectory from init cond to  stable steady state
    for n_steps in np.arange(1, numTimeSteps):
        # update dxdt, dydt
        dxdt, dydt = VecFnc([x_p, y_p])
        
        # update x, y
        dx = dxdt * dt + np.sqrt(2*D) * np.sqrt(dt) * np.random.randn()
        dy = dydt * dt + np.sqrt(2*D) * np.sqrt(dt) * np.random.randn()
        
        # dx = dxdt * dt
        # dy = dydt * dt

        x_p = x_p + dx
        y_p = y_p + dy
        
        if x_p < x_lim[0]:
            x_p = 2 * x_lim[0] - x_p
        if y_p < x_lim[0]:
            y_p = 2 * y_lim[0] - y_p
            
        if x_p > x_lim[1]:
            x_p = 2 * x_lim[1] - x_p
        if y_p > y_lim[1]:
            y_p = 2 * y_lim[1] - y_p
            
        dxdt, dydt = VecFnc([x_p, y_p])  
        x_path.append(x_p)
        y_path.append(y_p)
        
        if n_steps > starttime:
            A = int((x_p - x_lim[0]) * Tra_grid / (x_lim[1] - x_lim[0]))   
            B = int((y_p - y_lim[0]) * Tra_grid / (y_lim[1] - y_lim[0]))
            if A < Tra_grid and B<Tra_grid:
                num_tra[A, B] = num_tra[A, B] + 1;
                total_Fx[A, B] = total_Fx[A, B] + dxdt
                total_Fy[A, B] = total_Fy[A, B] + dydt
        
    np.savetxt('num_tra_' + np.str(i) + '.csv', num_tra, delimiter=",") 
    np.savetxt('total_Fx_' + np.str(i) + '.csv', total_Fx, delimiter=",") 
    np.savetxt('total_Fy_' + np.str(i) + '.csv', total_Fy, delimiter=",") 
    

        
    # print("Saving path_time to txt")
    # np.savetxt('path_time' + np.str(i) + '.txt', list(zip(x_path, y_path)), fmt='%.5f %.5f',
               # header='{:<8} {:<25}'.format('umap1', 'umap2'))
               
    # plt.plot(np.linspace(0, 1, int(numTimeSteps-1)), x_path, color='blue', label='umap1') 
    # plt.plot(np.linspace(0, 1, int(numTimeSteps-1)), y_path, color='green', label='umap2')   
    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('X_umap')
    # print("Saving path_time to png")
    # plt.savefig('path_time' + np.str(i) + '.png')
    # plt.close()




start = time.time()

Parallel(n_jobs=100)(
    delayed(path_function)(0.1, 5e-2, i)
    for i in range(N))


num_tra = np.zeros((Tra_grid, Tra_grid))
total_Fx = np.zeros((Tra_grid, Tra_grid))
total_Fy = np.zeros((Tra_grid, Tra_grid))

for i in range(N):
    num_tra_i = np.loadtxt(open('num_tra_' + np.str(i) + '.csv',"rb"),delimiter=",")
    total_Fx_i = np.loadtxt(open('total_Fx_' + np.str(i) + '.csv',"rb"),delimiter=",")
    total_Fy_i = np.loadtxt(open('total_Fy_' + np.str(i) + '.csv',"rb"),delimiter=",")
    num_tra = num_tra + num_tra_i
    total_Fx = total_Fx + total_Fx_i
    total_Fy = total_Fy + total_Fy_i
    
p_tra = num_tra / (sum(sum(num_tra)))
print([sum(sum(num_tra)), N*(numTimeSteps-starttime)])
pot_U = -np.log(p_tra)
mean_Fx = total_Fx / num_tra
mean_Fy = total_Fy / num_tra

xlin = np.linspace(x_lim[0], x_lim[1], Tra_grid)
ylin = np.linspace(y_lim[0], y_lim[1], Tra_grid)
Xgrid, Ygrid = np.meshgrid(xlin, ylin)

np.savetxt("num_tra.csv", num_tra, delimiter=",") 
np.savetxt("total_Fx.csv", total_Fx, delimiter=",") 
np.savetxt("total_Fy.csv", total_Fy, delimiter=",") 
np.savetxt("p_tra.csv", p_tra, delimiter=",") 
np.savetxt("pot_U.csv", pot_U, delimiter=",") 
np.savetxt("mean_Fx.csv", mean_Fx, delimiter=",") 
np.savetxt("mean_Fy.csv", mean_Fy, delimiter=",") 
np.savetxt("Xgrid.csv", Xgrid, delimiter=",") 
np.savetxt("Ygrid.csv", Ygrid, delimiter=",") 

Time = time.time() - start
print(str(Time) + 's')