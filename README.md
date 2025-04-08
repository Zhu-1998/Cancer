# Deciphering early warning for cancer progression from single-cell transcriptomics and lineage-tracing
![image](https://github.com/Zhu-1998/Cancer/blob/main/abstract.jpg)

# Analysis tutorials
## 1. Raw data
The raw sequencing reads data can be downloaded from ENA with an accession number [PRJNA803321](https://www.ebi.ac.uk/ena/browser/view/PRJNA803321). The processed data can be downloaded from [KPTracer-Data](https://zenodo.org/records/5847462).

## 2. Estimating RNA velocity
First, [Cellranger](https://github.com/10XGenomics/cellranger) be used to process the raw data to get the count data, and then one can utilize [Samtools](https://github.com/samtools/samtools), [Velocyto](https://github.com/velocyto-team/velocyto.py) to quantify spliced counts and unspliced counts for each gene. Then, RNA velocity could be estimated by various toolkits following the protocol under the default parameters, for example, [Velocyto](https://github.com/velocyto-team/velocyto.py), [Scvelo](https://github.com/theislab/scvelo), [Dynamo](https://github.com/aristoteleo/dynamo-release), etc. 

## 3. Reconstructing the cell state vector field
One can reconstruct the cell state vector field from RNA velocity with `dyn.vf.VectorField` by using [Dynamo](https://github.com/aristoteleo/dynamo-release). Then, the differential geometry (divergence, curl, acceleration, curvature, jacobian, etc) could be analyzed based on the reconstructed vector field. One can also learn an analytical function of vector field from sparse single-cell samples on the entire space robustly by `vector_field_function` function in [Dynamo](https://github.com/aristoteleo/dynamo-release).

## 4. Quantifying landscape-flux of cancer progression
One could simulate stochastic dynamics by solving the Langevin equation based on the analytical function to get the steady-state probability distribution and quantify the non-equilibrium landscape and flux. 
For example, one can run `landscape_multi.py` in `./landscape-flux` to generate the steady-state probability distribution of the dynamics of cancer progression. The step can output grid data (`Xgrid.csv`, `Ygrid.csv`), probability distribution data (`p_tra.csv`), and stochastic force distribution data (`mean_Fx.csv`, `mean_Fy.csv`). Then, `plot_landscape_path.m` in `./landscape-flux` can be run to plot the landscape and least action path.

## 5. Deciphering early warning indicators
### Transition Nucleation & Transition state and Pioneer gene
One can quantify the action along the optimal path in the Hamiltonian-Jacobian form, and then select the position at the last global maximum of action value along the least action path as the transition state (nucleation locations), this transition state deviates from the expected landscape saddle on the gradient path. Then, the gene expression trajectory along the optimal path will be used to screen the pioneer genes (nucleation seeds) that quickly reach the target state.

### GRN frustration
Conflicting interactions within the GRN increase frustration as the system approaches critical transitions. The GRN frustration score be calculated by referring to [Wang's](https://doi.org/10.1103/PRXLife.2.043009) paper.

### Single-cell transcriptional complexity index
The cell complexity index (CCI) and gene complexity index (GCI) be quantified by referring to [SCTC](https://github.com/hailinphysics/sctc).

### Note: 
One can reproduce the results by running `.ipynb` files in `./notebook`, please note that the name of the notebook file may be inconsistent with the officially published article. Of course, users can also load user-defined data for analysis following the tutorials.

## Cite:
Zhu L, Wang J. Deciphering early warning for cancer progression from single-cell transcriptomics and lineage-tracing. (2025).
