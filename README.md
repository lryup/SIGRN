# SIGRN
SIGRN: Inferring Gene Regulatory Network with Soft Introspective Variational Autoencoders

# Architecture

![MDWGANGP2](/images/SIGRN_arc.png)

# Dependencies
- python =3.8
- torch==2.1.0
- scanpy==1.9.1
- other detailed installation packages can be found in requirements.txt
- CUDA toolkit 11.0 or later.

# Installation

1. Clone the SIGRN repository from GitHub:
```
git clone https://github.com/lryup/SIGRN.git
```
2. Create a new environment:
```
conda create -n your_env_name python=3.8.0
  ```
3. Activate the environment:
```
conda activate your_env_name
 ```
4. Install the required packages:
 ```
pip install -r requirements.txt# It is recommended to install only the missing packages
 ```

# Data Preparation
In our study, we trained our model using data from [BEENLINE](https://bcb.cs.tufts.edu/DAZZLE/BEELINE.zip).
You can download the datasets from the provided link. 
# Runing
The training command we used is as follows:
```
python run.py
```
# Example tutorial

We provide an example tutorial  in **Tutorial.ipynb**
Check out the [this tutorial](https://github.com/lryup/SIGRN/blob/main/Tutorial.ipynb) for a quick overview  of how to use SIGRN for your research!

# Usage

SIGRN accepts input data in CSV, TSV format, or H5AD format as provided by Scanpy (genes in rows and cells in columns for TSV and CSV). The output of the  GRN inference task includes an adjacency matrix and various evaluation metrics, such as AUC, EPR, and AUPRR.

# Baseline methods
- Beeline https://github.com/Murali-group/Beeline/tree/master
- DeepSEM https://github.com/HantaoShu/DeepSEM
- GRN-VAE/DAZZLE https://github.com/TuftsBCB/dazzle/tree/main

# References

- Thanks to the following authors for their papers and codes.

  [1] Pratapa, A., Jalihal, A.P., Law, J.N., Bharadwaj, A., Murali, T.: Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nature methods 17(2), 147–154 (2020)

  [2] Shu, H., Zhou, J., Lian, Q., Li, H., Zhao, D., Zeng, J., Ma, J.: Modeling gene regulatory networks using neural network architectures. Nature Computational Science 1(7), 491–501 (2021)

  [3] Zhu, H., Slonim, D.: Grn-vae: A simplified and stabilized sem model for gene regulatory network inference. bioRxiv pp. 2023–01 (2023)

  
