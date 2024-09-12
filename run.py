import numpy as np
from data import load_beeline
from logger import LightLogger
from SIGRN import runSIGRN
from evaluate import extract_edges, get_metrics_auc
import time
#Model Configurations
# The three key concepts proposed in the SIGRN paper are controlled by the following parameters.
# delayed_steps_on_sparse: Number of delayed steps on introducing the sparse loss.
# dropout_augmentation: The proportion of data that will be randomly masked as dropout in each traing step.
# train_on_non_zero: Whether to train the model on non-zero expression data

DEFAULT_GRNVAE_CONFIGS = {
    # Train/Test split
    'train_split': 1.0,# Use all data for training
    'train_split_seed': None, # Seed for random splitting

    # Neural Net Definition
    'hidden_dim': 128,#  Size of dimension in the MLP layers
    'z_dim': 1,# Size of dimension of Z
    'A_dim': 1,#Number of Adjacency matrix to be modeled at the same time
    'train_on_non_zero': True,# Whether to train on non-zero data only
    'dropout_augmentation_p': 0.1,#Probability of augmented dropout. For example, 0.1 means that 10% of data will be temporarily assign to zero in each forward  pass
    'dropout_augmentation_type': 'all',#Choose among 'all' (default), 'belowmean', 'belowhalfmean'. This option specifies where dropout augmentation would happen. If 'belowmean' is selected, the augmentation would only happen on values below global mean.
    'cuda': True,

    # Loss term, hyperparameters
    'alpha': 100,
    # 'beta': 1,
    'chi': 0.5,
    'h_scale': 0,
    'delayed_steps_on_sparse': 30,

    # Neural Net Training
    'number_of_opt': 2,# Number of optimizations
    'batch_size': 64, # Size of training batches
    'n_epochs': 120, # Number of training epochs
    # 'schedule': [120, 240],
    'eval_on_n_steps': 10,# Evaluation frequency
    'early_stopping': 0,# Early stopping criteria
    'lr_nn': 1e-4, # Learning rate for neural network
    'lr_adj': 2e-5, # Learning rate for adjacency matrix
    'K1': 1,
    'K2': 1
}
# Load data from a BEELINE benchmark
# BEELINE benchmarks could be loaded by the load_beeline function, where you specify where to look for data and which benchmark to load. If it's the first time, this function will download the files automatically.
data, ground_truth = load_beeline(
    data_dir='data',
    benchmark_data='hESC', #hESC,hHep,mDC,mESC,mHSC-E,mHSC-GM,mHSC-L
    benchmark_setting='500_STRING'#500_STRING,1000_STRING,1000_Non-ChIP,500_Non-ChIP
)
print(data)
# Initialize logger to track training progress
logger = LightLogger()
start_time = time.time()

#Model Training
vae, adjs, result_rec = runSIGRN(
    data.X, DEFAULT_GRNVAE_CONFIGS, ground_truth=ground_truth, logger=logger)

# Output results
A = vae.get_adj()
end_time = time.time()
total_time = end_time - start_time  # seconds
#
ppi_auc = get_metrics_auc(A, ground_truth)
print(ppi_auc)

