# Multi-GPU-CNN-Training-on-CIFAR-10-with-PyTorch-Fabric
This project implements a multi-GPU image classification pipeline for the CIFAR-10 dataset, optimized for the NCSA Delta supercomputing cluster. It uses PyTorch Lightning Fabric for distributed training and TensorBoard for live metric tracking.

# Description 
This repository demonstrates how to train a CNN on CIFAR-10 using multiple GPUs on Delta’s A100 nodes. The training pipeline is powered by:
 - PyTorch Lightning Fabric – to manage device placement, precision, and distributed training
 - TensorBoard – to visualize training metrics (loss, accuracy, etc.)
 - A SLURM-compatible shell script to launch jobs on Delta
It is designed for flexibility, scalability, and reproducibility in a high-performance computing (HPC) environment.

# System Requirements
 - NCSA Delta cluster access
 - SLURM scheduler
 - A100 or equivalent GPU
 - Conda or module-based Python environment with PyTorch Lightning and Fabric

# Features
 - Loads and preprocesses CIFAR-10 using torchvision
 - Fabric-powered training for distributed/accelerated environments
 - TensorBoard logging (to local or cluster-scratch)
 - Configurable via CLI: epochs, batch size, model params
 - Shell script with SLURM-compatible launch commands

## License
This project is licensed under the MIT License - see the LICENSE file for details.
```
Feel free to adjust any part of this README to better fit your specific needs or preferences.


