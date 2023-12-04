# CSCo-DTA
The CSCo-DTA is a novel cross-scale graph contrastive learning approach for drug-target binding affinity prediction. The proposed model utilizes graph convolutional network encoders to extract molecule-scale and network-scale features of drugs and targets. A contrastive learning framework is employed to maximize the mutual information between the features of two scales and explore their potential relationship.

# Dependency
    python 3.9.12
    numpy 1.21.5
    torch 1.11.0
    torch-geometric 2.0.4
    rdkit 2022.03.2

# Data preparation
1. Unpacking data.zip.
2. The target molecule graphs data is downloaded from https://drive.google.com/open?id=1rqAopf_IaH3jzFkwXObQ4i-6bUUwizCv. Move the downloaded folders to the directory of each dataset. 

    * /data/davis/aln/
    * /data/davis/pconsc4/
    * /data/kiba/aln/
    * /data/kiba/pconsc4

# Running
    python inference.py --cuda 0
