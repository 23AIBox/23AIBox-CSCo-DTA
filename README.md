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

