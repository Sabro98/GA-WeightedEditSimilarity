# GA-WeightedEditSimilarity
This repository is the official implementation of "**Weighted Edit Distance optimized using Genetic Algorithm for SMILES-based Compound Similarity**, [PAAA](https://www.springer.com/journal/10044)(SCIE)".

Authors: [In-Hyuk Choi](https://orcid.org/0000-0002-4986-9757) and [Il-Seok Oh](https://scholar.google.com/citations?user=GIe5gKsAAAAJ&hl=ko&oi=ao)

Accept: 2023.01.26  
Published: xxxx.xx.xx

## Abstract
[Edit distance(Levenshtein distance)](https://en.wikipedia.org/wiki/Levenshtein_distance) has three operations; insert, delete, substitute. We set each operation to have a different weight, which is **Weighted Edit Distance**. With Genetic Algorithm(GA), we present optimal weight set of weighted edit distance for each [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) data. 


## Environment
```sh
conda create -n GA-WeightedEditSimilarity python=3.7 -y
conda activate GA-WeightedEditSimilarity
conda install numpy scipy scikit-learn matplotlib tqdm -y
```

## Run
```sh
python main.py -d [e, ic, gpcr, nr]
```

## Dataset
We use four [dataset](http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/); Enzyme, Ion channel, GPCR, Nuclear receptor.
