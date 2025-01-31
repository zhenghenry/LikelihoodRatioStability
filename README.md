# Optimizers for Stabilizing Likelihood-free Inference

This is the codebase for [...](https://arxiv.org/pdf/2501.18419)

### Setup
Run `pip install -r requirements.txt' to install all necessary packages

### Downloading ALEPH dataset
Run the following to download the ALEPH dataset
```
wget https://www.dropbox.com/s/p0lp85fw0xff5wq/inputs.tar.gz
tar -xf inputs.tar.gz
```

### Reproduce ALEPH results
To train an ensemble of 10 classifiers using $\mathsf{ECD}_{q=1}$, run the following code:
```
python ECD_aleph.py
```
Optional flags are defined in `parser.py`.

### Reproduce NF results
We provide two notebooks for each optimizer corresponding to the type of loss functional `Adam_MLC_NF.ipynb`, `Adam_BCE_NF.ipynb`, `ECD_MLC_NF.ipynb`, and `ECD_BCE_NF.ipynb`.

### Analyzing results
We provide two notebooks `analyze_metrics_NF.ipynb` and `analyze_metrics_ALEPH.ipynb` to produce the tables and plots of the paper.
