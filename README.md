## Autoencoder Asset Pricing Models

This repository implements the conditional autoencoder asset pricing models from [Gu, Kelly, and Xiu (2021)](https://www.sciencedirect.com/science/article/pii/S0304407620301998), "Autoencoder asset pricing models," Journal of Econometrics. The script is designed to perform both an in-sample estimation and a rolling out-of-sample estimation. It leverages PyTorch for GPU acceleration and includes a hyperparameter tuning pipeline to optimize model performance.

### Environment Setup

The analysis is managed within a Conda environment to ensure reproducibility. I created a new environment named "EAPenv" with Python 3.9.

```bash
conda create --name EAPenv python=3.9 -y
conda activate EAPenv 
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision torchaudio pytorch-cuda=12.1 pandas numpy tqdm scikit-learn statsmodels joblib seaborn matplotlib openpyxl pyarrow pytables ipykernel -y
python -m ipykernel install --user --name EAPenv --display-name "EAPenv"
conda env export > EAPenvironment.yml
```

To create the environment from this file on a new machine, use:

```bash
conda env create -f EAPenvironment.yml
conda activate EAPenv
```

### Data

Data were constructed and preprocessed in a separate repository: [Prepare-Data-for-Asset-Pricing-Projects
](https://github.com/rongwang0824/Prepare-Data-for-Asset-Pricing-Projects).



### Execution

The script `_model_AE.py` is designed to be executed from the terminal and supports configuration via command-line arguments. Use the `--model` flag to specify one of the predefined architectures: `AE1`, `AE2`, `AE3`, `AE4`, or `AE5`. The `--num_factors` flag allows you to set the desired number of latent factors.

```bash
python3 model_AE.py --model AE2 --num_factors 5
```
