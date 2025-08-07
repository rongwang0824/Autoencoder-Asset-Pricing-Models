"""
Autoencoder Models
This program implements AE models by Gu, Kelly, and Xiu (2021).
"""

# Packages
import numpy as np
import pandas as pd
import os
import gc
import time
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import *
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import itertools
import argparse
import pickle

# --- Configuration & Fixed Settings ---
class Config:
    max_epochs = 100
    batch_size = 1
    early_stopping_patience = 5

# --- PyTorch Panel Dataset ---
class PanelDataset(Dataset):
    def __init__(self, df, charc_list, port_df, port_list):
        self.dates = df['date'].unique()
        self.df = df
        self.charc_list = charc_list
        self.df_grouped = df.groupby('date')
        self.permno_map = {date: group['permno'].values for date, group in self.df_grouped}
        
        # Set date as index for efficient lookup of portfolio returns
        self.port_df = port_df.set_index('date')
        self.port_list = port_list

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        month_data = self.df_grouped.get_group(date)
        Z = month_data[self.charc_list].values.astype(np.float32)
        R = month_data['ret'].values.astype(np.float32)
        permno = month_data['permno'].values
        
        # Look up the external portfolio returns for the given date
        try:
            port_ret = self.port_df.loc[date, self.port_list].values.astype(np.float32)
        except KeyError:
            # Handle cases where a date might be missing in the portfolio file
            port_ret = np.zeros(len(self.port_list), dtype=np.float32)
            
        # Return the index instead of the date object
        return Z, R, port_ret, permno, idx

# --- Conditional Autoencoder Model Architecture with an Intercept ---
class Autoencoder(nn.Module):
    def __init__(self, num_charcs, num_ports, num_factors, hidden_layers):
        super(Autoencoder, self).__init__()
        output_dim = num_factors + 1
        beta_layers = []
        input_dim = num_charcs
        for hidden_units in hidden_layers:
            beta_layers.append(nn.Linear(input_dim, hidden_units))
            beta_layers.append(nn.ReLU())
            input_dim = hidden_units
        beta_layers.append(nn.Linear(input_dim, output_dim))
        self.beta_net = nn.Sequential(*beta_layers)
        
        # Factor network now takes the number of external portfolios as input
        self.factor_net = nn.Linear(num_ports, num_factors, bias=False)

    def forward(self, Z, port_ret):
        params = self.beta_net(Z)
        alpha = params[:, 0]
        betas = params[:, 1:]
        # Factors are now a function of external portfolio returns
        factors = self.factor_net(port_ret)
        risk_premium = torch.einsum('ni,i->n', betas, factors)
        fitted_returns = alpha + risk_premium
        return fitted_returns

    def get_factors(self, port_ret):
        return self.factor_net(port_ret)

# --- Model Training and Parameter Extraction ---
def train_single_model(params, config, train_loader, val_loader, num_charcs, num_ports, device):
    """Trains a single model instance and returns its validation loss."""
    model = Autoencoder(
        num_charcs=num_charcs,
        num_ports=num_ports,
        num_factors=params['num_factors'],
        hidden_layers=params['beta_hidden_layers']
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.max_epochs):
        model.train()
        for Z, R, port_ret, _, _ in train_loader:
            Z, R, port_ret = Z.squeeze(0).to(device), R.squeeze(0).to(device), port_ret.squeeze(0).to(device)
            if Z.shape[0] < num_charcs: continue
            optimizer.zero_grad()
            predicted_returns = model(Z, port_ret)
            mse_loss = criterion(predicted_returns, R)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = mse_loss + params['l1_lambda'] * l1_norm
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for Z, R, port_ret, _, _ in val_loader:
                Z, R, port_ret = Z.squeeze(0).to(device), R.squeeze(0).to(device), port_ret.squeeze(0).to(device)
                if Z.shape[0] < num_charcs: continue
                predicted_returns = model(Z, port_ret)
                loss = criterion(predicted_returns, R)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break
    return best_val_loss

def extract_results(model, dataloader, device, num_factors):
    """Extracts alphas, betas, factors, and returns for a given dataset."""
    model.eval()
    results = {
        'params': [],
        'factors': [],
        'returns': []
    }
    with torch.no_grad():
        for Z, R, port_ret, permnos, idxs in dataloader:
            Z_tensor, R_tensor, port_ret_tensor = Z.squeeze(0).to(device), R.squeeze(0).to(device), port_ret.squeeze(0).to(device)
            if Z_tensor.shape[0] < model.beta_net[0].in_features: continue

            # Get the date using the index from the dataloader
            date = dataloader.dataset.dates[idxs.item()]

            params = model.beta_net(Z_tensor).cpu().numpy()
            factors = model.get_factors(port_ret_tensor).cpu().numpy()
            
            month_params_df = pd.DataFrame({
                'date': date, 'permno': permnos.squeeze(0).numpy(), 'alpha': params[:, 0]
            })
            for i in range(num_factors):
                month_params_df[f'beta_{i+1}'] = params[:, i+1]
            results['params'].append(month_params_df)
            
            results['factors'].append(pd.DataFrame([factors], columns=[f'factor_{i+1}' for i in range(num_factors)], index=[date]))
            results['returns'].append(pd.DataFrame({'date': date, 'permno': permnos.squeeze(0).numpy(), 'ret': R.squeeze(0).cpu().numpy()}))

    params_df = pd.concat(results['params'], ignore_index=True) if results['params'] else pd.DataFrame()
    factors_df = pd.concat(results['factors']) if results['factors'] else pd.DataFrame()
    returns_df = pd.concat(results['returns'], ignore_index=True) if results['returns'] else pd.DataFrame()
    
    return params_df, factors_df, returns_df

def calculate_portfolio_parameters(individual_params_df, port_wts, num_factors):
    """Calculates portfolio-level alphas and betas."""
    port_wts_subset = port_wts[port_wts['date'].isin(individual_params_df['date'].unique())]
    merged_df = pd.merge(individual_params_df, port_wts_subset, on=['date', 'permno'])
    port_cols = [col for col in port_wts.columns if col not in ['date', 'permno']]
    
    results = []
    for port_name in port_cols:
        if port_name not in merged_df.columns: continue
        merged_df['weighted_alpha'] = merged_df['alpha'] * merged_df[port_name]
        for i in range(num_factors):
            merged_df[f'weighted_beta_{i+1}'] = merged_df[f'beta_{i+1}'] * merged_df[port_name]
        
        beta_cols = [f'weighted_beta_{i+1}' for i in range(num_factors)]
        agg_dict = {'weighted_alpha': 'sum', **{col: 'sum' for col in beta_cols}}
        
        port_params = merged_df.groupby('date').agg(agg_dict).reset_index()
        port_params.rename(columns={'weighted_alpha': 'alpha', **{f'weighted_beta_{i+1}': f'beta_{i+1}' for i in range(num_factors)}}, inplace=True)
        port_params['portfolio'] = port_name
        results.append(port_params)
        
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def calculate_comprehensive_r2(full_results_df, port_wts, port_list):
    """Calculates comprehensive R2 for individual stocks and external portfolios."""
    
    # --- Universe 1: Individual Stocks ---
    r2_total_idiv = calculate_r2(full_results_df['ret'], full_results_df['fitted_ret'])
    r2_pred_idiv = calculate_r2(full_results_df['ret'], full_results_df['pred_ret'])

    # --- Universe 2: Externally Defined Portfolios ---
    port_ret_df = pd.merge(full_results_df, port_wts, on=['date', 'permno'])
    all_monthly_port_returns = []
    for p_col in port_list:
        if p_col not in port_ret_df.columns: continue
        # Calculate weighted returns for the current portfolio
        port_ret_df['w_ret'] = port_ret_df['ret'] * port_ret_df[p_col]
        port_ret_df['w_fitted'] = port_ret_df['fitted_ret'] * port_ret_df[p_col]
        port_ret_df['w_pred'] = port_ret_df['pred_ret'] * port_ret_df[p_col]
        # Aggregate to get the monthly return for this portfolio
        monthly_port_ret = port_ret_df.groupby('date').agg({'w_ret': 'sum', 'w_fitted': 'sum', 'w_pred': 'sum'})
        all_monthly_port_returns.append(monthly_port_ret)
    
    # Concatenate all monthly portfolio returns into one DataFrame
    comprehensive_port_df = pd.concat(all_monthly_port_returns) if all_monthly_port_returns else pd.DataFrame()
    r2_total_port = calculate_r2(comprehensive_port_df['w_ret'], comprehensive_port_df['w_fitted']) if not comprehensive_port_df.empty else 0
    r2_pred_port = calculate_r2(comprehensive_port_df['w_ret'], comprehensive_port_df['w_pred']) if not comprehensive_port_df.empty else 0
    
    return pd.DataFrame([{
        'r2_total_idiv': r2_total_idiv, 'r2_pred_idiv': r2_pred_idiv,
        'r2_total_port': r2_total_port, 'r2_pred_port': r2_pred_port
    }])

def calculate_r2(true_ret, fitted_ret):
    """Calculates Total or Predictive R-squared."""
    sse = ((true_ret - fitted_ret)**2).sum()
    sst = (true_ret**2).sum()
    return 1 - sse / sst if sst != 0 else 0

# --- Main Function: INS and OOS Estimation ---
def run_model(data, param_grid, config, oos_start, oos_end, val_window, num_factors, beta_hidden_layers, analysis_type):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"*Using device: {device}")

    df, charc_list, port, port_list, port_wts = data['df'], data['charc_list'], data['port'], data['port_list'], data['port_wts']
    num_charcs = len(charc_list)
    num_ports = len(port_list)

    if os.path.dirname(config.output_path):
        os.makedirs(os.path.dirname(config.output_path), exist_ok=True)

    # --- In-Sample Estimation ---
    if analysis_type in ['INS', 'ALL']:
        ins_start_time = time.time()
        print("\n" + "="*30 + "\nIn-Sample Estimation\n" + "="*30)
        
        ins_val_end_date = df['date'].max()
        ins_train_end_date = ins_val_end_date - DateOffset(years=val_window)
        ins_train_df = df[df['date'] <= ins_train_end_date]
        ins_val_df = df[df['date'] > ins_train_end_date]

        ins_train_loader = DataLoader(PanelDataset(ins_train_df, charc_list, port, port_list), batch_size=config.batch_size, shuffle=True)
        ins_val_loader = DataLoader(PanelDataset(ins_val_df, charc_list, port, port_list), batch_size=config.batch_size, shuffle=False)
        
        print("- Tuning Hyperparameters")
        tuning_results = []
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for i, grid_params in enumerate(param_combinations):
            current_params = {**grid_params, 'num_factors': num_factors, 'beta_hidden_layers': beta_hidden_layers}
            val_loss = train_single_model(current_params, config, ins_train_loader, ins_val_loader, num_charcs, num_ports, device)
            tuning_results.append({'params': grid_params, 'loss': val_loss})
        best_grid_params = min(tuning_results, key=lambda x: x['loss'])['params']
        print(f" - Best Tuned Params for In-Sample Run: {best_grid_params}")
        
        final_ins_model = Autoencoder(num_charcs, num_ports, num_factors, beta_hidden_layers).to(device)
        optimizer = optim.Adam(final_ins_model.parameters(), lr=best_grid_params['learning_rate'])
        criterion = nn.MSELoss()
        full_loader = DataLoader(PanelDataset(df, charc_list, port, port_list), batch_size=config.batch_size, shuffle=True)
        
        for epoch in tqdm(range(config.max_epochs), desc=" - Final In-Sample Estimation"):
            final_ins_model.train()
            for Z, R, port_ret, _, _ in full_loader:
                Z, R, port_ret = Z.squeeze(0).to(device), R.squeeze(0).to(device), port_ret.squeeze(0).to(device)
                if Z.shape[0] < num_charcs: continue
                optimizer.zero_grad()
                fitted_returns = final_ins_model(Z, port_ret)
                mse_loss = criterion(fitted_returns, R)
                l1_norm = sum(p.abs().sum() for p in final_ins_model.parameters())
                loss = mse_loss + best_grid_params['l1_lambda'] * l1_norm
                loss.backward()
                optimizer.step()

        params_df, factors_df, returns_df = extract_results(final_ins_model, full_loader, device, num_factors)
        
        with pd.HDFStore(config.output_path, 'a') as store:
            port_params_df = calculate_portfolio_parameters(params_df, port_wts, num_factors)
            store.put(f'INS/F{num_factors}/individual_params', params_df, format='table', data_columns=True)
            store.put(f'INS/F{num_factors}/port_params', port_params_df, format='table', data_columns=True)
            store.put(f'INS/F{num_factors}/factors', factors_df, format='table', data_columns=True)

        returns_df = pd.merge(returns_df, params_df, on=['date', 'permno'])
        returns_df = pd.merge(returns_df, factors_df, left_on='date', right_index=True)
        
        beta_cols = [f'beta_{i+1}' for i in range(num_factors)]
        factor_cols = [f'factor_{i+1}' for i in range(num_factors)]
        returns_df['risk_premium'] = (returns_df[beta_cols].values * returns_df[factor_cols].values).sum(axis=1)
        returns_df['fitted_ret'] = returns_df['alpha'] + returns_df['risk_premium']
        
        # Use simple mean of all historical factors for lambda
        lambda_values = factors_df.mean()
        lambda_cols = [f'lambda_{i+1}' for i in range(num_factors)]
        for i, col in enumerate(lambda_cols):
            returns_df[col] = lambda_values[i]

        returns_df['pred_risk_premium'] = (returns_df[beta_cols].values * returns_df[lambda_cols].values).sum(axis=1)
        returns_df['pred_ret'] = returns_df['alpha'] + returns_df['pred_risk_premium']

        print("- INS R2s")
        ins_r2_df = calculate_comprehensive_r2(returns_df, port_wts, port_list)    
        print(ins_r2_df)
        with pd.HDFStore(config.output_path, 'a') as store:
            store.put(f'INS/F{num_factors}/r2', ins_r2_df, format='table', data_columns=True)
        
        ins_elapsed = time.time() - ins_start_time
        print(f"In-sample analysis finished. Time spent: {ins_elapsed / 60:.2f} minutes.")

    # --- Out-of-Sample Estimation ---
    if analysis_type in ['OOS', 'ALL']:
        oos_start_time = time.time()
        print("\n" + "="*30 + "\nOut-of-Sample Estimation\n" + "="*30)

        oos_results_accumulator = []
        for year in range(oos_start, oos_end + 1):
            print("\n" + "-"*15 + f"\nTest Year: {year}\n" + "-"*15)

            test_start_date, test_end_date = pd.to_datetime(f'{year}-01-01'), pd.to_datetime(f'{year}-12-31')
            val_end_date = test_start_date - MonthEnd(1)
            train_end_date = val_end_date - DateOffset(years=val_window)

            train_df = df[df['date'] <= train_end_date]
            val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)]
            test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)]
            
            train_loader = DataLoader(PanelDataset(train_df, charc_list, port, port_list), batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(PanelDataset(val_df, charc_list, port, port_list), batch_size=config.batch_size, shuffle=False)
            test_loader = DataLoader(PanelDataset(test_df, charc_list, port, port_list), batch_size=config.batch_size, shuffle=False)
            
            print("- Tuning Hyperparameters")
            tuning_results = []
            keys, values = zip(*param_grid.items())
            param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            for i, grid_params in enumerate(param_combinations):
                current_params = {**grid_params, 'num_factors': num_factors, 'beta_hidden_layers': beta_hidden_layers}
                val_loss = train_single_model(current_params, config, train_loader, val_loader, num_charcs, num_ports, device)
                tuning_results.append({'params': grid_params, 'loss': val_loss})
            best_grid_params = min(tuning_results, key=lambda x: x['loss'])['params']
            print(f"- Best Tuned Params: {best_grid_params}")

            final_model = Autoencoder(num_charcs, num_ports, num_factors, beta_hidden_layers).to(device)
            optimizer = optim.Adam(final_model.parameters(), lr=best_grid_params['learning_rate'])
            criterion = nn.MSELoss()
            
            training_data_loader = DataLoader(PanelDataset(pd.concat([train_df, val_df]), charc_list, port, port_list), batch_size=config.batch_size, shuffle=True)
            
            for epoch in tqdm(range(config.max_epochs), desc=f"- Final Training"):
                final_model.train()
                for Z, R, port_ret, _, _ in training_data_loader:
                    Z, R, port_ret = Z.squeeze(0).to(device), R.squeeze(0).to(device), port_ret.squeeze(0).to(device)
                    if Z.shape[0] < num_charcs: continue
                    optimizer.zero_grad()
                    fitted_returns = final_model(Z, port_ret)
                    mse_loss = criterion(fitted_returns, R)
                    l1_norm = sum(p.abs().sum() for p in final_model.parameters())
                    loss = mse_loss + best_grid_params['l1_lambda'] * l1_norm
                    loss.backward()
                    optimizer.step()

            print("- Extracting & Saving OOS Results")
            # Extract for test set
            test_params_df, test_factors_df, test_returns_df = extract_results(final_model, test_loader, device, num_factors)
            
            # Extract for training set
            train_params_df, train_factors_df, _ = extract_results(final_model, training_data_loader, device, num_factors)

            if not test_params_df.empty:
                # Filter training parameters to only the last 12 months (validation period)
                last_12m_start_date = val_end_date - DateOffset(years=1) + MonthBegin(1)
                train_params_df_last_12m = train_params_df[train_params_df['date'] >= last_12m_start_date]

                # Calculate and save portfolio parameters for both sets
                train_port_params_df = calculate_portfolio_parameters(train_params_df_last_12m, port_wts, num_factors)
                test_port_params_df = calculate_portfolio_parameters(test_params_df, port_wts, num_factors)
                
                with pd.HDFStore(config.output_path, 'a') as store:
                    store.put(f'OOS/F{num_factors}/{year}/train_factors', train_factors_df, format='table', data_columns=True)
                    store.put(f'OOS/F{num_factors}/{year}/train_individual_params', train_params_df_last_12m, format='table', data_columns=True)
                    store.put(f'OOS/F{num_factors}/{year}/train_port_params', train_port_params_df, format='table', data_columns=True)
                    
                    store.put(f'OOS/F{num_factors}/{year}/test_factors', test_factors_df, format='table', data_columns=True)
                    store.put(f'OOS/F{num_factors}/{year}/test_individual_params', test_params_df, format='table', data_columns=True)
                    store.put(f'OOS/F{num_factors}/{year}/test_port_params', test_port_params_df, format='table', data_columns=True)

                # Accumulate results for final R2 calculation
                test_returns_df = pd.merge(test_returns_df, test_params_df, on=['date', 'permno'])
                test_returns_df = pd.merge(test_returns_df, test_factors_df, left_on='date', right_index=True)
                
                beta_cols = [f'beta_{i+1}' for i in range(num_factors)]
                factor_cols = [f'factor_{i+1}' for i in range(num_factors)]
                test_returns_df['risk_premium'] = (test_returns_df[beta_cols].values * test_returns_df[factor_cols].values).sum(axis=1)
                test_returns_df['fitted_ret'] = test_returns_df['alpha'] + test_returns_df['risk_premium']
                
                # Use simple mean of the training sample factors for lambda
                lambda_values = train_factors_df.mean()
                lambda_cols = [f'lambda_{i+1}' for i in range(num_factors)]
                for i, col in enumerate(lambda_cols):
                    test_returns_df[col] = lambda_values[i]
                
                test_returns_df['pred_risk_premium'] = (test_returns_df[beta_cols].values * test_returns_df[lambda_cols].values).sum(axis=1)
                test_returns_df['pred_ret'] = test_returns_df['alpha'] + test_returns_df['pred_risk_premium']
                
                oos_results_accumulator.append(test_returns_df)

        # --- Final OOS R2 Calculation ---
        if oos_results_accumulator:
            full_oos_returns_df = pd.concat(oos_results_accumulator, ignore_index=True)
            
            oos_r2_df = calculate_comprehensive_r2(full_oos_returns_df, port_wts, port_list)
            print("\n- OOS R2s")
            print(oos_r2_df)
            with pd.HDFStore(config.output_path, 'a') as store:
                store.put(f'OOS/F{num_factors}/r2', oos_r2_df, format='table', data_columns=True)
        
        oos_elapsed = time.time() - oos_start_time
        print(f"Out-of-sample analysis finished. Time spent: {oos_elapsed / 60:.2f} minutes.")


def load_data(data_path):
    """Loads all necessary data files from the specified path."""
    start_time = time.time()
    try:
        ret = pd.read_parquet(f'{data_path}/ret_long.parquet')
        charc = pd.read_parquet(f'{data_path}/charc.parquet')
        charc_list = [col for col in charc.columns if col not in ['date', 'permno']]
        port = pd.read_csv(f'{data_path}/port.csv')
        port_list = [col for col in port.columns if col not in ['date']]
        port_wts = pd.read_parquet(f'{data_path}/port_wts.parquet')
    except Exception as e:
        print(f"Error loading data: {e}.")
        return None

    df = pd.merge(ret, charc, on=['date', 'permno'], how='inner')
    df['date'] = pd.to_datetime(df['date'], format='%Y%m')
    df = df.sort_values(by=['date', 'permno']).reset_index(drop=True)
    port['date'] = pd.to_datetime(port['date'], format='%Y%m')
    port_wts['date'] = pd.to_datetime(port_wts['date'], format='%Y%m')
    port_wts.fillna(0, inplace=True) # Fill NaN with 0

    data = {'df': df, 'charc_list': charc_list, 'port': port, 'port_list': port_list, 'port_wts': port_wts}
    print(f"Data loaded. Time spent: {time.time() - start_time:.2f} seconds.")
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Autoencoder Asset Pricing Models.')
    parser.add_argument('--model', type=str, default='AE2', choices=['AE1', 'AE2', 'AE3'], help='Specify the model architecture (AE1, AE2, or AE3).')
    parser.add_argument('--analysis_type', type=str, default='ALL', choices=['ALL', 'INS', 'OOS'], help='Specify which analysis to run.')
    args = parser.parse_args()

    total_start_time = time.time()

    # --- Caching Logic ---
    data_path = '/work/rw196/data/ALL'
    cache_path = os.path.join(data_path, 'data_cache.pkl')

    if os.path.exists(cache_path):
        print("*Loading data from cache...*")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from cache. Time spent: {time.time() - total_start_time:.2f} seconds.")
    else:
        print("*Cache not found. Loading data from source files...*")
        data = load_data(data_path)
        if data is None:
            exit()
        print("*Saving data to cache for future runs...*")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    # --- End of Caching Logic ---

    if data is None:
        exit()

    # Set up
    model_archs = {
        'AE1': [32],
        'AE2': [32, 16],
        'AE3': [32, 16, 8]
    }
    beta_hidden_layers = model_archs[args.model]
    
    param_grid = {'learning_rate': [1e-3, 1e-4], 'l1_lambda': [1e-4, 1e-5]}
    oos_start = 1990
    oos_end = 2024
    val_window = 12 
    
    config = Config()
    config.output_path = f'model_est_{args.model}.h5'
    
    # Run models
    for num_factors in [1, 3, 5, 6]:
        print(f"\nEstimating {args.model} -> {beta_hidden_layers} hidden layers, {num_factors} factors.")
        run_model(
            data=data,
            param_grid=param_grid,
            config=config,
            oos_start=oos_start,
            oos_end=oos_end,
            val_window=val_window,
            num_factors=num_factors,
            beta_hidden_layers=beta_hidden_layers,
            analysis_type=args.analysis_type
        )
    
    elapsed = time.time() - total_start_time
    print(f"\nTotal execution time: {elapsed / 60:.2f} minutes.")
