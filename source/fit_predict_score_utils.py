import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Dict
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='paper', style='whitegrid', font_scale=1.2, palette='muted')

# Constants
PARTS = [1, 3, 4, 5]
TARGET_PART = 'part_score'
TARGET_OVERALL = 'overall_score'


def fit_and_predict(df: pd.DataFrame, features: List, pipeline: Pipeline) -> Tuple[pd.Series, Dict]:
    """
    Fits a separate pipeline for each part using the training subset and predicts part scores for all data.
    """
    pipelines = {}
    predictions = pd.Series(dtype=float, name='predicted_part_score') # Stores predictions for all parts
    for part in PARTS:
        df_part = df[df['part'] == part]

        # Fit on training subset for this part
        X_train = df_part[df_part['subset'] == 'train'][features]
        y_train = df_part[df_part['subset'] == 'train'][TARGET_PART]
        pipeline_part = clone(pipeline)
        pipeline_part.fit(X_train, y_train)
        pipelines[part] = pipeline_part

        # Predict on all subsets for this part
        X_all = df_part[features]
        y_all_pred = pipeline_part.predict(X_all)
        predictions = pd.concat([predictions, pd.Series(y_all_pred, index=df_part.index)])
    
    return predictions, pipelines


def build_score_df(df: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """
    Builds a score_df DataFrame that contains the predicted and actual scores 
    for each part and overall indexed by speaker_id.
    """
    score_df = pd.DataFrame()
    score_df['subset'] = df.groupby('speaker_id')['subset'].first()

    df_with_predictions = df.copy()
    df_with_predictions['predicted_score'] = predictions
    for part in PARTS:
        df_part = df_with_predictions[df_with_predictions.part == part]
        pred_part = df_part.groupby('speaker_id')['predicted_score'].mean()
        actual_part = df_part.groupby('speaker_id')[TARGET_PART].first()
        score_df[f'part_{part}_pred'] = pred_part
        score_df[f'part_{part}_actual'] = actual_part
    
    score_df['overall_pred'] = score_df.filter(like='_pred').mean(axis=1)
    score_df['overall_actual'] = df_with_predictions.groupby('speaker_id')[TARGET_OVERALL].first()
    return score_df


def calibrate_score_df(score_df, calibration_set='dev'):

    # Compute calibration mask
    calibration_set_mask = score_df['subset'] == calibration_set

    # Calibrate part predictions
    score_df_calibrated = score_df.copy()
    for part in ['part_1', 'part_3', 'part_4', 'part_5']:
        # Compute calibration parameters on calibration set
        pred = score_df[f'{part}_pred'][calibration_set_mask].dropna()
        actual = score_df[f'{part}_actual'][calibration_set_mask].dropna()
        model = LinearRegression()
        model.fit(pred.values.reshape(-1, 1), actual.values)
        b0, b1 = model.intercept_, model.coef_[0]

        # Apply calibration to all data
        score_df_calibrated[f'{part}_pred'] = b0 + b1 * score_df[f'{part}_pred']

    # Compute overall predictions from calibrated part predictions
    score_df_calibrated['overall_pred'] = score_df_calibrated.filter(like='_pred').mean(axis=1)

    # Compute calibration parameters for overall predictions on calibration set
    pred = score_df_calibrated['overall_pred'][calibration_set_mask]
    actual = score_df_calibrated['overall_actual'][calibration_set_mask]
    pred_nans = pred.isna()
    actual_nans = actual.isna()
    pred = pred[~pred_nans & ~actual_nans]
    actual = actual[~pred_nans & ~actual_nans]
    model = LinearRegression()
    model.fit(pred.values.reshape(-1, 1), actual.values)
    b0, b1 = model.intercept_, model.coef_[0]

    # Apply calibration to all overall predictions
    score_df_calibrated['overall_pred'] = b0 + b1 * score_df_calibrated['overall_pred']

    return score_df_calibrated


def plot_score_df(score_df, experiment_name, part='overall', subset='dev'):
    plot_df = score_df[score_df['subset'] == subset]
    sns.scatterplot(data=plot_df, x=f'{part}_actual', y=f'{part}_pred')
    
    # Fit regression line and get slope
    X = plot_df[[f'{part}_actual']].values
    y = plot_df[f'{part}_pred'].values
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    
    sns.regplot(
        data=plot_df,
        x=f'{part}_actual',
        y=f'{part}_pred',
        scatter=False,
        color='blue',
        label=f'Regression Line (m={slope:.3f})',
        ci=None
    )
    start, end = 1.75, 5.75
    plt.xlim(start, end)
    plt.ylim(start, end)
    plt.plot([start, end], [start, end], color='red', linestyle='--', label='Ideal Line')
    plt.xlabel('Actual Overall Score')
    plt.ylabel('Predicted Overall Score')
    plt.title(experiment_name)
    
    color_05 = 'green'
    color_10 = 'yellow'
    plt.fill_between([start, end], [start-0.5, end-0.5], [start+0.5, end+0.5], color=color_05, alpha=0.3, label='Within 0.5')
    plt.fill_between([start, end], [start-1, end-1], [start-0.5, end-0.5], color=color_10, alpha=0.3, label='Within 1.0')
    plt.fill_between([start, end], [start+0.5, end+0.5], [start+1, end+1], color=color_10, alpha=0.3)

    plt.legend()
    plt.show()


def compute_metrics(score_df: pd.DataFrame, n_bootstraps=1000, subsets=['train', 'dev', 'eval']) -> Dict:
    """
    Computs RMSE, PCC, SRC, Percentage within 0.5 and 1.0 for each part and overall,
    separately for each subset using bootstrapping to estimate standard deviations.
    """
    
    np.random.seed(42)  # For reproducibility
    metrics = {}
    for subset in subsets:
        metrics[subset] = {}
        subset_df = score_df[score_df['subset'] == subset]
        for col in [f'part_{part}' for part in PARTS] + ['overall']:
            metrics[subset][col] = {}

            # Prepare the DataFrame for the current column
            pred_col, actual_col = f'{col}_pred', f'{col}_actual'
            col_df = subset_df[[pred_col, actual_col]]
            mask = col_df[actual_col].notna() # Filter out rows with NaN in actual scores
            col_df = col_df[mask]

            if n_bootstraps == 1:
                # No bootstrapping, compute metrics once
                preds, actuals = col_df[pred_col], col_df[actual_col]
                rmse = np.sqrt(((preds - actuals) ** 2).mean())
                pcc = np.corrcoef(preds, actuals)[0, 1]
                src = spearmanr(preds, actuals)[0]
                p05 = np.mean(np.abs(preds - actuals) <= 0.5) * 100
                p1 = np.mean(np.abs(preds - actuals) <= 1.0) * 100
                
                # Store the metrics with NaN for std
                metrics[subset][col]['rmse'] = float(rmse), float('nan')
                metrics[subset][col]['pcc'] = float(pcc), float('nan')
                metrics[subset][col]['src'] = float(src), float('nan')
                metrics[subset][col]['p05'] = float(p05), float('nan')
                metrics[subset][col]['p1'] = float(p1), float('nan')
            else:
                # Compute RMSE, PCC, SRC along with their standard deviations based on bootstrapping
                rmse_list = []
                pcc_list = []
                src_list = []
                p05_list = []
                p1_list = []
                for _ in range(n_bootstraps):
                    sample = col_df.sample(frac=1, replace=True)
                    preds, actuals = sample[pred_col], sample[actual_col]
                    rmse = np.sqrt(((preds - actuals) ** 2).mean())
                    pcc = np.corrcoef(preds, actuals)[0, 1]
                    src = spearmanr(preds, actuals)[0]
                    p05 = np.mean(np.abs(preds - actuals) <= 0.5) * 100
                    p1 = np.mean(np.abs(preds - actuals) <= 1.0) * 100
                    rmse_list.append(rmse)
                    pcc_list.append(pcc)
                    src_list.append(src)
                    p05_list.append(p05)
                    p1_list.append(p1)
                
                # Store the metrics
                metrics[subset][col]['rmse'] = float(np.mean(rmse_list)), float(np.std(rmse_list))
                metrics[subset][col]['pcc'] = float(np.mean(pcc_list)), float(np.std(pcc_list))
                metrics[subset][col]['src'] = float(np.mean(src_list)), float(np.std(src_list))
                metrics[subset][col]['p05'] = float(np.mean(p05_list)), float(np.std(p05_list))
                metrics[subset][col]['p1'] = float(np.mean(p1_list)), float(np.std(p1_list))
    
    return metrics


def format_metrics_as_dfs(metrics: Dict) -> Dict:
    """
    Format the computed metrics as pandas DataFrames for better readability.
    Returns a dictionary of DataFrames, one for each subset.
    """
    metric_dfs = {}
    for subset in metrics:
        metrics_subset = metrics[subset]
        metric_df = pd.DataFrame(metrics_subset).T
        metric_df.index.name = 'part'
        metric_df = metric_df.map(lambda x: f"{x[0]:.4f} ± {x[1]:.4f}")
        metric_dfs[subset] = metric_df
    return metric_dfs