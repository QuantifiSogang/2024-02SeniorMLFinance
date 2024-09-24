# library import
# dependency : pandas, numpy, statsmodels, torch, scipy
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import stats

# statsmodels
from statsmodels.iolib.summary2 import Summary
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

__docformat__ = 'restructuredtext en'
__author__ = "<Tommy Lee>"
__all__ = ['LinearResult']

class LinearResult(object):
    def __init__(
            self,
            model,
            X: pd.Series | pd.DataFrame,
            y: pd.Series | pd.DataFrame,
            residuals: np.ndarray,
            y_pred: np.ndarray | pd.Series) -> None:
        self.model = model
        self.X = X
        self.y = y
        self.y_pred = y_pred
        self.residuals = residuals

    def summary(self, weights, bias) -> Summary:
        smry = Summary()

        # Loop through all layers' weights and combine them into a final weight
        for i, layer in enumerate(self.model.children()):
            if isinstance(layer, nn.Linear):
                layer_weights = layer.weight.detach().numpy()
                layer_bias = layer.bias.detach().numpy()

                # Multiply layer weights across the layers
                if weights is None:
                    weights = layer_weights
                    bias = layer_bias
                else:
                    weights = np.dot(weights, layer_weights.T)
                    bias = np.dot(bias, layer_weights.T) + layer_bias

        # Add constant (bias) to X_temp for summary calculations
        X_temp = sm.add_constant(self.X, has_constant='add')
        X_temp['const'] = bias[0]

        n = len(self.y)  # Number of observations
        p = X_temp.shape[1]  # Number of parameters (including bias)

        mse = np.sum(self.residuals ** 2) / (n - p)
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * mse) + 1)

        k = X_temp.values.shape[1] + 1
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)
        hqic = -2 * log_likelihood + 2 * k * np.log(np.log(n))

        # evaluation
        mse = mean_squared_error(self.y, self.y_pred)
        r2 = r2_score(self.y, self.y_pred)

        # Model Information
        model_info = [
            ('Dep. Variable:', 'predicted'),
            ('Model:', 'LinearNet'),
            ('Date:', pd.Timestamp.now().strftime('%a, %d %b %Y')),
            ('Time:', pd.Timestamp.now().strftime('%H:%M:%S')),
            ('Sample:', f'{len(self.X)}'),
            ('No. Observations:', f"{len(X_temp)}"),
            ('R-squared:', f"{r2:.4f}"),
            ('MSE:', f"{mse:.4f}"),
            ('Log Likelihood', f"{log_likelihood:.3f}"),
            ('AIC', f"{aic:.3f}"),
            ('BIC', f"{bic:.3f}"),
            ('HQIC', f"{hqic:.3f}"),
        ]

        smry.add_dict(dict(model_info))

        XTX_inv = np.linalg.inv(np.dot(X_temp.T, X_temp))  # (X'X)^-1
        param_variances = mse * np.diag(XTX_inv)
        std_err = np.sqrt(param_variances)  # Standard error

        # Estimated t-values and p-values
        t_values = np.append(bias, weights) / std_err
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=n - p))

        # Confidence intervals
        conf_int = np.column_stack([
            np.append(bias, weights) - 1.96 * std_err,
            np.append(bias, weights) + 1.96 * std_err
        ])

        # Params data
        params_data = pd.DataFrame({
            "coef": np.append(bias, weights),
            "std err": std_err,
            "t": t_values,
            "P>|t|": p_values,
            "[0.025": conf_int[:, 0],
            "0.975]": conf_int[:, 1]
        }, index=['const'] + list(self.X.columns))

        smry.add_df(params_data)

        # Residual tests
        jb_test = jarque_bera(self.residuals)
        dw_test = durbin_watson(self.residuals)
        bp_test = het_breuschpagan(self.residuals, X_temp)

        residual_tests = [
            ('Jarque-Bera Test:', f"{jb_test.statistic:.2f}"),
            ('Prob(JB):', f"{jb_test.pvalue:.2f}"),
            ('Skew:', f"{skew(self.residuals):.2f}"),
            ('Kurtosis:', f"{kurtosis(self.residuals):.2f}"),
            ('Durbin-Watson:', f"{dw_test:.2f}"),
            ('Breusch-Pagan Test:', f"{bp_test[0]:.2f}"),
            ('Prob(H):', f"{bp_test[1]:.2f}")
        ]

        smry.add_dict(dict(residual_tests))
        smry.title = 'LinearNet Results'

        return smry
