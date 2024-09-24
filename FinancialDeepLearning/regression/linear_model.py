import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from tqdm import tqdm
from FinancialDeepLearning.base.summary import LinearResult

__docformat__ = 'restructuredtext en'
__author__ = "<Tommy Lee>"
__all__ = ['LinearRegression','LinearNet']

class LinearRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim : int) -> None:
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class LinearNet(object) :
    def __init__(
            self,
            X : pd.DataFrame,
            y : pd.Series | pd.DataFrame) -> None :
        super(LinearNet, self).__init__()
        self.X = X
        self.y = y

        self.model = None

    def fit(
        self, criterion : str,
        optimizer : str,
        activation_function : str,
        learning_rate: float = 0.001,
        num_epochs : int = 10 ) -> None :

        if (criterion == 'mse') or (criterion == 'MSE') :
            cr = nn.MSELoss()
        else : assert ValueError(f'not supported {criterion} criterion')

        X_train_tensor = torch.tensor(self.X.values, dtype = torch.float32)
        y_train_tensor = torch.tensor(self.y.values, dtype = torch.float32).unsqueeze(1)

        output_dim = 1 if isinstance(self.y, pd.Series) else self.y.shape[1]
        self.model = LinearRegression(input_dim=self.X.shape[1], output_dim=output_dim)

        if (optimizer == 'sgd') or (optimizer == 'SGD') :
            opt = optim.SGD(self.model.parameters(), lr=learning_rate) # Stochastic Gradient Descent
        elif (optimizer == 'adam') or (optimizer == 'ADAM') :
            opt = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f'Optimizer {optimizer} is not supported')

        for epoch in tqdm(range(num_epochs)) :
            self.model.train()  # set train mode
            opt.zero_grad()  # optimizer initialize

            # predict and calculate loss function
            outputs = self.model(X_train_tensor)
            loss = cr(outputs, y_train_tensor)  # calculate MSE Loss

            # backpropagation
            loss.backward()
            opt.step()

        return self.model

    def predict(self, X_test) -> pd.Series:
        X_test_tensor = torch.tensor(X_test.values, dtype = torch.float32)

        self.model.eval()  # set evaluation mode
        with torch.no_grad():
            predicted = self.model(X_test_tensor).numpy()

        res = pd.Series(predicted.squeeze(), index = X_test.index, name = 'predicted')

        return res

    def fittedvalues(self) -> pd.Series:
        X_train_tensor = torch.tensor(self.X.values, dtype=torch.float32)

        self.model.eval()  # set evaluation mode
        with torch.no_grad():
            predicted = self.model(X_train_tensor).numpy()

        res = pd.Series(predicted.squeeze(), index = self.X.index, name = 'fittedvalues')
        return res

    def summary(self) -> LinearResult.summary :
        y_pred = self.fittedvalues()
        resid = self.y - y_pred
        smry = LinearResult(
            self.model,
            self.X, self.y,
            residuals = resid,
            y_pred = y_pred
        ).summary()
        return smry
