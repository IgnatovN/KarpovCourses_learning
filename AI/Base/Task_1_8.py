"""Напишите функцию function04. Она должна принимать тензор-матрицу с объектами и тензор с правильными
ответами, будем решать задачу регрессии: def function04(x: torch.Tensor, y: torch.Tensor)

Создайте внутри функции полносвязный слой, обучите этот полносвязный слой на входных данных с
омощью градиентного спуска (используйте длину шага около 1e-2). Верните его из функции.
Ваш полносвязный слой должен давать MSE на обучающей выборке меньше 0.3.
"""

import torch
from torch import nn


def function04(x: torch.Tensor, y: torch.Tensor):
    layer = nn.Linear(in_features=x.shape[1], out_features=1)
    step_size = 0.05

    mse = torch.tensor([9])
    while mse >= 0.3:
        y_pred = layer(x).ravel()

        mse = torch.mean((y_pred - y) ** 2)

        mse.backward()

        with torch.no_grad():
            layer.weight -= layer.weight.grad * step_size
            layer.bias -= layer.bias.grad * step_size

        layer.zero_grad()

    return layer