"""Напишите функцию evaluate. Она должна принимать на вход нейронную сеть, даталоадер и функцию потерь.
Она должна иметь следующую сигнатуру: def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn)

Внутри функции сделайте следующие шаги:

1. Переведите модель в режим инференса (применения)

2. Проитерируйтесь по даталоадеру

3. На каждой итерации:

    - Сделайте проход вперед (forward pass)

    - Посчитайте ошибку

Функция должна вернуть среднюю ошибку за время прохода по даталоадеру."""


import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn):
    errors = []
    with torch.no_grad():
        model.eval()
        for x, y in data_loader:
            output = model(x)
            loss = loss_fn(output, y)
            errors.append(loss.item())

    return np.mean(errors)