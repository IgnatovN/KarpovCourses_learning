"""Напишите функцию train. Она должна принимать на вход нейронную сеть, даталоадер, оптимизатор и функцию потерь.
Она должна иметь следующую сигнатуру:
def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn)
Внутри функции сделайте следующие шаги:

1. Переведите модель в режим обучения.

2. Проитерируйтесь по даталоадеру.

3. На каждой итерации:

    - Занулите градиенты с помощью оптимизатора

    - Сделайте проход вперед (forward pass)

    - Посчитайте ошибку

    - Сделайте проход назад (backward pass)

    - Напечатайте ошибку на текущем батче с точностью до 5 символов после запятой (только число)

    - Сделайте шаг оптимизации

Функция должна вернуть среднюю ошибку за время прохода по даталоадеру."""

from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Optimizer


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()
    errors = []
    for x, y in data_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        print(round(loss.item(), 5))
        errors.append(loss.item())
        optimizer.step()

    return np.mean(errors)
