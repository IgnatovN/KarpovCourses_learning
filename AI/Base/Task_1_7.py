"""Напишите функцию function03. Она должна принимать тензор-матрицу с объектами и тензор-вектор с
правильными ответами, будем решать задачу регрессии: def function03(x: torch.Tensor, y: torch.Tensor)

Создайте внутри функции веса для линейной регрессии (без свободного коэффициента), можете
воспользоваться функцией из предыдущего степа. С помощью градиентного спуска подберите оптимальные
веса для входных данных (используйте длину шага около 1e-2). Верните тензор-вектор с оптимальными
весами из функции. Ваши обученные веса должны давать MSE на обучающей выборке меньше единицы.
"""

import torch
import numpy as np


def function03(x: torch.Tensor, y: torch.Tensor):
    w = torch.tensor(np.random.uniform(0.0, 1.0, x.shape[1]), requires_grad=True, dtype=torch.float32)
    step_size = 1e-2

    i = 1
    mse = torch.tensor([2])
    while mse.item() >= 1:
        y_pred = torch.matmul(x, w)

        mse = torch.mean((y_pred - y) ** 2)

        # if i < 20 or i % 10 == 0 or i == n_steps - 1:
        #     print(f'MSE на шаге {i + 1} {mse.item():.5f}')

        mse.backward()

        with torch.no_grad():
            w -= w.grad * step_size

        i += 1

    print(f'MSE на шаге {i + 1} {mse.item():.5f}')

    return w