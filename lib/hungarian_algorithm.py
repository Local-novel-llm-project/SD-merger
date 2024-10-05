import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def hungarian_algorithm(cost_matrix: torch.Tensor) -> torch.Tensor:
    cost_matrix_np = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
    assignments = np.stack([row_ind, col_ind], axis=1)
    return torch.from_numpy(assignments).to(cost_matrix.device)


def lowmem_hungarian_algorithm(cost_matrix: torch.Tensor) -> torch.Tensor:
    device = cost_matrix.device
    n = cost_matrix.size(0)

    # 各行の最小値とそのインデックスを取得
    min_cost, min_col_indices = torch.min(cost_matrix, dim=1)

    # 割り当てを初期化
    assigned_cols = set()
    assignments = []

    # 行ごとに割り当てを決定
    for row in range(n):
        col = min_col_indices[row].item()
        if col not in assigned_cols:
            assignments.append([row, col])
            assigned_cols.add(col)
        else:
            # 既に割り当て済みの場合、次善の列を探す
            row_costs = cost_matrix[row].clone()
            if assigned_cols:
                indices = torch.tensor(
                    list(assigned_cols), dtype=torch.long, device=device
                )
                row_costs[indices] = float("inf")
            col = torch.argmin(row_costs).item()
            assignments.append([row, col])
            assigned_cols.add(col)

    return torch.tensor(assignments, dtype=torch.long, device=device)


if __name__ == "__main__":
    # サンプルのコスト行列
    cost_matrix = torch.tensor(
        [[82, 83, 69, 92], [77, 37, 49, 92], [11, 69, 5, 86], [8, 9, 98, 23]],
        dtype=torch.float32,
    )

    assignment = hungarian_algorithm(cost_matrix)
    print("最適な割り当て:")
    for i, j in assignment:
        print(f"行 {i} が列 {j} に割り当てられました。")
