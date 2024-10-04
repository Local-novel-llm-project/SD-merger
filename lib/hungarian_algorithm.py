import torch


def hungarian_algorithm(cost_matrix):
    # コスト行列をGPUに転送
    C = cost_matrix.clone().to("cuda")
    n = C.size(0)

    # ステップ1と2：各行と各列から最小値を引く
    C -= C.min(dim=1, keepdim=True)[0]
    C -= C.min(dim=0, keepdim=True)[0]

    # 初期化
    zero_mask = C == 0
    starred_zeros = torch.zeros_like(C, dtype=torch.bool, device="cuda")
    primed_zeros = torch.zeros_like(C, dtype=torch.bool, device="cuda")
    row_covered = torch.zeros(n, dtype=torch.bool, device="cuda")
    col_covered = torch.zeros(n, dtype=torch.bool, device="cuda")

    # ステップ3：ゼロをマーキング
    for i in range(n):
        for j in range(n):
            if zero_mask[i, j] and not row_covered[i] and not col_covered[j]:
                starred_zeros[i, j] = True
                row_covered[i] = True
                col_covered[j] = True

    # カバーをリセット
    row_covered[:] = False
    col_covered[:] = False

    # 必要な内部関数を定義
    def cover_columns_with_starred_zeros():
        col_covered[:] = starred_zeros.any(dim=0)

    def find_uncovered_zero():
        # 未カバーのゼロの位置を取得
        positions = (C == 0) & (~row_covered.unsqueeze(1)) & (~col_covered)
        positions = positions.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return None, None
        else:
            return positions[0][0].item(), positions[0][1].item()

    def augment_path(path):
        for r, c in path:
            starred_zeros[r, c] = ~starred_zeros[r, c]

    max_iterations = 1000
    iteration = 0

    while True:
        cover_columns_with_starred_zeros()
        if col_covered.sum() == n:
            break  # 最適なカバーが見つかった

        if iteration >= max_iterations:
            print("最大反復回数に達しました。アルゴリズムを中断します。")
            break

        done = False
        while not done:
            row, col = find_uncovered_zero()
            if row is None:
                min_uncovered = C[~row_covered, :][:, ~col_covered].min()
                if min_uncovered == 0 or torch.isinf(min_uncovered):
                    print(
                        "min_uncovered が 0 または無限大です。アルゴリズムを中断します。"
                    )
                    done = True
                    break
                # 調整
                C[~row_covered, :] -= min_uncovered
                C[:, col_covered] += min_uncovered
            else:
                primed_zeros[row, col] = True
                # スター付きゼロがその行にあるか確認
                star_col = starred_zeros[row, :].nonzero(as_tuple=True)[0]
                if star_col.numel() == 0:
                    # ステップ5：パスを構築してマーキングを更新
                    path = [(row, col)]
                    done_inner = False
                    while not done_inner:
                        star_row = starred_zeros[:, path[-1][1]].nonzero(as_tuple=True)[
                            0
                        ]
                        if star_row.numel() > 0:
                            star_row = star_row.item()
                            path.append((star_row, path[-1][1]))
                            prime_col = (
                                primed_zeros[star_row, :]
                                .nonzero(as_tuple=True)[0]
                                .item()
                            )
                            path.append((star_row, prime_col))
                        else:
                            done_inner = True
                    augment_path(path)
                    primed_zeros[:] = False
                    row_covered[:] = False
                    col_covered[:] = False
                    done = True
                else:
                    row_covered[row] = True
                    col_covered[star_col[0]] = False  # カバーを更新
        iteration += 1

    # 結果の割り当てを抽出
    assignment = []
    for i in range(n):
        j = starred_zeros[i, :].nonzero(as_tuple=True)[0]
        if j.numel() > 0:
            assignment.append((i, j.item()))
    return assignment


# メモリ効率の良いアルゴリズムも含める場合
def hungarian_algorithm_low_mem(cost_matrix: torch.Tensor) -> torch.Tensor:
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


def sinkhorn_algorithm(cost_matrix, epsilon=1e-3, max_iter=100):
    device = cost_matrix.device
    dtype = cost_matrix.dtype  # データ型を取得

    # データ型を float32 に統一（必要に応じて float16 に変更可能）
    cost_matrix = cost_matrix.to(dtype=torch.float16)

    epsilon = torch.tensor(epsilon, device=device, dtype=cost_matrix.dtype)

    # コスト行列を負の値に変換（正の類似度に対応）
    K = torch.exp(-cost_matrix / epsilon).to(device=device, dtype=cost_matrix.dtype)
    n = K.size(0)

    # u と v のデータ型を K と一致させる
    u = torch.ones(n, device=device, dtype=cost_matrix.dtype) / n
    v = torch.ones(n, device=device, dtype=cost_matrix.dtype) / n

    one = torch.tensor(1.0, device=device, dtype=cost_matrix.dtype)

    for _ in range(max_iter):
        K_v = torch.mv(K, v)
        u = one / K_v
        K_t_u = torch.mv(K.t(), u)
        v = one / K_t_u

    # P = diag(u) @ K @ diag(v)
    P = (u.unsqueeze(1) * K) * v.unsqueeze(0)

    assignment = torch.argmax(P, dim=1)
    return list(enumerate(assignment.cpu().numpy()))


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
