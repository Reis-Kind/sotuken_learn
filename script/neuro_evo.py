import numpy as np
import matplotlib.pyplot as plt
import torch


def linear_function(x, w, b):
    """
    一次関数
    Parameters
    ----------
    x: torch.Tensor
        入力データ
    w: torch.Tensor
        一次関数の傾き(重み)
    b: torch.Tensor
        一次関数の切片(バイアス)

    Returns
    -------
    fx: torch.Tensor
        一次関数の出力値
    """
    fx = w * x + b
    return fx


def rmse(x, w, b, y_true):
    """
    平均二乗誤差の平方根

    Parameters
    ----------
    x: torch.Tensor
        入力データ
    w: torch.Tensor
        一次関数の傾き(重み)
    b: torch.Tensor
        一次関数の切片(バイアス)
    y_true: torch.Tensor
        真値

    Returns
    -------
    RMSE: float
        RMSEのスカラー値
    """
    y = linear_function(x, w, b)
    mse = torch.mean((y - y_true) ** 2)
    RMSE = torch.sqrt(mse)
    return RMSE.item()


def evaluate(x, w, b, y_true):
    """
    評価関数

    Parameters
    ----------
    x: torch.Tensor
        入力データ
    w: torch.Tensor
        一次関数の傾き(重み)
    b: torch.Tensor
        一次関数の切片(バイアス)
    y_true: torch.Tensor
        真値

    Returns
    -------
    score: float
        適応度
    """
    rmse_val = rmse(x, w, b, y_true)
    score = 1.0 / (0.5 + rmse_val)
    return score


def genetic_algorithm(x, y):
    """
    島モデルを採用した遺伝的アルゴリズムによる一次関数のパラメータ推定

    Parameters
    ----------
    x: torch.Tensor
        入力データ
    y: torch.Tensor
        真値

    Returns
    -------
    w: torch.Tensor
        最適化された傾き
    b: torch.Tensor
        最適化された切片
    error_history: list
        各世代のRMSEの履歴
    """
    # 島の数
    islands = 4
    # 島ごとの個体数
    models_per_island = 16
    # 世代数
    generations = 121
    # 移住を行う世代の間隔
    migration_interval = 20
    # 突然変異の強さ
    mutation_strength = 0.2
    # 突然変異の確率
    mutation_rate = 0.1

    # 再現性のための乱数シード設定
    torch.manual_seed(42)

    # 初期値としてランダムなwとbを生成
    islands_w = torch.randn(islands, models_per_island) * 5
    islands_b = torch.randn(islands, models_per_island) * 5

    # 各世代のRMSEの履歴を保存するリスト
    error_history = []

    for gen in range(generations):
        island_scores = []
        island_best_w = []
        island_best_b = []
        island_worst = []

        # 各島で
        for i in range(islands):
            # 島の全個体のスコアを計算
            scores = [evaluate(x, islands_w[i][j], islands_b[i][j], y) for j in range(models_per_island)]

            # 最高スコアと、その個体のインデックスを保存
            island_scores.append(max(scores))
            best_j = np.argmax(scores)
            worst_j = np.argmin(scores)

            # 最高個体のパラメータを保存
            island_best_w.append(islands_w[i][best_j].clone())
            island_best_b.append(islands_b[i][best_j].clone())
            island_worst.append(worst_j)

            # 最高個体以外のパラメータを更新
            for j in range(models_per_island):
                if j != best_j:
                    # 最高個体のパラメータを50％混ぜる
                    islands_w[i][j] = (0.5 * islands_w[i][best_j] + 0.5 * islands_w[i][j])
                    islands_b[i][j] = (0.5 * islands_b[i][best_j] + 0.5 * islands_b[i][j])

                    # 確率で突然変異
                    if torch.rand(1) < mutation_rate:
                        islands_w[i][j] += torch.randn(1).item() * mutation_strength
                    if torch.rand(1) < mutation_rate:
                        islands_b[i][j] += torch.randn(1).item() * mutation_strength

        # 一定の世代ごとに移住を実行
        if (gen + 1) % migration_interval == 0:
            for now_islands in range(islands):
                next_i = (now_islands + 1) % islands
                target = island_worst[next_i]
                islands_w[next_i][target] = island_best_w[now_islands].clone()
                islands_b[next_i][target] = island_best_b[now_islands].clone()

        best_score = np.argmax(island_scores)
        w = island_best_w[best_score]
        b = island_best_b[best_score]

        fx_best = linear_function(x, w, b)
        error_val = rmse(x, w, b, y)

        error_history.append(error_val)

        log = "{:3d}世代  Error: {:10.4f},  w: {:6.4f},  b: {:6.4f}"
        print(log.format(gen + 1, error_val, w.item(), b.item()))

        # 世代を重ねるごとに突然変異の強度を小さく
        mutation_strength = mutation_strength * (1.0 - gen / generations)

    return w, b, error_history


def main():
    data = 500

    true_w = 0.8
    true_b = 2.0

    torch.manual_seed(42)

    x = torch.rand(data) * 25
    noise = torch.randn(data) * 0.1
    y = true_w * x + true_b + noise

    w, b, error_history = genetic_algorithm(x, y)
    print(f"\n最終結果 -> w: {w.item():.4f}, b: {b.item():.4f}")

    y_fit = linear_function(x, w, b)

    plt.figure(1)
    plt.scatter(x.numpy(), y.numpy(), alpha=0.5, s=10, label="Data")

    x_sort, idx = torch.sort(x)

    plt.plot(
        x_sort.numpy(),
        y_fit[idx].numpy(),
        "r-",
        label=f"Fit: y={w.item():.2f}x+{b.item():.2f}",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([x.min().item() - 0.5, x.max().item() + 0.5])
    plt.ylim([y.min().item() - 1.0, y.max().item() + 1.0])
    plt.title("Data Fitting Result")
    plt.legend()
    plt.savefig("../data/fit_result.png")

    plt.figure(2)
    plt.plot(range(1, len(error_history) + 1), error_history, "b-", linewidth=2)
    plt.xlabel("Generation (Iteration)")
    plt.ylabel("Error (RMSE)")
    plt.title("Error History (Learning Curve)")
    plt.grid(True)
    plt.savefig("../data/error_history.png")
    plt.show()


if __name__ == "__main__":
    main()
