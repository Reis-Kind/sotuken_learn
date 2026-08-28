import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class BinarizedNeuroEvo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.where(self.fc1(x) >= 0, 1.0, -1.0)
        # 出力層は二値化しない
        x = self.fc2(x)
        return x


def evaluate(model, w, b, x, y):
    """
    個体の適応度を計算する関数

    """
    w_size = 128 * 784
    b_size = 128

    # GAではパラメータを一次元で管理しているため、一次元のパラメータをfc1とfc2に分割
    model.fc1.weight.data = w[:w_size].view(128, 784).clone()
    model.fc1.bias.data = b[:b_size].clone()
    model.fc2.weight.data = w[w_size:].view(10, 128).clone()
    model.fc2.bias.data = b[b_size:].clone()

    model.eval()
    with torch.no_grad():
        outputs = model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        predict = outputs.argmax(dim=1)
        acc = (predict == y).float().mean().item()


    return acc, loss.item(),

def genetic_algorithm(model, x_eval, y_eval):
    """
    島モデルGA
    
    """
    islands = 5
    models_per_island = 16
    generation = 200
    migration_interval = 50
    mutation_strength = 0.1
    origin_mutation_strength = mutation_strength
    mutation_rate = 0.001

    # 乱数シード固定
    torch.manual_seed(42)

    weight = 128 * 784 + 10 * 128
    bias = 128 + 10

    # 島の数 * 島あたりの個体数 * 重みの総数の乱数を生成し、二値化
    island_w = torch.where(
        torch.randn(islands, models_per_island, weight) > 0, 1.0, -1.0
    )
    # 島の数 * 島アタありの総数 * バイアスの総数の乱数を生成
    island_b = torch.randn(islands, models_per_island, bias) * 0.1

    loss_history = []
    acc_history = []

    for gen in range(generation):
        island_best_score = []
        island_best_w = []
        island_best_b = []
        island_worst = []

        for i in range(islands):
            scores_acc = []
            scores_loss = []
            for j in range(models_per_island):
                # 各島ごとのスコアを計算
                acc, loss = evaluate(model, island_w[i][j], island_b[i][j], x_eval, y_eval)
                scores_acc.append(acc)
                scores_loss.append(loss)

            # 正答率と損失を両方考慮して評価
            combined_scores = []
            for acc, loss in zip(scores_acc, scores_loss):
                # Accuracy をベースに，Loss が小さいほど少しだけ値が高くなるように計算
                score = acc - (0.01 * loss)
                combined_scores.append(score)

            best_j = np.argmax(combined_scores)
            worst_j = np.argmin(combined_scores)

            island_best_score.append(combined_scores[best_j])
            island_best_w.append(island_w[i][best_j].clone())
            island_best_b.append(island_b[i][best_j].clone())
            island_worst.append(worst_j)

            # エリート以外は交叉と突然変異
            for j in range(models_per_island):
                # エリートは除外
                if j != best_j:
                    # 重み(２値)の交叉、２値だから遺伝子を混ぜずに50%の確率でエリートの遺伝子と入れ替え
                    island_w[i][j] = torch.where(torch.rand(weight) < 0.5, island_best_w[i], island_w[i][j])

                    # 重み(２値)の突然変異↓のif文みたいなイメージ（実際は違う）
                    """if torch.rand(weight) < mutation_rate:
                        island_w[i][j] *= -1.0"""
                    w_mut_mask = torch.rand(weight) < mutation_rate
                    island_w[i][j][w_mut_mask] *= -1.0

                    # バイアス(float)の交叉、エリートのと自分のを50％ずつ合体
                    island_b[i][j] = (0.5 * island_best_b[i] + 0.5 * island_b[i][j])
                    # バイアス(float)の突然変異、
                    # True、Falseを.float()で1.0,-1.0にして突然変異を起こすか起こさないかのスイッチ
                    b_mut_mask = (torch.rand(bias) < mutation_rate).float()
                    b_noise = torch.randn(bias) * mutation_strength
                    island_b[i][j] += b_mut_mask * b_noise

        # 移住処理
        if (gen + 1) % migration_interval == 0:
            for now_island in range(islands):
                # 隣の島の番号を取得
                next_island = (now_island + 1) % islands
                # 隣の島の最弱の個体のインデックスを取得
                target = island_worst[next_island]
                island_w[next_island][target] = island_best_w[now_island].clone()
                island_b[next_island][target] = island_best_b[now_island].clone()

        # 全島の最高個体を取得
        best_island = np.argmax(island_best_score)
        best_w = island_best_w[best_island]
        best_b = island_best_b[best_island]

        best_acc, best_loss = evaluate(model, best_w, best_b, x_eval, y_eval)
        acc_history.append(best_acc)
        loss_history.append(best_loss)

        # 10世代ごとにログを出力（および最終世代）
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"{gen + 1:3d}世代 | Loss: {best_loss:.4f} | Accuracy: {best_acc * 100:.2f}%")

        # 変異強度の減衰(常に初期値を参照して計算)
        mutation_strength = origin_mutation_strength * (1.0 - (gen / generation) * 0.5)

    return best_w, best_b, acc_history, loss_history

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    eval_size = 2000
    indices = torch.randperm(len(dataset))[:eval_size]
    x_eval = torch.stack([dataset[i][0] for i in indices])
    y_eval = torch.tensor([dataset[i][1] for i in indices])

    model = BinarizedNeuroEvo()

    best_w, best_b, acc_history, loss_history = genetic_algorithm(model, x_eval, y_eval)

    # 最終的な最強パラメータをセット
    w1_size = 128 * 784
    b1_size = 128
    model.fc1.weight.data = best_w[:w1_size].view(128, 784)
    model.fc1.bias.data = best_b[:b1_size]
    model.fc2.weight.data = best_w[w1_size:].view(10, 128)
    model.fc2.bias.data = best_b[b1_size:]

    # 学習曲線の描画
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(acc_history) + 1), acc_history, "g-")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("Accuracy History (2-Layer BNN)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(loss_history) + 1), loss_history, "r-")
    plt.xlabel("Generation")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss History")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./output/binarize_ga.png")
    print("グラフを result.png に保存した．")


if __name__ == "__main__":
    main()





                    





