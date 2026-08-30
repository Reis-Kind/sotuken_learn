import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class BinarizedNeuroEvo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=False)
        # 移動平均は無視
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=False)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.where(x >= 0, 1.0, -1.0)
        x = self.fc2(x)
        return x


def evaluate(model, w, b, x, y):
    """
    個体の適応度を計算する関数

    """
    w_size = 128 * 784

    model.fc1.weight.data = w[:w_size].view(128, 784).clone()
    model.fc2.weight.data = w[w_size:].view(10, 128).clone()


    model.bn1.weight.data = b[:128].clone()
    model.bn1.bias.data   = b[128:256].clone()
    model.fc2.bias.data   = b[256:].clone()

    model.eval()
    with torch.no_grad():
        outputs = model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        predict = outputs.argmax(dim=1)
        acc = (predict == y).float().mean().item()

    return acc, loss.item(),

def tournament_select(combined_scores, k):
    """
    交叉するための親をランダムにk体選び，その中で最もスコアの高い個体のインデックスを返す関数

    """
    candidates = np.random.choice(len(combined_scores), k, replace=False)

    best_idx = candidates[0]
    best_score = combined_scores[candidates[0]]

    for i in candidates:
        if combined_scores[i] > best_score:
            best_score = combined_scores[i]
            best_idx = i

    return best_idx 

def genetic_algorithm(model, x_eval, y_eval):

    """
    島モデルGA
    
    """
    islands = 5
    models_per_island = 32
    generation = 200
    migration_interval = 50
    mutation_strength = 0.1
    origin_mutation_strength = mutation_strength
    mutation_rate = 0

    torch.manual_seed(42)
    np.random.seed(42)

    weight = 128 * 784 + 10 * 128
    bias = 128 + 128 + 10

    island_w = torch.where(
        torch.randn(islands, models_per_island, weight) > 0, 1.0, -1.0
    )
    island_b = torch.randn(islands, models_per_island, bias) * 0.1

    loss_history = []
    acc_history = []

    for gen in range(generation):

        progress = gen / generation
        mutation_rate = 0.001 * (1.0 - progress) + 0.0005 * progress

        island_best_score = []

        island_best_w = []
        island_best_b = []
        island_worst = []

        for i in range(islands):
            scores_acc = []
            scores_loss = []
            for j in range(models_per_island):
                acc, loss = evaluate(model, island_w[i][j], island_b[i][j], x_eval, y_eval)
                scores_acc.append(acc)
                scores_loss.append(loss)

            combined_scores = []
            for acc, loss in zip(scores_acc, scores_loss):
                score = acc - (0.01 * loss)
                combined_scores.append(score)

            best_j = np.argmax(combined_scores)
            worst_j = np.argmin(combined_scores)

            island_best_score.append(combined_scores[best_j])
            island_best_w.append(island_w[i][best_j].clone())
            island_best_b.append(island_b[i][best_j].clone())
            island_worst.append(worst_j)

            for j in range(models_per_island):
                if j != best_j:

                    parent_idx = tournament_select(combined_scores, 3)
                    parent_w = island_w[i][parent_idx]
                    parent_b = island_b[i][parent_idx]


                    island_w[i][j] = torch.where(torch.rand(weight) < 0.5, parent_w, island_w[i][j])

                    w_mut_mask = torch.rand(weight) < mutation_rate
                    island_w[i][j][w_mut_mask] *= -1.0

                    island_b[i][j] = (0.5 * parent_b + 0.5 * island_b[i][j])
                    b_mut_mask = (torch.rand(bias) < mutation_rate).float()
                    b_noise = torch.randn(bias) * mutation_strength
                    island_b[i][j] += b_mut_mask * b_noise

        if (gen + 1) % migration_interval == 0:
            for now_island in range(islands):
                next_island = (now_island + 1) % islands
                target = island_worst[next_island]
                island_w[next_island][target] = island_best_w[now_island].clone()
                island_b[next_island][target] = island_best_b[now_island].clone()

        best_island = np.argmax(island_best_score)
        best_w = island_best_w[best_island]
        best_b = island_best_b[best_island]

        best_acc, best_loss = evaluate(model, best_w, best_b, x_eval, y_eval)
        acc_history.append(best_acc)
        loss_history.append(best_loss)

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"{gen + 1:3d}世代 | Loss: {best_loss:.4f} | Accuracy: {best_acc * 100:.2f}%")

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

    w1_size = 128 * 784
    model.fc1.weight.data = best_w[:w1_size].view(128, 784)
    model.fc2.weight.data = best_w[w1_size:].view(10, 128)
    model.bn1.weight.data = best_b[:128].clone()
    model.bn1.bias.data = best_b[128:256].clone()
    model.fc2.bias.data = best_b[256:].clone()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(acc_history) + 1), acc_history, "g-")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("Accuracy History (BN, affine=True)")
    plt.grid(True)


    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(loss_history) + 1), loss_history, "r-")
    plt.xlabel("Generation")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss History")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./output/binarize_ga_bn_affine.png")

    np.save("./output/acc_bn_affine.npy", np.array(acc_history))
    np.save("./output/loss_bn_affine.npy", np.array(loss_history))

    print("グラフを ./output/binarize_ga_bn_affine.png に保存した．")


if __name__ == "__main__":
    main()