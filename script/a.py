"""同じチェックポイントから、GAなし/GA1回ありを5epoch比較する。

既存の ga_test_cifar.py と同じフォルダで:
    uv run ga_branch_compare.py
モデル定義だけを既存ファイルから読み込む(mainは実行しない)。
局所解の証明ではなく、GA後の継続学習への影響を調べる実験。
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def make_optimizer(model):
    return optim.Adam(
        list(model.layers.parameters()) + list(model.bn1.parameters())
        + list(model.bn2.parameters()) + list(model.fc.parameters()), lr=0.001
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss_sum += nn.functional.cross_entropy(output, y, reduction='sum').item()
        correct += (output.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total, loss_sum / total


def train_epoch(branches, loader, device):
    # 同じバッチを両方へ渡し、画像と順番を確実に揃える。
    totals = [0.0] * len(branches)
    for model, _, _ in branches:
        model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for i, (model, optimizer, _) in enumerate(branches):
            loss = nn.functional.cross_entropy(model(x), y)
            model.update(optimizer, loss)
            totals[i] += loss.item() * y.numel()
    for _, _, scheduler in branches:
        scheduler.step()
    return [v / len(loader.dataset) for v in totals]


def apply_ga(model, select_loader, device, seed):
    origin = copy.deepcopy(model.state_dict())
    signs = [(layer.weight.detach() >= 0).clone() for layer in model.layers]
    before, _ = evaluate(model, select_loader, device)
    best_acc, best_state, selected = before, origin, 0
    generator = torch.Generator(device=device).manual_seed(seed)
    candidates = []
    for i in range(8):
        model.load_state_dict(origin)
        with torch.no_grad():
            for layer in model.layers:
                mask = torch.rand(layer.weight.shape, device=device,
                                  generator=generator) < 0.001
                layer.weight[mask] *= -1
        acc, _ = evaluate(model, select_loader, device)
        changed = sum(((layer.weight.detach() >= 0) != old).sum().item()
                      for layer, old in zip(model.layers, signs))
        print(f'  候補{i + 1}: 符号変化={changed}個, 選択用={acc:.2f}%')
        candidates.append(dict(candidate=i + 1, changed=changed, accuracy=acc))
        if acc > best_acc:
            best_acc, selected = acc, i + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    changed = sum(((layer.weight.detach() >= 0) != old).sum().item()
                  for layer, old in zip(model.layers, signs))
    print(f'  GA選択: 候補{selected} (0=元モデル), {before:.2f}% → {best_acc:.2f}%')
    return dict(before=before, after=best_acc, selected=selected,
                changed=changed, candidates=candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-file', default='ga_test_cifar.py')
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--follow', type=int, default=5)
    parser.add_argument('--schedule-epochs', type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.follow < 1 or args.warmup + args.follow > args.schedule_epochs:
        parser.error('warmup/followは1以上、合計はschedule-epochs以下にしてください')
    source = Path(args.model_file).resolve()
    if not source.is_file():
        parser.error(f'モデルのファイルがありません: {source}')
    spec = importlib.util.spec_from_file_location('user_cifar_model', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(args.seed)
    # 対象モデルにはDropoutやランダムなデータ拡張がない。
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用デバイス: {device}')
    model = module.BinaryConnectCifar10().to(device)
    optimizer = make_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_epochs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train, val = random_split(dataset, [45000, 5000],
                             generator=torch.Generator().manual_seed(42))
    select, check = random_split(val, [500, 4500],
                                generator=torch.Generator().manual_seed(args.seed + 1000))
    test = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    def eval_loader(data):
        return DataLoader(data, batch_size=128, shuffle=False,
                          generator=torch.Generator().manual_seed(0))

    select_loader, check_loader, test_loader = map(eval_loader, [select, check, test])

    def train_loader(epoch):
        return DataLoader(train, batch_size=64, shuffle=True,
                          generator=torch.Generator().manual_seed(args.seed + 10000 + epoch))

    out = Path('output') / f'ga_branch_seed{args.seed}_{time.time_ns()}'
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args) | dict(model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                              torch_version=torch.__version__, device=str(device))
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for epoch in range(1, args.warmup + 1):
        loss = train_epoch([(model, optimizer, scheduler)], train_loader(epoch), device)[0]
        print(f'共通学習 [{epoch}/{args.warmup}] Loss={loss:.4f}', flush=True)

    # 分岐点の状態を保存。最適化履歴と学習率も両分岐へ引き継ぐ。
    checkpoint = dict(model=copy.deepcopy(model.state_dict()),
                      optimizer=copy.deepcopy(optimizer.state_dict()),
                      scheduler=copy.deepcopy(scheduler.state_dict()), epoch=args.warmup,
                      rng=torch.get_rng_state(), config=config)
    if device.type == 'cuda':
        checkpoint['cuda_rng'] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, out / 'branch_checkpoint.pt')
    branches = []
    for _ in range(2):
        branch = copy.deepcopy(model)
        opt = make_optimizer(branch)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.schedule_epochs)
        opt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        sched.load_state_dict(copy.deepcopy(checkpoint['scheduler']))
        branches.append((branch, opt, sched))
    started = time.perf_counter()
    result = apply_ga(branches[1][0], select_loader, device, args.seed + 2000)
    result['seconds'] = time.perf_counter() - started
    (out / 'ga.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if result['selected'] == 0:
        print('候補不採用: 今回はGAによる重み変更なし。A/B一致の対照実験になります。')

    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['step', 'epoch', 'branch', 'train_loss', 'next_lr',
                  'check_acc', 'check_loss', 'test_acc', 'test_loss']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in range(args.follow + 1):
            losses = [None, None]
            if step:
                losses = train_epoch(branches, train_loader(args.warmup + step), device)
            accuracies = []
            for name, (branch, opt, _), loss in zip(['A_no_ga', 'B_ga'], branches, losses):
                check_acc, check_loss = evaluate(branch, check_loader, device)
                test_acc, test_loss = evaluate(branch, test_loader, device)
                writer.writerow(dict(step=step, epoch=args.warmup + step, branch=name,
                                     train_loss=loss, next_lr=opt.param_groups[0]['lr'],
                                     check_acc=check_acc, check_loss=check_loss,
                                     test_acc=test_acc, test_loss=test_loss))
                print(f'分岐後{step}epoch {name}: 確認用={check_acc:.2f}% '
                      f'(Loss={check_loss:.4f}) | Test={test_acc:.2f}%', flush=True)
                accuracies.append(test_acc)
            print(f'  Test差 B-A: {accuracies[1] - accuracies[0]:+.2f}ポイント')
            stream.flush()
    print(f'結果保存先: {out.resolve()}')
    print('step=0はGA直後、step=5は追加学習5epoch後。局所解からの脱出の証明ではありません。')


if __name__ == '__main__':
    main()
