import argparse
import platform
import nni
from utils import *
import warnings
from tqdm import trange
from torch.optim import AdamW, Adam
from model import *
import torch.nn as nn
from load_data import load_data
from torch_sparse import SparseTensor

def train(data):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].view(-1))

    loss.backward()
    optimizer.step()
    return loss, out

def val(data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        val_correct = pred[data.val_mask] == data.y[data.val_mask].squeeze()  # Check against ground-truth labels.
        val_correct = int(val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
        return val_correct


def test(data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='computers')
    parser.add_argument('--hidden', '-H', type=int, default=64)
    parser.add_argument('--lr', '-L', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model', '-M', type=str, default='MoE')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--activate', type=str, default='SwishGLU')
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--is_A', action='store_false', default=True)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--path', type=str, default=r'D:\DeskTop\资料\GNNProject\MPMT\GNNFormer\data')  # TODO
    parser.add_argument('--times', type=int, default=10, help="10 times exp for mean std")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    args_dict = vars(args)
    if platform.system().lower() == "linux":
        optimized_params = nni.get_next_parameter()
        args_dict.update(optimized_params)
    args = argparse.Namespace(**args_dict)

    # print details parameters
    print(args)
    # load data
    data, dataset = load_data(args.path, args.dataset)
    if args.is_A:
        row, col = data.edge_index
        row = row - row.min()
        data.edge_index_sparse = SparseTensor(row=row, col=col,
                                              sparse_sizes=(data.x.shape[0], data.x.shape[0])
                                              ).to_torch_sparse_coo_tensor()
    if not hasattr(dataset, 'num_node_features'):
        dataset.num_node_features = data.x.shape[1]
        print(data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    edge_weight = torch.ones(size=(data.edge_index.shape[1],))
    data.edge_weight = edge_weight

    data = data.to(device)
    num_nodes = data.x.shape[0]

    # train/val/test
    val_acc_list, test_acc_list = [], []
    ep_count_list = []
    for t in trange(args.times):
        best_val_acc = 0
        set_seed(t)

        # 2 random split for dataset
        train_rate = 0.48
        val_rate = 0.32
        if args.dataset in ['roman_empire','tolokers', 'questions']:
            train_masks, val_masks, test_masks = fixed_split(args.path, args.dataset, device)
            data.train_mask = train_masks[t]
            data.val_mask = val_masks[t]
            data.test_mask = test_masks[t]
        elif args.dataset in ['ogbn-arxiv', 'ogbn-arxiv2', 'ogbn-proteins']:
            split_idx = dataset.get_idx_split()
            data.train_mask = split_idx['train']
            data.val_mask = split_idx['valid']
            data.test_mask = split_idx['test']
        else:
            data = random_split(data, dataset.num_classes, train_rate, val_rate)

        if args.dataset in ['minesweeper', 'tolokers', 'questions', 'genius']:
            args.is_binary = True
            criterion = nn.BCELoss()
        else:
            args.is_binary = False
            criterion = nn.CrossEntropyLoss()

        # 3 load model
        if args.model == 'MoE':
            model = MoE(data, dataset, args).to(device)

        # 4 optimizer, loss function and other hypermeter
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # 5 train/val/test
        model.apply(reset_weight)
        es_count = patience = args.patience
        ep_count = 0
        test_acc = 0.0
        for ep in range(args.epochs):
            ep_count += 1
            loss = train(data)
            val_acc = val(data)
            if val_acc > best_val_acc:
                es_count = patience
                best_val_acc = val_acc
                test_acc = test(data)
            else:
                es_count -= 1
            if es_count <= 0:
                break
        val_acc_list.append(best_val_acc)
        test_acc_list.append(test_acc)

    val_acc_list = torch.tensor(val_acc_list)
    test_acc_list = torch.tensor(test_acc_list)
    print(f"{args.dataset} valid acc is {100 * val_acc_list.mean().item():.2f} ± {100 * val_acc_list.std().item():.2f}")
    print(f"{args.dataset} test acc is {100 * test_acc_list.mean().item():.2f} ± {100 * test_acc_list.std().item():.2f}")