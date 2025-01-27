import random
import os.path as osp
import torch
import numpy as np

def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_weight(model):
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        model.reset_parameters()


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_split(data, num_classes, train_rate=0.48, val_rate=0.32):
    y_has_label = (data.y != -1).nonzero().contiguous().view(-1)
    num_nodes = y_has_label.shape[0]
    indices = torch.randperm(y_has_label.size(0))
    indices = y_has_label[indices]
    train_num = int(round(train_rate * num_nodes))
    val_num = int(round(val_rate * num_nodes))
    train_index = indices[:train_num]
    val_index = indices[train_num:train_num + val_num]
    test_index = indices[train_num + val_num:]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    return data


def fixed_split(path, name, device):
    data = np.load(osp.join(path, name + '.npz'))
    train_masks = torch.tensor(data['train_masks']).to(device)
    val_masks = torch.tensor(data['val_masks']).to(device)
    test_masks = torch.tensor(data['test_masks']).to(device)
    return train_masks, val_masks, test_masks
    # train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
    # val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
    # test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]
    # return train_idx_list, val_idx_list, test_idx_list


