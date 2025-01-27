from torch_geometric.datasets import Amazon, Actor

def load_data(path, name):
    if name in ['computers', 'photo']:
        dataset = Amazon(root=path, name=name)
    elif name in ['Actor']:
        dataset = Actor(root=path + '/film')
    return dataset[0], dataset


