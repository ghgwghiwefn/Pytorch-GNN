from torch_geometric.data import Dataset

class SLICDataset(Dataset):
    def __init__(self, processed_graphs):
        super(SLICDataset, self).__init__()
        self.processed_graphs = processed_graphs

    def len(self):
        return len(self.processed_graphs)

    def get(self, idx):
        return self.processed_graphs[idx]