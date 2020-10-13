from data.mpi_sintel import MpiSintel
from data.util import transform
from utils.config import opt

class Dataset:
    def __init__(self, opt):
        self.dataset = MpiSintel(root = dataset_dir)

    def __getitem__(self, idx):
        images, flow = self.dataset.get_example(idx)
        return images, flow

    def __len__(self):
        return self.dataset.size * self.dataset.replicates