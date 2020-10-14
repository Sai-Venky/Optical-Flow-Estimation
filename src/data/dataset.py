from data.mpi_sintel import MpiSintel
from utils.config import opt

class Dataset:

    '''
        Base Dataset Class
    '''

    def __init__(self, dataset_dir):
        self.dataset = MpiSintel(root = dataset_dir)


    def __getitem__(self, idx):

        '''
            Return the two image and flow applicable for those images
            Arguments:
                idx  :- the id
            Returns:
                images :- Returns the two images for that index
                flow   :- Returns the flow for that index and associated images
        '''

        images, flow = self.dataset.get_example(idx)
        return images, flow


    def __len__(self):

        '''
            Returns the length of the dataset
        '''

        return self.dataset.size * self.dataset.replicates