from pprint import pprint


class Config:

    '''
        Contains the configuration for the model
    '''

    total_epochs = 1
    batch_size = 8
    lr = 0.001
    
    crop_size = [256, 256]
    rgb_max = 255.0

    number_workers = 1
    number_gpus = 1

    seed = 1
    name = 'run'
    save = './work'
    inference_size = [-1, 1]

    save_flow = True

    dataset_dir = './MPI-Sintel/flow/'
    flow_folder = './flows'
    load_path = '/Users/ecom-v.ramesh/Documents/Personal/2020/DL/Optical-Flow-Estimation/opticalflow'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
