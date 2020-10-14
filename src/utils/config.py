from pprint import pprint


class Config:

    start_epoch = 1
    total_epochs = 1
    batch_size = 8
    lr = 0.001
    
    train_n_batches = -1
    crop_size = [256, 256]
    schedule_lr_frequency = 0
    schedule_lr_fraction = 10
    rgb_max = 255.0

    validation_frequency = 5
    validation_n_batches = -1
    render_validation = True
    resume = 'PATH'
    log_frequency = 1

    number_workers = 1
    number_gpus = 1
    no_cuda = True

    seed = 1
    name = 'run'
    save = './work'

    inference = True
    inference_visualize = True
    inference_size = [-1, 1]
    inference_batch_size = 1
    inference_n_batches = -1
    save_flow = True

    skip_training = True
    skip_validation = True

    fp16 = True
    fp16_scale = 1024.0

    model = 'FlowNet2S'
    dataset_dir = './MPI-Sintel/flow/'
    flow_folder = './flows'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
