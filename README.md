# Optical-Flow-Estimation

Implements Optical Flow Estimation.
Optical Flow Estimation is done to visualize and capture the movement in between images.
This is very useful in training videos etc where 'capturing' the flow is important.

The code is straightforward and easy to follow.

<ul>
<li><a href="https://github.com/Sai-Venky/Optical-Flow-Estimation#installation-and-running">Installation and Running</a></li>
<li><a href="https://github.com/Sai-Venky/Optical-Flow-Estimation#dataset">Dataset</a></li>
<li><a href="https://github.com/Sai-Venky/Optical-Flow-Estimation#directory-layout">Directory Layout</a></li>
<li><a href="https://github.com/Sai-Venky/Optical-Flow-Estimation#results">Results</a></li>
<li><a href="https://github.com/Sai-Venky/Optical-Flow-Estimation#acknowledgement">Acknowledgement</a></li>
<li><a href="https://github.com/Sai-Venky/Optical-Flow-Estimation#contributing">Contributing</a></li>
<li><a href="https://github.com/Sai-Venky/Optical-Flow-Estimation#licence">Licence</a></li>
</ul>

### Installation and Running

```pip install requirements.txt```

To run the model, please run 
```python src/main.py```


### Dataset

The MPI Sintel dataset can be downloaded from the following links :-

`http://sintel.is.tue.mpg.de/`

The dataset can be extracted and stored in the parent directory. If not, its location can be changed in `src/utils/config.py` at `dataset_dir`

### Directory Layout

The directory structure is as follows :-

* **data :** contains the necessary files needed for loading the MPI Sintel dataset along with transformation functions.
  * dataset : base class which instantiates the mpi sintel dataset and transforms the raw data.
  * util : helper functions for preprocessing the image, flows
  * mpi_sintel : contains class needed to process/parse the mpi dataset data
* **models :** this contains the optical flow models and all of its constituent methods.
    * optical_flow : core model with train, validate functions. Instantiates and Calls all other models (flownet, loss).
    * flow_net2SD : contains the flow net SD model
    * loss : contains the loss functions used by model
* **utils :** this contains the utility methods needed.
    * config : contains the configuration/options.

 ### Results

#### Ground Truth
![Ground Truth ](https://github.com/Sai-Venky/Optical-Flow-Estimation/blob/master/imgs/gt1.png)

#### Predicted
![Predicted ](https://github.com/Sai-Venky/Optical-Flow-Estimation/blob/master/imgs/pred1.png)

#### Ground Truth
![Ground Truth ](https://github.com/Sai-Venky/Optical-Flow-Estimation/blob/master/imgs/gt2.png)

#### Predicted
![Predicted ](https://github.com/Sai-Venky/Optical-Flow-Estimation/blob/master/imgs/pred2.png)

 ### Acknowledgement

 https://github.com/NVIDIA/flownet2-pytorch/

 ### Contributing

 You can contribute in serveral ways such as creating new features, improving documentation etc.

 ### Licence

 MIT Licence
