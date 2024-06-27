


## Dependencies

- dgl==0.5.2
- gym==0.17.3
- hiredis==1.1.0
- idna==2.8
- importlib-metadata==2.0.0
- ipdb==0.13.4
- ipython==7.18.1
- matplotlib==3.3.2
- numpy
- opencv-python==4.0.0.21
- Pillow
- PyYAML>=5.4
- ray==1.0.0
- scipy==1.5.2
- torch==1.6.0
- torchvision==0.7.0
- tqdm==4.31.1





## Setup
Clone the [HandMeThat](https://github.com/Simon-Wan/HandMeThat) repository 

Clone the third party repositories (XTX, ALFWorld):
```
git clone https://github.com/princeton-nlp/XTX.git
git clone https://github.com/alfworld/alfworld.git
```
Then put this code under the folder "/baseline_models/LGR".


## Dataset

We can download *train* and *test* datasets used in from This [link](https://drive.google.com/file/d/1QoCL5veGnuJNhK1mMDryCrvpwVXTupdI/view), and place the zipped file at /datasets.


### Logic base

We follow [Logic_Point_Processes](https://github.com/FengMingquan-sjtu/Logic_Point_Processes_ICLR) to prepare the data. The logic rules contain temporal relations and spatial relations. We put the demo of logic rules in [baidu drive](https://pan.baidu.com/s/1x-_WDYPrI_WwCMKg6NherA), password:5de0. Dataset_id denotes the index of logic rules. You can design specific predicates and relation types based on the dataset you are using.

You can use deep-learning based algorithms to encode this logic rules as latent variable to guide the training process of your own models. We provide one pretrained model to process logic predicates (https://pan.baidu.com/s/1bR3S1tkw1Hu9jozOcGJ7CA password：1z79)


To train the model:

```
python scripts/train_rl.py --model LGR --observability fully
```

To evaluate the model (e.g., validate) on specific hardness level (e.g., level1):

```
python scripts/eval_rl.py --model LGR --observability fully --level level1 
```

## Contact
Should you have any question, please contact chengzhicao@mail.ustc.edu.cn.

## Notes and references
The code is based on [HandMeThat dataset](https://github.com/Simon-Wan/HandMeThat)

[1] [Li S, Feng M, Wang L, et al. Explaining point processes by learning interpretable temporal logic rules[C]//International Conference on Learning Representations. 2021.](https://openreview.net/pdf?id=P07dq7iSAGr) <br />
[2] [Li S, Wang L, Zhang R, et al. Temporal logic point processes[C]//International Conference on Machine Learning. PMLR, 2020: 5990-6000.](https://proceedings.mlr.press/v119/li20p/li20p.pdf) <br />
