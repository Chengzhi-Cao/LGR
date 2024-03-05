# STRA

<!-- <img src= "https://github.com/Chengzhi-Cao/STRA/blob/main/network.jpg" width="120%"> -->
<!-- <img src= "https://github.com/Chengzhi-Cao/LGR/blob/main/pic/network.jpg" width="120%"> -->
<img src= "pic/network.jpg" width="120%">

This repository provides the implementation of the following paper:

> Enhancing Human-AI Collaboration Through Logic-Guided Reasoning
>
> Chengzhi Cao, Yinghao Fu*, Sheng Xu, Ruimao Zhang, Shuang Li
>
> International Conference on Learning Representations
>
> Paper Link:
>
> We present a systematic framework designed to enhance human-robot perception and collaboration through the integration of logical rules and Theory of Mind (ToM). Logical rules provide interpretable predictions and generalize well across diverse tasks, making them valuable for learning and decision-making. Leveraging the ToM for understanding others' mental states, our approach facilitates effective collaboration. In this paper, we employ logic rules derived from observational data to infer human goals and guide human-like agents. These rules are treated as latent variables, and a rule encoder is trained alongside a multi-agent system in the robot's mind. We assess the posterior distribution of latent rules using learned embeddings, representing entities and relations. Confidence scores for each rule indicate their consistency with observed data. Then, we employ a hierarchical reinforcement learning model with ToM to plan robot actions for assisting humans. Extensive experiments validate each component of our framework, and results on multiple benchmarks demonstrate that our model outperforms the majority of existing approaches.




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
Clone the [VirtualHome API](https://github.com/xavierpuigf/virtualhome.git) repository one folder above this repository

```bash
cd ..
git clone --branch wah https://github.com/xavierpuigf/virtualhome.git
cd virtualhome
pip install -r requirements.txt
```

Download the simulator, and put it in an `executable` folder, one folder above this repository


- [Download](http://virtual-home.org/release/simulator/v2.0/linux_exec.zip) Linux x86-64 version.
- [Download](http://virtual-home.org/release/simulator/v2.0/macos_exec.zip) Mac OS X version.
- [Download](http://virtual-home.org/release/simulator/v2.0/windows_exec.zip) Windows version.

### Install Requirements
```bash
pip install -r requirements.txt
```

## Dataset
We include a dataset of environments and activities that agents have to perform in them. During the **Watch** phase and the training of the **Help** phase, we use a dataset of 5 environments. When evaluating the **Help** phase, we use a dataset of 2 held out environments.

The **Watch** phase consists of a set of episodes in 5 environments showing Alice performing the task. These episodes were generated using a planner, and they can be downloaded [here](http://virtual-home.org/release/watch_and_help/watch_data.zip). The training and testing split information can be found in `datasets/watch_scenes_split.json`. 

The **Help** phase, contains a set of environments and tasks definitions. You can find the *train* and *test* datasets used in `dataset/train_env_set_help.pik` and `dataset/test_env_set_help.pik`. Note that the *train* environments are independent, whereas the *test* environments match the tasks in the **Watch** test split.



## Visualization

<img src= "pic/visual.jpg" width="120%">


## Contact
Should you have any question, please contact chengzhicao@mail.ustc.edu.cn.

## Notes and references
The  code is based on the paper 'Watch-And-Help: A Challenge for Social Perception and Human-AI Collaboration'(https://github.com/xavierpuigf/watch_and_help/tree/main)
