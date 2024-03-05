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

We can download *train* and *test* datasets used in [google drive](https://drive.google.com/drive/folders/12QWa6cQKlC6SksU9uj7HCSVr4V6i6n86?usp=drive_link).

### Create your own dataset 
You can also create your dataset, and modify it to incorporate new tasks. For that, run

```bash
python gen_data/vh_init.py --num-per-apartment {NUM_APT} --task {TASK_NAME}
```
Where `NUM_APT` corresponds to the number of episodes you want for each apartment and task and `TASK_NAME` corresponds to the task name you want to generate, which can be `setup_table`, `clean_table`, `put_fridge`, `prepare_food`, `read_book`, `watch_tv` or `all` to generate all the tasks.

After creating your dataset, you can create the data for the **Watch** phase running the *Alice alone* baseline (see [Evaluate Baselines](#evaluate-baselines)).

You can then generate a dataset of tasks in a new environment where the tasks match those of the **Watch phase**. We do that in our work to make sure that the environment in the **Watch** phase is different than that in the **Help Phase** while having the same task specification. You can do that by running:

```bash
python gen_data/vh_init_gen_test.py
```

It will use the tasks from the test split of the **Watch** phase to create a **Help** dataset.



## Training
First, download the dataset for the **Watch** phase and put it under `dataset`. 
You can train the goal prediction model for the **Watch** phase as follows:

```bash
sh scripts/train_watch.sh
```

To test the goal prediction model, run:

```bash
sh scripts/test_watch.sh
```


### Evaluate baselines
Below is the code to evaluate the different planning-based models. The results will be saved in a folder called `test_results`. Make sure you create it first, one level above this repository. 


## Visualization

<img src= "pic/visual.jpg" width="120%">


## Contact
Should you have any question, please contact chengzhicao@mail.ustc.edu.cn.

## Notes and references
The  code is based on the paper 'Watch-And-Help: A Challenge for Social Perception and Human-AI Collaboration'(https://github.com/xavierpuigf/watch_and_help/tree/main)
