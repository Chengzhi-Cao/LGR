# STRA

<!-- <img src= "https://github.com/Chengzhi-Cao/STRA/blob/main/network.jpg" width="120%"> -->
<!-- <img src= "https://github.com/Chengzhi-Cao/LGR/blob/main/pic/network.jpg" width="120%"> -->
<img src= "pic/network.jpg" width="120%">

This repository provides the official PyTorch implementation of the following paper:

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
- Python
- Pytorch (1.4)
- scikit-image
- opencv-python



## Dataset

- Download deblur dataset from the [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro.html) .

- Unzip files ```dataset``` folder.



## Contact
Should you have any question, please contact chengzhicao@mail.ustc.edu.cn.

## Notes and references
The  code is based on the paper 'Watch-And-Help: A Challenge for Social Perception and Human-AI Collaboration'(https://github.com/xavierpuigf/watch_and_help/tree/main)
