# Deep-Anomaly-Detection-for-Time-Series-A-Survey  

This repository updates the comprehensive list of state of the art deep anomaly detection methods for Time Series

## Table of Contents

- [Methods for Deep Anomaly Detection in Time Series](#methods-for-deep-anomaly-detection-in-time-series)  
  - [Forecasting Models](#forecasting-models)  
   
  - [Reconstruction Models](#reconstruction-models)  
    - [Autoencoder-Based Methods](#autoencoder-based-methods)  
    - [GNN/Transformer-Based Models](#gnntransformer-based-models)  
  - [Generative Models](#generative-models)  
    - [VAE-Based Methods](#vae-based-methods)  
    - [GAN-Based Methods](#gan-based-methods)  
  - [Density Models](#density-models)  
  - [Contrastive Models](#contrastive-models)  

- [Application Areas](#application-areas)  
  - [Network Services](#network-services)  
  - [Cyber-Physical System](#cyber-physical-system)  
  - [Smart Grid](#smart-grid)  
  - [Smart City](#smart-city)  
  - [Healthcare](#healthcare)  
  - [Meteorological and Natural Disaster](#meteorological-and-natural-disaster)  
  - [Other Areas](#other-areas)  

- [Challenges and Future Research Directions](#challenges-and-future-research-directions)  
  - [Data Augmentation](#data-augmentation)  
    - [Frequency Augmentation](#frequency-augmentation)  
    - [Decomposition-Based Augmentation](#decomposition-based-augmentation)  
    - [Data Generation and Anomaly Synthesis](#data-generation-and-anomaly-synthesis)  
  - [Robustness of the Models](#robustness-of-the-models)  
    - [Data Filtering and Pseudo-labeling Approaches](#data-filtering-and-pseudo-labeling-approaches)  
    - [Distribution-Aware Learning Objectives](#distribution-aware-learning-objectives)  
    - [Contextual Representation and Architecture Enhancements](#contextual-representation-and-architecture-enhancements)  
  - [Generalization of the Models](#generalization-of-the-models)  
    - [Intra-domain Generalization](#intra-domain-generalization)  
    - [Cross-domain Generalization](#cross-domain-generalization)  
  - [Foundational Models, Pre-trained Models, and Large Language Models](#foundational-models-pre-trained-models-and-large-language-models)  
    - [Foundational Models](#foundational-models)  
    - [Pre-trained Models](#pre-trained-models)  
    - [Leveraging Large Language Models for Time Series Analysis](#leveraging-large-language-models-for-time-series-analysis)  
      - [TS for LLM](#ts-for-llm)  
      - [LLM for TS](#llm-for-ts)  
  - [AutoML for Time Series Anomaly Detection](#automl-for-time-series-anomaly-detection)  
    - [Automated Hyperparameter Tuning and Model Selection](#automated-hyperparameter-tuning-and-model-selection)  
    - [Optimization Objectives and Task Adaptation](#optimization-objectives-and-task-adaptation)  
  - [Lightweight Models](#lightweight-models)  
    - [Simplifying Network Architectures and Reducing Model Parameters](#simplifying-network-architectures-and-reducing-model-parameters)  
    - [Optimizing the Execution Pipeline](#optimizing-the-execution-pipeline)  
  - [Others](#others)  
    - [Efficient Labeling](#efficient-labeling)  
    - [Irregular Data Modeling](#irregular-data-modeling)  
    - [Anomaly Prediction and Early Warning](#anomaly-prediction-and-early-warning)  

## Methods for Deep Anomaly Detection in Time Series

### Forecasting Models

| Model                                                                                                | Main Architecture      | Learning Strategy | UTS/MTS   | Code |
|------------------------------------------------------------------------------------------------------|------------------------|-------------------|-----------|------|
| [Ergen et al. (TNNLS 2020)](https://doi.org/10.1109/TNNLS.2019.2935975)                              | LSTM                   | Un.               | MTS       | —    |
| [AQADF (TKDE 2022)](https://doi:10.1109/TKDE.2020.3014806)                                           | LSTM, CNN              | Self.             | UTS       | —    |
| [AD-LTI (TKDE 2022)](https://ieeexplore.ieee.org/document/9247440)                                   | GRU                    | Un.               | UTS       | —    |
| [SES-AD (ESWA 2022)](https://www.sciencedirect.com/science/article/pii/S0957417422011423?via%3Dihub) | LSTM                   | Un.               | MTS       | [Code](https://github.com/JakeJiUThealth/SESAD_V1.0)   |
| [DeepAnT (Access 2019)](https://ieeexplore.ieee.org/document/8581424)                                | CNN                    | Un.               | UTS & MTS | —    |
| [CAD (FSE 2023)](https://dl.acm.org/doi/10.1145/3611643.3613896)                                     | CNN                    | Un.               | MTS       | [Code](https://github.com/dawnvince/MTS_CAD)    |
| [GDN (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/16523)                            | GNN                    | Un.               | MTS       | [Code](https://github.com/d-ailin/GDN)    |
| [GraphAD (SIGIR 2022)](https://dl.acm.org/doi/10.1145/3477495.3531848)                               | GNN                    | Un.               | MTS       | —    |
| [HGTMA (J. Supercomput. 2023)](https://link.springer.com/article/10.1007/s11227-023-05503-w)         | GNN, Transformer       | Un.               | MTS       | —    |
| [CST-GL (TNNLS 2023)](https://ieeexplore.ieee.org/document/10316684)                                 | CNN, GNN               | Un.               | MTS       | [Code](https://github.com/huankoh/CST-GL)    |
| [CGAD (arxiv 2023)](https://arxiv.org/abs/2312.09478)                                                | GNN                    | Un.               | MTS       | [Code](https://github.com/falihgoz/cgad)    |
| [Graph-MoE (arxiv 2024)](https://arxiv.org/abs/2412.19108)                                           | GNN                    | Un.               | MTS       | —    |
| [GCAD (arxiv 2025)](https://arxiv.org/abs/2501.13493)                                                | GNN                    | Un.               | MTS       | —    |
| [DDGCT (Cluster Comput. 2025)](https://link.springer.com/article/10.1007/s10586-024-04707-w)         | GNN, Transformer       | Un.               | MTS       | —    |
| [GDTS (Neurocomputing 2025)](https://www.sciencedirect.com/science/article/pii/S0925231224019398?via%3Dihub)          | GNN, Transformer       | Un.               | MTS       | —    |



### Reconstruction Models

| Model                                                                                                                                        | Main Architecture     | Learning Strategy | UTS/MTS     | Code                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|-------------------|-------------|------------------------------------------------------|
| [MSCRED (AAAI 2019)](https://ojs.aaai.org/index.php/AAAI/article/view/3942)                                                                  | CNN, LSTM, AE         | Un.               | MTS         | [Code](https://github.com/7fantasysz/MSCRED)         |
| [Kieu et al. (IJCAI 2019)](https://www.ijcai.org/proceedings/2019/378)                                                                       | RNN, AE               | Un.               | UTS & MTS   | —                                                    |
| [USAD (SIGKDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403392)                                                                         | AE                    | Un.               | MTS         | [Code](https://github.com/manigalati/usad)           |
| [APAE (IJCAI 2020)](http://dx.doi.org/10.24963/IJCAI.2020/173)                                                                               | AE                    | Un.               | MTS         | —                                                    |
| [RAMED (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17152)                                                                  | RNN, AE               | Un.               | UTS         | —                                                    |
| [TSAE (ICDMW 2021)](https://ieeexplore.ieee.org/document/9679903)                                                                            | AE                    | Un.               | MTS         | —                                                    |
| [CAE-Ensemble (VLDB 2021)](https://dl.acm.org/doi/10.14778/3494124.3494142)                                                                  | CNN, AE               | Un.               | MTS         | [Code](https://github.com/d-gcc/CAE-Ensemble)        |
| [PAMFE (Appl. Intell. 2023)](https://link.springer.com/article/10.1007/s10489-022-04324-3)                                                   | CNN, AE               | Un.               | MTS         | —                                                    |
| [SVD-AE (NN 2024)](https://www.sciencedirect.com/science/article/pii/S0893608023006469?via%3Dihub)                                           | AE                    | Un.               | MTS         | —                                                    |
| [Song et al. (TNNLS 2024)](https://ieeexplore.ieee.org/document/10091135)                                                                    | AE                    | Un.               | MTS         | —                                                    |
| [AEVAE (TNNLS 2025)](https://ieeexplore.ieee.org/document/10345653)                                                                          | AE                    | Un.               | UTS         | —                                                    |
| [STGAT-MAD (ICASSP 2022)](https://ieeexplore.ieee.org/document/9747274)                                                                      | GNN, LSTM             | Un.               | MTS         | [Code](https://github.com/zhanjun717/STGAT)          |
| [MIXAD (ICPR 2024)](https://link.springer.com/chapter/10.1007/978-3-031-78189-6_16)                                                          | GNN, LSTM             | Un.               | MTS         | [Code](https://github.com/mhkim9714/MIXAD)           |
| [Anomaly Transformer (ICLR 2022)](https://openreview.net/forum?id=3bqJqBgimY)                                                                | Transformer           | Un.               | MTS         | [Code](https://github.com/thuml/Anomaly-Transformer) |
| [MAN-QSM (ICME 2023)](http://dx.doi.org/10.1109/ICME55011.2023.00466)                                                                        | Transformer           | Un.               | MTS         | [Code](https://github.com/zeg-datamining/MAN-QSM)    |
| [TranAD (VLDB 2022)](https://dl.acm.org/doi/10.14778/3514061.3514067)                                                                        | Transformer           | Un.               | MTS         | [Code](https://github.com/imperial-qore/TranAD)      |
| [Uni-AD (ISSRE 2022)](http://dx.doi.org/10.1109/ISSRE55969.2022.00014)                                                                       | Transformer           | Un.               | MTS         | [Code](https://github.com/IntelligentDDS/Uni-AD)     |
| [GCFormer (ICDM 2023)](https://ieeexplore.ieee.org/document/10415739/)                                                                       | Transformer           | Un.               | MTS         | —                                                    |
| [MEMTO (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract-Conference.html) | Transformer           | Un.               | MTS         | [Code](https://github.com/gunny97/MEMTO)             |
| [EdgeConvFormer (arxiv 2023)](https://arxiv.org/abs/2312.01729)                                                                              | GNN, Transformer      | Un.               | MTS         | —                                                    |
| [ADFormer (FGCS 2023)](https://www.sciencedirect.com/science/article/pii/S0167739X23000560?via%3Dihub)                                       | Transformer           | Un.               | MTS         | —                                                    |
| [FLAD (JCRD 2023)](https://crad.ict.ac.cn/cn/article/doi/10.7544/issn1000-1239.202220490)                                                    | TCN, Transformer      | Un.               | MTS         | —                                                    |
| [PAFormer (TNNLS 2025)](https://ieeexplore.ieee.org/document/10352961)                                                                       | Transformer           | Un.               | MTS         | —                                                    |
| [GDFormer (arxiv 2025)](https://arxiv.org/abs/2501.18196)                                                                                                                    | Transformer           | Un.               | MTS         | [Code](https://github.com/yuppielqx/GDformer)                                             |


### Generative Models

#### VAE-Based Methods

#### GAN-Based Methods

### Density Models

### Contrastive Models

## Application Areas

### Network Services

### Cyber-Physical System

### Smart Grid

### Smart City

### Healthcare

### Meteorological and Natural Disaster

### Other Areas

## Challenges and Future Research Directions

### Data Augmentation

#### Frequency Augmentation

#### Decomposition-Based Augmentation

#### Data Generation and Anomaly Synthesis

### Robustness of the Models

#### Data Filtering and Pseudo-labeling Approaches

#### Distribution-Aware Learning Objectives

#### Contextual Representation and Architecture Enhancements

### Generalization of the Models

#### Intra-domain Generalization

#### Cross-domain Generalization

### Foundational Models, Pre-trained Models, and Large Language Models

#### Foundational Models

#### Pre-trained Models

#### Leveraging Large Language Models for Time Series Analysis

##### TS for LLM

##### LLM for TS

### AutoML for Time Series Anomaly Detection

#### Automated Hyperparameter Tuning and Model Selection

#### Optimization Objectives and Task Adaptation

### Lightweight Models

#### Simplifying Network Architectures and Reducing Model Parameters

#### Optimizing the Execution Pipeline

### Others

#### Efficient Labeling

#### Irregular Data Modeling

#### Anomaly Prediction and Early Warning
