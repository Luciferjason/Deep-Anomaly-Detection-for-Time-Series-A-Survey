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

| Model                      | Main Architecture      | Learning Strategy | UTS/MTS   | Code |
|----------------------------|------------------------|-------------------|-----------|------|
| [Ergen et al. (TNNLS 2020)](https://doi.org/10.1109/TNNLS.2019.2935975)         | LSTM                   | Un.               | MTS       | —    |
| [AQADF (TKDE 2022)](https://doi:10.1109/TKDE.2020.3014806)                | LSTM, CNN              | Self.             | UTS       | —    |
| [AD-LTI (TKDE 2022)](https://ieeexplore.ieee.org/document/9247440)              | GRU                    | Un.               | UTS       | —    |
| [SES-AD (ESWA 2022)](https://www.sciencedirect.com/science/article/pii/S0957417422011423?via%3Dihub)              | LSTM                   | Un.               | MTS       | [Code](https://github.com/JakeJiUThealth/SESAD_V1.0)   |
| [DeepAnT (IEEE Access 2019)](https://ieeexplore.ieee.org/document/8581424)             | CNN                    | Un.               | UTS & MTS | —    |
| [CAD (FSE 2023)](https://dl.acm.org/doi/10.1145/3611643.3613896)                 | CNN                    | Un.               | MTS       | [Code](https://github.com/dawnvince/MTS_CAD)    |
| [GDN (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/16523)                 | GNN                    | Un.               | MTS       | [Code](https://github.com/d-ailin/GDN)    |
| [GraphAD (SIGIR 2022)](https://dl.acm.org/doi/10.1145/3477495.3531848)             | GNN                    | Un.               | MTS       | —    |
| [HGTMA (J. Supercomput. 2023)](https://link.springer.com/article/10.1007/s11227-023-05503-w)              | GNN, Transformer       | Un.               | MTS       | —    |
| [CST-GL (TNNLS 2023)](https://ieeexplore.ieee.org/document/10316684)              | CNN, GNN               | Un.               | MTS       | [Code](https://github.com/huankoh/CST-GL)    |
| [CGAD (arxiv 2023)](https://arxiv.org/abs/2312.09478)                | GNN                    | Un.               | MTS       | [Code](https://github.com/falihgoz/cgad)    |
| [Graph-MoE (arxiv 2024)](https://arxiv.org/abs/2412.19108)           | GNN                    | Un.               | MTS       | —    |
| [GCAD (arxiv 2025)](https://arxiv.org/abs/2501.13493)                | GNN                    | Un.               | MTS       | —    |
| [DDGCT (Cluster Comput. 2025)](https://link.springer.com/article/10.1007/s10586-024-04707-w)               | GNN, Transformer       | Un.               | MTS       | —    |



### Reconstruction Models

#### Autoencoder-Based Methods

#### GNN/Transformer-Based Models

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
