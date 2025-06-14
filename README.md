# Deep-Anomaly-Detection-for-Time-Series-A-Survey

This repository updates the comprehensive list of state of the art deep anomaly detection methods for Time Series

## Table of Contents

- [Methods for Deep Anomaly Detection in Time Series](#methods-for-deep-anomaly-detection-in-time-series)
    - [Forecasting Models](#forecasting-models)

    - [Reconstruction Models](#reconstruction-models)

    - [Generative Models](#generative-models)

    - [Density Models](#density-models)
    - [Contrastive Models](#contrastive-models)
    - [Hybrid and Other Methods](#hybrid-and-other-methods)
- [Application Areas](#application-areas)

- [Challenges and Future Research Directions](#challenges-and-future-research-directions)
    - [Data Augmentation](#data-augmentation)
       
    - [Robustness of the Models](#robustness-of-the-models)
       
    - [Generalization of the Models](#generalization-of-the-models)
        
    - [Foundational Models, Pre-trained Models, and Large Language Models](#foundational-models-pre-trained-models-and-large-language-models)
       
    - [AutoML for Time Series Anomaly Detection](#automl-for-time-series-anomaly-detection)
        
    - [Lightweight Models](#lightweight-models)
        
    - [Others](#others)
        

## Methods for Deep Anomaly Detection in Time Series

### Forecasting Models

| Model                                                                                                        | Main Architecture | Learning Strategy | UTS/MTS   | Code                                                 |
|--------------------------------------------------------------------------------------------------------------|-------------------|-------------------|-----------|------------------------------------------------------|
| [Ergen et al. (TNNLS 2020)](https://doi.org/10.1109/TNNLS.2019.2935975)                                      | LSTM              | Un.               | MTS       | —                                                    |
| [AQADF (TKDE 2022)](https://doi:10.1109/TKDE.2020.3014806)                                                   | LSTM, CNN         | Self.             | UTS       | —                                                    |
| [AD-LTI (TKDE 2022)](https://ieeexplore.ieee.org/document/9247440)                                           | GRU               | Un.               | UTS       | —                                                    |
| [SES-AD (ESWA 2022)](https://www.sciencedirect.com/science/article/pii/S0957417422011423?via%3Dihub)         | LSTM              | Un.               | MTS       | [Code](https://github.com/JakeJiUThealth/SESAD_V1.0) |
| [DeepAnT (Access 2019)](https://ieeexplore.ieee.org/document/8581424)                                        | CNN               | Un.               | UTS & MTS | —                                                    |
| [CAD (FSE 2023)](https://dl.acm.org/doi/10.1145/3611643.3613896)                                             | CNN               | Un.               | MTS       | [Code](https://github.com/dawnvince/MTS_CAD)         |
| [GDN (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/16523)                                    | GNN               | Un.               | MTS       | [Code](https://github.com/d-ailin/GDN)               |
| [GraphAD (SIGIR 2022)](https://dl.acm.org/doi/10.1145/3477495.3531848)                                       | GNN               | Un.               | MTS       | —                                                    |
| [HGTMA (J. Supercomput. 2023)](https://link.springer.com/article/10.1007/s11227-023-05503-w)                 | GNN, Transformer  | Un.               | MTS       | —                                                    |
| [CST-GL (TNNLS 2023)](https://ieeexplore.ieee.org/document/10316684)                                         | CNN, GNN          | Un.               | MTS       | [Code](https://github.com/huankoh/CST-GL)            |
| [CGAD (arxiv 2023)](https://arxiv.org/abs/2312.09478)                                                        | GNN               | Un.               | MTS       | [Code](https://github.com/falihgoz/cgad)             |
| [Graph-MoE (arxiv 2024)](https://arxiv.org/abs/2412.19108)                                                   | GNN               | Un.               | MTS       | —                                                    |
| [GCAD (arxiv 2025)](https://arxiv.org/abs/2501.13493)                                                        | GNN               | Un.               | MTS       | —                                                    |
| [DDGCT (Cluster Comput. 2025)](https://link.springer.com/article/10.1007/s10586-024-04707-w)                 | GNN, Transformer  | Un.               | MTS       | —                                                    |
| [GDTS (Neurocomputing 2025)](https://www.sciencedirect.com/science/article/pii/S0925231224019398?via%3Dihub) | GNN, Transformer  | Un.               | MTS       | —                                                    |

### Reconstruction Models

| Model                                                                                                                                        | Main Architecture | Learning Strategy | UTS/MTS   | Code                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|-----------|------------------------------------------------------|
| [MSCRED (AAAI 2019)](https://ojs.aaai.org/index.php/AAAI/article/view/3942)                                                                  | CNN, LSTM, AE     | Un.               | MTS       | [Code](https://github.com/7fantasysz/MSCRED)         |
| [Kieu et al. (IJCAI 2019)](https://www.ijcai.org/proceedings/2019/378)                                                                       | RNN, AE           | Un.               | UTS & MTS | —                                                    |
| [USAD (SIGKDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403392)                                                                         | AE                | Un.               | MTS       | [Code](https://github.com/manigalati/usad)           |
| [APAE (IJCAI 2020)](http://dx.doi.org/10.24963/IJCAI.2020/173)                                                                               | AE                | Un.               | MTS       | —                                                    |
| [RAMED (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17152)                                                                  | RNN, AE           | Un.               | UTS       | —                                                    |
| [TSAE (ICDMW 2021)](https://ieeexplore.ieee.org/document/9679903)                                                                            | AE                | Un.               | MTS       | —                                                    |
| [CAE-Ensemble (VLDB 2021)](https://dl.acm.org/doi/10.14778/3494124.3494142)                                                                  | CNN, AE           | Un.               | MTS       | [Code](https://github.com/d-gcc/CAE-Ensemble)        |
| [PAMFE (Appl. Intell. 2023)](https://link.springer.com/article/10.1007/s10489-022-04324-3)                                                   | CNN, AE           | Un.               | MTS       | —                                                    |
| [SVD-AE (NN 2024)](https://www.sciencedirect.com/science/article/pii/S0893608023006469?via%3Dihub)                                           | AE                | Un.               | MTS       | —                                                    |
| [Song et al. (TNNLS 2024)](https://ieeexplore.ieee.org/document/10091135)                                                                    | AE                | Un.               | MTS       | —                                                    |
| [AEVAE (TNNLS 2025)](https://ieeexplore.ieee.org/document/10345653)                                                                          | AE                | Un.               | UTS       | —                                                    |
| [STGAT-MAD (ICASSP 2022)](https://ieeexplore.ieee.org/document/9747274)                                                                      | GNN, LSTM         | Un.               | MTS       | [Code](https://github.com/zhanjun717/STGAT)          |
| [MIXAD (ICPR 2024)](https://link.springer.com/chapter/10.1007/978-3-031-78189-6_16)                                                          | GNN, LSTM         | Un.               | MTS       | [Code](https://github.com/mhkim9714/MIXAD)           |
| [Anomaly Transformer (ICLR 2022)](https://openreview.net/forum?id=3bqJqBgimY)                                                                | Transformer       | Un.               | MTS       | [Code](https://github.com/thuml/Anomaly-Transformer) |
| [MAN-QSM (ICME 2023)](http://dx.doi.org/10.1109/ICME55011.2023.00466)                                                                        | Transformer       | Un.               | MTS       | [Code](https://github.com/zeg-datamining/MAN-QSM)    |
| [TranAD (VLDB 2022)](https://dl.acm.org/doi/10.14778/3514061.3514067)                                                                        | Transformer       | Un.               | MTS       | [Code](https://github.com/imperial-qore/TranAD)      |
| [Uni-AD (ISSRE 2022)](http://dx.doi.org/10.1109/ISSRE55969.2022.00014)                                                                       | Transformer       | Un.               | MTS       | [Code](https://github.com/IntelligentDDS/Uni-AD)     |
| [GCFormer (ICDM 2023)](https://ieeexplore.ieee.org/document/10415739/)                                                                       | Transformer       | Un.               | MTS       | —                                                    |
| [MEMTO (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract-Conference.html) | Transformer       | Un.               | MTS       | [Code](https://github.com/gunny97/MEMTO)             |
| [EdgeConvFormer (arxiv 2023)](https://arxiv.org/abs/2312.01729)                                                                              | GNN, Transformer  | Un.               | MTS       | —                                                    |
| [ADFormer (FGCS 2023)](https://www.sciencedirect.com/science/article/pii/S0167739X23000560?via%3Dihub)                                       | Transformer       | Un.               | MTS       | —                                                    |
| [PAFormer (TNNLS 2025)](https://ieeexplore.ieee.org/document/10352961)                                                                       | Transformer       | Un.               | MTS       | —                                                    |
| [GDFormer (arxiv 2025)](https://arxiv.org/abs/2501.18196)                                                                                    | Transformer       | Un.               | MTS       | [Code](https://github.com/yuppielqx/GDformer)        |

### Generative Models

| Model                                                                                              | Main Architecture | Learning Strategy | UTS/MTS | Code                                                |
|----------------------------------------------------------------------------------------------------|-------------------|-------------------|---------|-----------------------------------------------------|
| [OmniAnomaly (SIDKDD 2019)](http://dx.doi.org/10.1145/3292500.3330672)                             | RNN, VAE          | Un.               | MTS     | [Code](https://github.com/smallcowbaby/OmniAnomaly) |
| [InterFusion (SIGKDD 2021)](https://dl.acm.org/doi/10.1145/3447548.3467075)                        | VAE               | Un.               | MTS     | [Code](https://github.com/zhhlee/InterFusion)       |
| [SIS-VAE (TNNLS 2021)](https://ieeexplore.ieee.org/document/9064715)                               | VAE               | Un.               | MTS     | —                                                   |
| [GRELEN (IJCAI 2022)](https://www.ijcai.org/proceedings/2022/332)                                  | GNN, VAE          | Un.               | MTS     | —                                                   |
| [GAN-AD (arxiv 2019)](https://arxiv.org/abs/1809.04758)                                            | RNN, GAN          | Un.               | MTS     | [Code](https://github.com/LiDan456/GAN-AD)          |
| [MAD-GAN (ICANN 2019)](https://link.springer.com/chapter/10.1007/978-3-030-30490-4_56)             | RNN, GAN          | Un.               | MTS     | [Code](https://github.com/LiDan456/MAD-GANs)        |
| [DAEMON (ICDE 2021)](https://ieeexplore.ieee.org/document/9458835/)                                | AE, GAN           | Un.               | MTS     | —                                                   |
| [M3GAN (KBS 2023)](https://www.sciencedirect.com/science/article/pii/S0950705123003350?via%3Dihub) | GAN               | Un.               | MTS     | [Code](https://github.com/SLZWVICTOR/M3GAN)         |
| [DiffGAN (arxiv 2025)](https://arxiv.org/abs/2501.01591)                                           | GAN, DM           | Un.               | MTS     | [Code](https://github.com/guangqiangwu/diffgan)     |

### Density Models

| Model                                                                                                                 | Main Architecture    | Learning Strategy | UTS/MTS | Code                                      |
|-----------------------------------------------------------------------------------------------------------------------|----------------------|-------------------|---------|-------------------------------------------|
| [NSIBF (SIGKDD 2021)](https://dl.acm.org/doi/10.1145/3447548.3467137)                                                 | LSTM                 | Un.               | MTS     | [Code](https://github.com/cfeng783/NSIBF) |
| [BSSAD (arxiv 2023)](https://arxiv.org/abs/2301.13031)                                                                | RNN, AE              | Un.               | MTS     | —                                         |
| [GANF (ICLR 2022)](https://openreview.net/pdf?id=45L_dgP48Vd)                                                         | RNN, GNN, NF         | Un.               | MTS     | [Code](https://github.com/enyandai/ganf)  |
| [MTGFlow (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/25623)                                         | RNN, GNN, NF         | Un.               | MTS     | [Code](https://github.com/zqhang/MTGFLOW) |
| [GNF (IEEE Intell. Syst. 2023)](https://ieeexplore.ieee.org/document/10061588/)                                       | GNN, Transformer, NF | Un.               | MTS     | —                                         |
| [AFNF (ISA Trans. 2023)](https://www.sciencedirect.com/science/article/pii/S0019057823004020?via%3Dihub)              | Transformer, NF      | Un.               | MTS     | —                                         |
| [SCNF (Reliab. Eng. Syst. Saf. 2023)](https://www.sciencedirect.com/science/article/pii/S0951832023003241?via%3Dihub) | GNN, GRU             | Un.               | MTS     | —                                         |

### Contrastive Models

| Model                                                                      | Main Architecture | Learning Strategy | UTS/MTS   | Code                                                     |
|----------------------------------------------------------------------------|-------------------|-------------------|-----------|----------------------------------------------------------|
| [DCDetector (SIGKDD 2023)](https://dl.acm.org/doi/10.1145/3580305.3599295) | Transformer       | Self.             | MTS       | [Code](https://github.com/DAMO-DI-ML/KDD2023-DCdetector) |
| [ContrastAD (IJCNN 2023)](https://ieeexplore.ieee.org/document/10191358/)  | AE                | Self.             | UTS & MTS | —                                                        |
| [TiCTok (Access 2023)](https://ieeexplore.ieee.org/document/10201844/)     | CNN, Transformer  | Self.             | MTS       | —                                                        |
| [TriAD (ICDE 2024)](http://dx.doi.org/10.1109/ICDE60146.2024.00080)        | CNN               | Self.             | UTS       | —                                                        |
| [PCRTA (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/548)           | TCN               | Self.             | MTS       | —                                                        |

### Hybrid and Other Methods

| Model                                                                                                   | Main Architecture  | Learning Strategy | UTS/MTS | Objection Function Type      | Code                                               |
|---------------------------------------------------------------------------------------------------------|--------------------|-------------------|---------|------------------------------|----------------------------------------------------|
| [MTAD-GAT (ICDM 2020)](https://ieeexplore.ieee.org/document/9338317)                                    | GNN, GRU, VAE, MLP | Un.               | MTS     | Forecasting + Generative     | [Code](https://github.com/ML4ITS/mtad-gat-pytorch) |
| [MST-GAT (Inf. Fusion 2023)](http://dx.doi.org/10.1016/j.inffus.2022.08.011)                            | GNN, TCN, VAE, MLP | Un.               | MTS     | Forecasting + Generative     | —                                                  |
| [FuSAGNet (SIGKDD 2022)](https://dl.acm.org/doi/10.1145/3534678.3539117)                                | GNN, AE            | Semi.             | MTS     | Forecasting + Reconstruction | —                                                  |
| [DVGCRN (ICML 2022)](https://proceedings.mlr.press/v162/chen22x)                                        | GNN, VAE           | Un.               | MTS     | Forecasting + Generative     | —                                                  |
| [CAE-M (TKDE 2021)](https://ieeexplore.ieee.org/document/9507359)                                       | AE, LSTM           | Un.               | MTS     | Forecasting + Reconstruction | —                                                  |
| [HybridAD (TETCI 2023)](https://ieeexplore.ieee.org/document/10177380)                                  | GRU, CNN, AE       | Un.               | MTS     | Forecasting + Reconstruction | —                                                  |
| [CAE-AD (SDM 2022)](http://dx.doi.org/10.1137/1.9781611977653.ch78)                                     | AE, LSTM           | Self.             | MTS     | Contrastive + Reconstruction | —                                                  |
| [COCA (Inf. Sci. 2022)](https://www.sciencedirect.com/science/article/pii/S0020025522008775?via%3Dihub) | AE                 | Self.             | MTS     | Contrastive + One Class      | —                                                  |
| [DCAD (NN 2023)](https://www.sciencedirect.com/science/article/pii/S0893608023005385?via%3Dihub)        | CNN, Transformer   | Self.             | MTS     | Contrastive + Reconstruction | —                                                  |
| [ACVAE (NN 2024)](https://www.sciencedirect.com/science/article/pii/S0893608023007281?via%3Dihub)       | VAE                | Self.             | MTS     | Contrastive + Generative     | —                                                  |
| [DiffAD (SIGKDD 2023)](https://dl.acm.org/doi/10.1145/3580305.3599391)                                  | DM                 | Self.             | MTS     | Imputation                   | [Code](https://github.com/ChunjingXiao/DiffAD)     |
| [ImDiffusion (VLDB 2023)](https://dl.acm.org/doi/10.14778/3632093.3632101)                              | DM                 | Self.             | MTS     | Imputation                   | [Code](https://github.com/17000cyh/IMDiffusion)    |

## Application Areas

| Application                       | Model                                                                                                                                                                       | Main Architecture      | Strategy | UTS/MTS | Objective Function           | Code                                                        |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|----------|---------|------------------------------|-------------------------------------------------------------|
| Network Services                  | [Donut (WWW 2018)](https://dl.acm.org/doi/10.1145/3178876.3185996)                                                                                                          | VAE                    | Un.      | UTS     | Generative                   | [Code](https://github.com/NetManAIOps/donut)                |
| Network Services                  | [SR-CNN (SIGKDD 2019)](https://dl.acm.org/doi/10.1145/3292500.3330680)                                                                                                      | CNN                    | Un.      | UTS     | Forecasting                  | [Code](https://github.com/y-bar/ml-based-anomaly-detection) |
| Network Services                  | [Abdulaal et al. (SIGKDD 2021)](https://dl.acm.org/doi/10.1145/3447548.3467174)                                                                                             | AE                     | Un.      | MTS     | Reconstruction               | —                                                           |
| Network Services                  | [SDFVAE (WWW 2021)](https://dl.acm.org/doi/10.1145/3442381.3450013)                                                                                                         | VAE                    | Un.      | MTS     | Generative                   | [Code](https://github.com/dlagul/SDFVAE)                    |
| Network Services                  | [TopoMAD (TNNLS 2023)](https://ieeexplore.ieee.org/document/9228885)                                                                                                        | LSTM, GNN, VAE         | Un.      | MTS     | Generative                   | [Code](https://github.com/QAZASDEDC/TopoMAD)                |
| Network Services                  | [CGNN-MHSA-AR (FGCS 2023)](https://www.sciencedirect.com/science/article/pii/S0167739X23000973?via%3Dihub)                                                                  | GNN, GRU, Transformer  | Un.      | MTS     | Forecasting                  | —                                                           |
| Network Services                  | [CMAnomaly (ISSRE 2023)](https://ieeexplore.ieee.org/abstract/document/10301224/)                                                                                           | MLP                    | Un.      | MTS     | Forecasting                  | —                                                           |
| Network Services                  | [ACVAE (JSAC 2023)](https://ieeexplore.ieee.org/document/9844802/)                                                                                                          | VAE                    | Semi.    | MTS     | Contrastive + Generative     | —                                                           |
| Network Services                  | [Peng et al. (Comput. Networks 2022)](https://www.sciencedirect.com/science/article/pii/S138912862200250X?via%3Dihub)                                                       | LSTM, GAT              | Un.      | MTS     | Forecasting + Reconstruction | —                                                           |
| Cyber-Physical System             | [GRN (Comput. Secur. 2023)](http://dx.doi.org/10.1016/j.cose.2023.103094)                                                                                                   | GRU, GAT               | Un.      | MTS     | Forecasting                  | —                                                           |
| Cyber-Physical System             | [HiSTAR (TII 2023)](https://ieeexplore.ieee.org/document/9925618)                                                                                                           | LSTM, GNN, MLP         | Un.      | MTS     | One Class                    | —                                                           |
| Cyber-Physical System             | [EGNN (FGCS 2024)](http://dx.doi.org/10.1016/j.future.2023.09.028)                                                                                                          | GAT                    | Un.      | MTS     | Forecasting                  | —                                                           |
| Cyber-Physical System             | [GTA (IoTJ 2022)](https://ieeexplore.ieee.org/document/9497343/)                                                                                                            | GAT, Transformer       | Un.      | MTS     | Reconstruction               | —                                                           |
| Cyber-Physical System             | [DUMA (JSEN 2023)](https://ieeexplore.ieee.org/document/9969633)                                                                                                            | Transformer            | Un.      | MTS     | Forecasting                  | —                                                           |
| Cyber-Physical System             | [AMBi-GAN (TII 2023)](https://ieeexplore.ieee.org/document/9426423)                                                                                                         | LSTM, GAN, Transformer | Un.      | MTS     | Reconstruction + Generative  | —                                                           |
| Smart Grid                        | [Zheng et al. (The Journal of China Universities of Posts and Telecommunications 2017)](https://www.sciencedirect.com/science/article/abs/pii/S1005888517602437?via%3Dihub) | RNN, AE                | Un.      | UTS     | Reconstruction               | —                                                           |
| Smart Grid                        | [Basumallik et al. (Int. J. Electr. Power Energy Syst. 2019)](https://www.sciencedirect.com/science/article/pii/S0142061518319884?via%3Dihub)                               | CNN                    | Un.      | MTS     | Forecasting                  | —                                                           |
| Smart Grid                        | [Fenza et al. (Access 2019)](https://ieeexplore.ieee.org/document/8604042)                                                                                                  | LSTM                   | Un.      | UTS     | Forecasting                  | —                                                           |
| Smart City                        | [Zhang et al. (IJCAI 2019)](https://www.ijcai.org/proceedings/2019/837)                                                                                                     | GNN, MLP               | Un.      | MTS     | Forecasting                  | —                                                           |
| Smart City                        | [STGAN (TNNLS 2022)](https://ieeexplore.ieee.org/document/9669110/)                                                                                                         | GNN, GAN, GRU          | Un.      | MTS     | Generative                   | [Code](https://github.com/dleyan/STGAN)                     |
| Smart City                        | [Liu and Li (JSEN 2023)](https://ieeexplore.ieee.org/document/10068423)                                                                                                     | CNN                    | Un.      | MTS     | Forecasting                  | —                                                           |
| Healthcare                        | [Sucheta et al. (DSAA 2015)](https://ieeexplore.ieee.org/document/7344872)                                                                                                  | LSTM                   | Un.      | UTS     | Forecasting                  | —                                                           |
| Healthcare                        | [BeatGAN (IJCAI 2019)](https://www.ijcai.org/proceedings/2019/616)                                                                                                          | GAN                    | Un.      | UTS     | Generative                   | [Code](https://github.com/Vniex/BeatGAN)                    |
| Healthcare                        | [TSRNet (ISBI 2024)](https://ieeexplore.ieee.org/document/10635676)                                                                                                         | Transformer            | Un.      | UTS     | Reconstruction               | —                                                           |
| Healthcare                        | [TSAD-C (arxiv 2023)](https://arxiv.org/abs/2308.12563)                                                                                                                     | GNN, DPM               | Un.      | MTS     | Reconstruction               | —                                                           |
| Healthcare                        | [AMSL (TKDE 2023)](https://ieeexplore.ieee.org/document/9669068)                                                                                                            | AE                     | Self.    | MTS     | Reconstruction               | —                                                           |
| Meteorological & Natural Disaster | [Arora et al. (Concurrency Comput. Pract. Exper. 2021)](https://onlinelibrary.wiley.com/doi/10.1002/cpe.6707)                                                               | LSTM                   | Un.      | MTS     | Forecasting                  | —                                                           |
| Meteorological & Natural Disaster | [TacSas (ICML 2024 Workshop)](https://openreview.net/forum?id=q6E14hueUt)                                                                                                   | CNN, AE                | Un.      | MTS     | Reconstruction               | —                                                           |
| Meteorological & Natural Disaster | [ConvNetQuake (Sci. Adv. 2018)](https://www.science.org/doi/10.1126/sciadv.1700578)                                                                                         | CNN                    | Su.      | MTS     | One Class                    | —                                                           |
| Meteorological & Natural Disaster | [PhaseNet (Geophys. J. Int. 2019)](https://academic.oup.com/gji/advance-article/doi/10.1093/gji/ggy423/5129142)                                                             | CNN, U-net             | Su.      | MTS     | One Class                    | —                                                           |
| Meteorological & Natural Disaster | [Cai et al. (Appl. Geophys. 2019)](https://link.springer.com/article/10.1007/s11770-019-0774-1)                                                                             | LSTM                   | Un.      | MTS     | Forecasting                  | —                                                           |
| Meteorological & Natural Disaster | [CrowdQuake (SIGKDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403378)                                                                                                  | CNN, RNN               | Un.      | MTS     | Forecasting                  | —                                                           |
| Others                            | [Zhang and Zou (J. Phys. Conf. Ser. 2018)](https://iopscience.iop.org/article/10.1088/1742-6596/1061/1/012012)                                                              | LSTM                   | Un.      | MTS     | Forecasting                  | —                                                           |
| Others                            | [LSTM-NDT (SIGKDD 2018)](https://dl.acm.org/doi/10.1145/3219819.3219845)                                                                                                    | LSTM                   | Un.      | MTS     | Forecasting                  | [Code](https://github.com/khundman/telemanom)               |
| Others                            | [Kaufman et al. (arxiv 2022)](https://arxiv.org/abs/2207.11466)                                                                                                             | LSTM, CNN              | Un.      | MST     | Forecasting                  | —                                                           |

## Challenges and Future Research Directions

### Data Augmentation

| Method                              | Model                                                                                                     | Code                                                  |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| Frequency Augmentation              | [CAE-AD (Inf. Sci. 2022)](https://www.sciencedirect.com/science/article/pii/S0020025522008775?via%3Dihub) | —                                                     |
| Frequency Augmentation              | [DCFF-MTAD (Sensors 2022)](https://www.mdpi.com/1424-8220/23/8/3910)                                      | —                                                     |
| Frequency Augmentation              | [KfreqGAN (KBS 2022)](https://www.sciencedirect.com/science/article/pii/S0950705121009837?via%3Dihub)     | —                                                     |
| Frequency Augmentation              | [FCVAE (WWW 2024)](https://dl.acm.org/doi/10.1145/3589334.3645710)                                        | [Code](https://github.com/CSTCloudOps/FCVAE)          |
| Frequency Augmentation              | [CATCH (ICLR 2025)](https://openreview.net/forum?id=m08aK3xxdJ)                                           | [Code](https://github.com/decisionintelligence/catch) |
| Decomposition-Based Augmentation    | [TADNet (ICASSP 2024)](https://ieeexplore.ieee.org/document/10446482)                                     | —                                                     |
| Decomposition-Based Augmentation    | [AURORA (DMKD 2021)](http://dx.doi.org/10.1007/s10618-021-00771-7)                                        | —                                                     |
| Decomposition-Based Augmentation    | [Lei et al. (KBS 2023)](https://www.sciencedirect.com/science/article/pii/S0950705123007529?via%3Dihub)   | —                                                     |
| Decomposition-Based Augmentation    | [TFAD (CIKM 2022)](https://dl.acm.org/doi/10.1145/3511808.3557470)                                        | [Code](https://github.com/damo-di-ml/cikm22-tfad)     |
| Decomposition-Based Augmentation    | [MEGA (TIM 2023)](https://ieeexplore.ieee.org/document/9954430)                                           | —                                                     |
| Date Generation & Anomaly Synthesis | [CutAddPaste (SIGKDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671739)                               | [Code](https://github.com/ruiking04/CutAddPaste)                                              |
| Date Generation & Anomaly Synthesis | [GenIAS (arxiv 2025)](https://arxiv.org/abs/2502.08262)                                                   | —                                                     |
| Date Generation & Anomaly Synthesis | [BeatGAN (IJCAI 2019)](https://www.ijcai.org/proceedings/2019/616)                                                                                                   | [Code](https://github.com/Vniex/BeatGAN)                                                     |

### Robustness of the Models

| Method                                      | Model                                                                                                            | Code                                          |
|---------------------------------------------|------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| Data Filtering & Pseudo‐labeling            | [SaVAE‐SR (Neurocomputing 2021)](https://www.sciencedirect.com/science/article/pii/S0925231221009346?via%3Dihub) | —                                             |
| Data Filtering & Pseudo‐labeling            | [FGANomaly (TKDE 2023)](https://ieeexplore.ieee.org/document/9618824)                                            | [Code](https://github.com/sxxmason/FGANomaly) |
| Data Filtering & Pseudo‐labeling            | [STAD‐GAN (TKDD 2023)](https://dl.acm.org/doi/10.1145/3572780)                                                   | —                                             |
| Distribution‐Aware Learning Objectives      | [RDSSM (TKDE 2022)](https://ieeexplore.ieee.org/document/9773982/)                                               | —                                             |
| Distribution‐Aware Learning Objectives      | [RoSAS (IPM 2023)](https://www.sciencedirect.com/science/article/pii/S0306457323001966?via%3Dihub)               | [Code](https://github.com/xuhongzuo/rosas/)   |
| Contextual Representation & Architecture    | [CAE‐M (TKDE 2021)](https://ieeexplore.ieee.org/document/9507359)                                                | —                                             |
| Contextual Representation & Architecture    | [Li et al. (ICASSP 2022)](https://ieeexplore.ieee.org/document/9747668/)                                                                                      | [Code](https://github.com/hanhuili/MTCE-AnomalyDetection)                                      |


### Generalization of the Models

| Method                       | Model                                                                                                                                      | Code                                       |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| Intra-domain Generalization  | [CDAM (NPL 2023)](https://link.springer.com/article/10.1007/s11063-022-11015-0)                                                            | —                                          |
| Intra-domain Generalization  | [D3R (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html) | [Code](https://github.com/ForestsKing/D3R) |
| Intra-domain Generalization  | [Liu et al. (WWWJ 2023)](https://link.springer.com/article/10.1007/s11280-023-01181-z)                                                     | —                                          |
| Intra-domain Generalization  | [ADTCD (IoTJ 2023)](https://ieeexplore.ieee.org/document/10097858/)                                                                        | —                                          |
| Intra-domain Generalization  | [M2N2 (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29210)                                                                 | [Code](https://github.com/carrtesy/m2n2)   |
| Cross-domain Generalization  | [AnoTransfer (JSAC 2022)](https://ieeexplore.ieee.org/document/9791391)                                                                    | [Code](https://github.com/anotransfer/AnoTransfer-code)                                   |
| Cross-domain Generalization  | [ContexTDA (SDM 2023)](https://epubs.siam.org/doi/10.1137/1.9781611977653.ch76)                                                            | —                                          |
| Cross-domain Generalization  | [FS-ADAPT (Inf. Sci. 2023)](https://www.sciencedirect.com/science/article/pii/S0020025523011957?via%3Dihub)                                | —                                          |


### Foundational Models, Pre-trained Models, and Large Language Models

| Method                   | Model                                                                                                                                               | Code                                                                      |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Foundational Models      | [TS2Vec (AAAI 2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20881)                                                                        | [Code](https://github.com/zhihanyue/ts2vec)                               |
| Foundational Models      | [TimesNet (ICLR 2023)](https://openreview.net/forum?id=ju_Uqw384Oq)                                                                                 | [Code](https://github.com/thuml/timesnet)                                 |
| Foundational Models      | [MSD-Mixer (VLDB 2024)](https://dl.acm.org/doi/10.14778/3654621.3654637)                                                                            | [Code](https://github.com/zshhans/MSD-Mixer)                              |
| Foundational Models      | [Correlated Attention Transformer (arxiv 2023)](https://arxiv.org/abs/2311.11959)                                                                   | —                                                                         |
| Foundational Models      | [ModernTCN (ICLR 2024)](https://openreview.net/forum?id=vpJMJerXHU)                                                                                 | [Code](https://github.com/luodhhh/moderntcn)                              |
| Foundational Models      | [TimeDRL (ICDE 2024)](https://ieeexplore.ieee.org/document/10597874)                                                                                | [Code](https://github.com/blacksnail789521/TimeDRL)                       |
| Pre-trained Models       | [TF-C (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/194b8dac525581c346e30a2cebe9a369-Abstract-Conference.html)         | [Code](https://github.com/mims-harvard/tfc-pretraining)                   |
| Pre-trained Models       | [SimMTM (NeurIPS 2023)](https://openreview.net/pdf?id=ginTcBUnL8)                                                                                   | [Code](https://github.com/thuml/simmtm)                                   |
| Pre-trained Models       | [DADA (ICLR 2025)](https://openreview.net/forum?id=aKcd7ImG5e)                                                                                      | [Code](https://github.com/decisionintelligence/DADA)                      |
| Pre-trained Models       | [KAD-Disformer (SIGKDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671522)                                                                       | [Code](https://github.com/NetManAIOps/KAD-Disformer)                      |
| TS for LLM               | [One Fits All (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/86c17de05579cde52025f9984e6e2ebb-Abstract-Conference.html) | [Code](https://github.com/damo-di-ml/one_fits_all)                        |
| TS for LLM               | [TIME-LLM (ICLR 2024)](https://openreview.net/forum?id=Unb5CVPtae)                                                                                  | [Code](https://github.com/kimmeen/time-llm)                               |
| TS for LLM               | [TEMPO (ICLR 2024)](https://openreview.net/forum?id=YH5w12OUuU)                                                                                     | [Code](https://github.com/dc-research/tempo)                              |
| TS for LLM               | [TEST (ICLR 2024)](https://openreview.net/forum?id=Tuh4nZVb0g)                                                                                      | [Code](https://github.com/scxsunchenxi/test)                              |
| TS for LLM               | [UniTime (WWW 2024)](https://dl.acm.org/doi/10.1145/3589334.3645434)                                                                                | [Code](https://github.com/liuxu77/UniTime)                                |
| TS for LLM               | [LLM4TS (TIST 2025)](https://dl.acm.org/doi/10.1145/3719207)                                                                                        | [Code](https://github.com/blacksnail789521/LLM4TS)                        |
| TS for LLM               | [PromptCast (TKDE 2024)](https://ieeexplore.ieee.org/document/10356715/)                                                                            | [Code](https://github.com/HaoUNSW/PISA)                                   |
| TS for LLM               | [GPT-4V (arxiv 2023)](https://arxiv.org/abs/2311.02782)                                                                                             | [Code](https://github.com/caoyunkang/gpt4v-for-generic-anomaly-detection) |
| TS for LLM               | [AnomalyLLM (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/239)                                                                               | —                                                                         |
| TS for LLM               | [AnomalyGPT (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/27963)                                                                    | [Code](https://github.com/casia-iva-lab/anomalygpt)                       |
| LLM for TS               | [TimeGPT (arxiv 2023)](https://arxiv.org/abs/2310.03589)                                                                                            | [Code](https://github.com/Nixtla/nixtla)                                  |
| LLM for TS               | [Timer (ICML 2024)](https://proceedings.mlr.press/v235/liu24cb.html)                                                                                | [Code](https://github.com/thuml/Large-Time-Series-Model)                  |
| LLM for TS               | [MOMENT (ICML 204)](https://proceedings.mlr.press/v235/goswami24a.html)                                                                             | [Code](https://github.com/moment-timeseries-foundation-model/moment)      |
| LLM for TS               | [UniTS (NeurIPS 2024)](https://proceedings.mlr.press/v235/goswami24a.html)                                                                          | [Code](https://github.com/mims-harvard/UniTS)                             |
| LLM for TS               | [TimeMixer++ (ICLR 2025)](https://openreview.net/forum?id=1CLzLXSFNn)                                                                                                                         | [Code](https://github.com/kwuking/TimeMixer)                                                                  |


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
