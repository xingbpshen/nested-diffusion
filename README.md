<h1 align="center">
Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2310.15952">
    <img src="https://img.shields.io/badge/arXiv-2310.15952-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://scholar.googleusercontent.com/scholar.bib?q=info:8EHZDokzghEJ:scholar.google.com/&output=citation&scisdr=ClH2O-RNEO7Ohs15-Jw:AFWwaeYAAAAAZ2B_4Jy-b_c6Cd04v5ELUUrLUSo&scisig=AFWwaeYAAAAAZ2B_4It2kE2X8oFSph-xM3O-CAg&scisf=4&ct=citation&cd=-1&hl=en">
    <img src="https://img.shields.io/badge/Cite-BibTeX-green.svg" alt="BibTeX">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.8-blue.svg" alt="Python Version">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-1.10-red.svg" alt="PyTorch Version">
  </a>
</p>

This repository contains the official implementation of the paper:
> __Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles__  
> [Xing Shen](https://scholar.google.com/citations?hl=en&user=U69NqfQAAAAJ), [Hengguan Huang](https://scholar.google.com/citations?hl=en&user=GQm1eZEAAAAJ), [Brennan Nichyporuk](https://scholar.google.com/citations?user=GYKrS-EAAAAJ&hl=en), [Tal Arbel](https://www.cim.mcgill.ca/~arbel/)  
> _arXiv Preprint, Oct 2023_  
> __[Paper](https://arxiv.org/abs/2310.15952)&nbsp;/ [BibTeX](https://scholar.googleusercontent.com/scholar.bib?q=info:8EHZDokzghEJ:scholar.google.com/&output=citation&scisdr=ClH2O-RNEO7Ohs15-Jw:AFWwaeYAAAAAZ2B_4Jy-b_c6Cd04v5ELUUrLUSo&scisig=AFWwaeYAAAAAZ2B_4It2kE2X8oFSph-xM3O-CAg&scisf=4&ct=citation&cd=-1&hl=en)__

Ensemble deep learning has been shown to achieve high predictive accuracy and uncertainty estimation in a wide variety of medical imaging contexts. However, perturbations in the input images at test time (e.g., noise, distribution shifts) can still lead to significant performance degradation, posing challenges for trustworthy clinical deployment. To address this challenge, we propose LaDiNE, a novel probabilistic ensemble framework that integrates invariant feature extraction with diffusion models to improve robustness and uncertainty estimation.
![model](./assets/model.png)

## Requirements
The requirements are listed in `requirements.txt`. You can install them using the following command:
```bash
pip install -r requirements.txt
```

## Data
The datasets used in this work are publicly available, however, due to the license policy, we cannot provide the data directly. You can download the datasets from the following links:
- Tuberculosis chest X-ray: [README.md in SEViT](https://github.com/faresmalik/SEViT) or [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- Melanoma skin cancer: [ISIC](https://challenge2020.isic-archive.com/) or [Kaggle](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images/data)

## Training
Before proceeding, please ensure that at least one GPU with CUDA support and a memory capacity exceeding 40 GB is available.

Throughout the process, we will use the environment variable `${DATASET}` to get the dataset name. You can set this by:
```bash
export DATASET={DATASET}
```
where `{DATASET}` is the dataset name (e.g., `ChestXRay` or `ISICSkinCancer`).

Please also keep the data directory handy, as you will need to provide the path to the data directory in the training command and configuration files. We will use another environment variable `${DATA_DIR}` to get the data directory, you can set it by:
```bash
export DATA_DIR={DATA_DIR}
```
where `{DATA_DIR}` is the path to the data directory of the `{DATASET}` you want to use.

In addition, please modify the `data.dataroot` in the configuration files in the `nested-diffusion/diffusion/configs` directory to the path of the data directory `${DATA_DIR}`.

### Training Mapping Networks
We are now in the root directory `nested-diffusion`, please go to the `nested-diffusion/mapping` directory:
```bash
cd mapping
```
1. Train the transformer encoder blocks:
```bash
python train_transformer.py --dataset ${DATASET} --root_dir ${DATA_DIR}
```
Here, `${DATASET}` is the dataset name (e.g., `ChestXRay` or `ISICSkinCancer`). As mentioned previously, you need to provide the path to the data directory `${DATA_DIR}`.

2. Train the mapping networks:
```bash
python train_mapping.py --dataset ${DATASET} --root_dir ${DATA_DIR} --mn_idx {MN_IDX}
```
Here, `${DATASET}` and `${DATA_DIR}` are from the previous step, `{MN_IDX}` is the index of the mapping network (e.g., `0`, `1`, `2`, `3`, `4`). Make sure to run this step for all mapping networks `0 - 4`.

### Training Conditional Diffusion Models
After training the mapping networks, we can train the conditional diffusion models. We are now in the root directory `nested-diffusion`.
1. Run the bash script to move files:
```bash
bash make_files.sh
```
2. Please go to the `nested-diffusion/diffusion` directory and run the bash script to train the conditional diffusion models for the dataset `${DATASET}`: 
```bash
cd diffusion
bash training_scripts/train.sh
```
After the whole training process is finished, you can find the trained models in the `nested-diffusion/diffusion/results` directory.

3. Please look at the configuration files in the `nested-diffusion/diffusion/configs`, you can see that the field `diffusion.trained_diffusion_ckpt_path` is set to the path of the trained diffusion models. You need to modify this field to the path of the trained diffusion models. Here is an example:
```
diffusion:
    trained_diffusion_ckpt_path: [["./results/chest_x_ray/.../diffu0_ckpt_best...pth",
                                   "./results/chest_x_ray/.../diffu1_ckpt_best...pth",
                                   ...,
                                   "./results/chest_x_ray/.../diffu4_ckpt_best...pth"]]
```

## Evaluation
Once we follow all the steps in the training section, we can evaluate the trained models. We are now in the root directory `nested-diffusion`.
1. Please go to the `nested-diffusion/diffusion` directory:
```bash
cd diffusion
```
2. Run the bash script to evaluate the trained models for the dataset `${DATASET}`:
```bash
bash testing_scripts/test.sh
```
3. The above testing script has some customizable parameters, we can modify them in the script before running it. Here is an example:
```
export NOISE_PERTURBATION=0.5
```
then we can test it with the noise perturbation level of 0.5.

## Acknowledgement
This work is founded by the Natural Sciences and Engineering Research Council (NSERC) of Canada, the Canadian Institute for Advanced Research (CIFAR) Artificial Intelligence Chairs program, the Mila - Quebec AI Institute technology transfer program, Calcul Quebec, and the Digital Research Alliance of Canada. This repository contains code adapted from repositories [CARD](https://github.com/XzwHan/CARD) and [SEViT](https://github.com/faresmalik/SEViT). We thank to the above repositories' authors for their great work.

## Citation
If you find this repository useful in your research, please cite our paper:
```
@article{shen2023improving,
  title={Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles},
  author={Shen, Xing and Huang, Hengguan and Nichyporuk, Brennan and Arbel, Tal},
  journal={arXiv preprint arXiv:2310.15952},
  year={2023}
}
```

## References
- Han, X., Zheng, H., & Zhou, M. (2022). Card: Classification and regression diffusion models. Advances in Neural Information Processing Systems, 35, 18100-18115.
- Almalik, F., Yaqub, M., & Nandakumar, K. (2022, September). Self-ensembling vision transformer (sevit) for robust medical image classification. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 376-386). Cham: Springer Nature Switzerland.
