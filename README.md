# LatentG-Loss
Official Implementation of the LatentG Loss Paper


## Overview
This study presents a multi-stage approach to mental health classification by leveraging traditional machine learning algorithms, deep learning architectures, and transformer-based models. A novel data set was curated and utilized to evaluate the performance of various methods, starting with conventional classifiers and advancing through neural networks. To broaden the architectural scope, recurrent neural networks (RNNs) such as LSTM and GRU were also evaluated to explore their effectiveness in modeling sequential patterns in the data. Subsequently, transformer models such as BERT were fine-tuned to assess the impact of contextual embeddings in this domain. Beyond these baseline evaluations, the core contribution of this study lies in a novel training strategy involving a dual-model architecture composed of a teacher and a student network. Unlike standard distillation techniques, this method does not rely on soft label transfer; instead, it facilitates information flow through both the teacher model’s output and its latent representations by modifying the loss function. The experimental results highlight the effectiveness of each modeling stage and demonstrate that the proposed loss function and teacher-student interaction significantly enhance the model's learning capacity in mental health prediction tasks.

Model Chekcpoints, Datasets, Embeddings and Other util files can be found at : `https://drive.google.com/drive/folders/1SPyqW8wXPgBeK1i6CRVG9iV7sRSw_Lio?usp=sharing`

## Project Structure

This repository includes:

- 📜 `train_textcnn_w_latentG_Loss.py` — Proposed Model architecture training script
- 📜 `train_*.py, finetune_*.py, test_*.py, sentiment_analysis.ipynb` — Training finetuning and evaluation scripts with proposed hyperparameters. 
- 📜 `DualArchitecture.py, RNNBasedModel.py, TextAutoEncoder.py, TextCNN.py, TextDataset.py, fit_mog_2_latent_descriptor.ipynb, fit_mog_2_latent_descriptor_dualarchitecture.ipynb, latentG_Loss.ipynb` — Configs, Utils and Architecture implementations 
- 📜 `tversky_loss.py , dice_loss.py` — Implementation of different loss functions  
- 📁 `drive folder` — Generated results, models, vectors, etc.

##  Getting Started

### Prerequisties

- Python 3.8+
- PyTorch >= 1.10
- transformers == 4.47.1
- bitsandbytes == 0.44.1
- huggingface-hub >= 0.26
- numpy >= 1.26.3
- pandas >= 1.5.2
- scikit-learn >= 1.6.1
- scipy >= 1.13.1
- Other environment packages are listed in `environment_packages.txt` you can check the versions if you encounter any version problems.


### Installation
```bash
git clone https://github.com/yourusername/LatentG-Loss.git
cd LatentG-Loss
```

### Running Experiments

To train, test a model with Latent-G Loss, you can use the train and test scripts.
Also the weights of the trained model can be found in drive link.

## 📈 Results

The DualLatentGNet architecture and LatentGLoss presented in this work has demonstrated impressive results in text classification, outperforming traditional machine learning models and competing with leading transformer-based approaches. By focusing on efficiency, scalability, and feature extraction, our method provides an alternative to the highly resource-intensive transformer models, proving that performance and computational efficiency are not mutually exclusive.

*Quantitative and qualitative results are provided in the paper.*

## 📄 Citation

If you use this work in your research, or find it useful, please cite it as follows:

```
Korhan Sevinc, LatentGLoss, GitHub repository, 2025.
Available at https://github.com/korhansevinc/LatentG-Loss
```