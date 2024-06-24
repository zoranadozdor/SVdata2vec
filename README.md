# SVdata2vec


## Overview

This repository includes PyTorch implementation of the paper SVdata2vec: 

![Overview Image](/home/zorana/Documents/SVdata2vec/imgs/overview.png)

## Abstract

Recent advancements in action recognition have leveraged both skeleton and video modalities to achieve state-of-the-art performance. However, these methods often resort to late fusion, resulting in complex designs due to the challenges of early fusion, which tends to underutilize the strengths of each modality. Additionally, self-supervised learning approaches utilizing both modalities remain underexplored.
In this paper, we introduce a novel self-supervised framework for learning from skeleton and video data. Our approach, SV-data2vec, employs a student-teacher architecture, where the teacher network generates contextualized targets based on skeleton data. The student network then performs a masked prediction task using unmasked skeleton-visual data. Remarkably, after pretraining with both modalities, our method allows for fine-tuning with RGB data alone, achieving results on par with multimodal approaches by effectively learning video representations through skeleton data guidance.
Extensive experiments on benchmark datasets NTU RGB+D 60, NTU RGB+D 120, and Toyota Smarthome confirm that our method outperforms existing RGB based state-of-the-art techniques.

## Install the required packages

    ```bash
    pip install -r requirements.txt
    ```

## Prepare datasets

### Download datasets.

#### NTU RGB+D 60 and 120

    1.  Download the videos from the official website (https://rose1.ntu.edu.sg/dataset/actionRecognition/) and put them in /data/nturgbd_raw. Postprocess them with provided script: 

    ```python
    python process_data/compress_video.py
    ```
    
    2. Download the skeleton annotation files from mmaction (https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md)

#### Toyota Smarthome

    1.  Download the dataset from the official website (https://project.inria.fr/toyotasmarthome/).

    2.  Create annotations with script: 

    ```python
    python process_data/smarthome_gendata.py
    ```
    
#### Extract human bounding boxes

    Extract the bounding boxes by 

## Usage

    1. Modify data paths in config
    2. Train using provided bash script



## Acknowledgment

This code is based on [MAMP](https://github.com/maoyunyao/MAMP), [mmaction2](https://github.com/open-mmlab/mmaction2/tree/main) and [data2vec](https://github.com/arxyzan/data2vec-pytorch). 



