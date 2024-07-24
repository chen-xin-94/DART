# DART
An automated end-to-end object detection pipeline with data **D**iversification, open-vocabulary bounding box **A**nnotation, pseudo-label **R**eview, and model **T**raining

[arXiv](https://arxiv.org/abs/2407.09174) | [dataset](https://syncandshare.lrz.de/getlink/fi9HcSsruiQLQHV4LK8Tpa/Liebherr_Product.zip)

## Overview

This repository contains the implementation of **DART**, an automated end-to-end object detection pipeline featuring:

- Data **D**iversification based on DreamBooth with Stable Diffusion XL
- Open-vocabulary bounding box **A**nnotation via GroundingDINO
- LMM-based **R**eview of pseudo-labels and image photorealism using InternVL-1.5 and GPT-4o
- Real-time object detector **T**raining for YOLOv8 and YOLOv10

The current instantiation of DART significantly increases the average precision (AP) from 0.064 to 0.832 for a YOLOv8n model on the [Liebherr Product dataset](#liebherr-product-dataset), demonstrating the effectiveness of our approach.

![DART](/figures/DART_flowchart.svg)

## Liebherr Product Dataset
This repository contains a self-collected dataset of construction machines named Liebherr Product (LP), which contains over 15K high-quality images across 23 categories. This extensive collection focuses on a diverse range of construction machinery from Liebherr products, including articulated dump trucks, bulldozers, combined piling and drilling rigs, various types of cranes, excavators, loaders, and more. A list of all 23 classes can be found in [classes.json](/Liebherr_Product/metadata/classes.json). For detailed information on the data collection, curation, and preprocessing of this dataset, please check out [our paper](https://arxiv.org/abs/2407.09174). The images can be downloaded and processed by following the instructions in [this section](#data-preparation).

## Repository Structure

This repository contains the following folders and files, each serving a specific purpose:

### `./diversification`
contains the code for training and inference of SDXL with `dreambooth`, as well as generated `class_data` and collected `instance_data`.

### `./figures`
contains figures used in the repo.

### `./Liebherr_Product`
the dataset folder. `images` should be downloaded separately (following instructions in [this section](#data-preparation)). This folder also includes lists and statistics of pseudo `labels`, `metadata` containing useful information extracted during dasets preprocessing, responses from GPT-4-based `reviews`, `questionnaire` used for evaluating GPT-4's performance, and general `tools` for facilitating interaction with the dataset.

### `./lmm`
contains code for two LMM-based review: GPT-4o-based pseudo-label review and image photorealism for generated data via InternVL-1-5. 

### `./ovd`
contains code for bounding box generation with Grounidng DINO and label processing.

### `./vis`
contains figures used in the paper and their corresponding code.

### `./yolo`
contains code and commands for data split, hyperparameter fine-tuning, training and prediction with yolov8.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/dart.git
    ```

2. Create an Anaconda environment, e.g. named "dart":
    ```bash
    conda create -n dart python=3.10
    conda activate dart
    ```

3. Follow [this link](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) to install Grounding DINO.

4. Install other required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data preparation

1. Download the dataset via [this link](https://syncandshare.lrz.de/getlink/fi9HcSsruiQLQHV4LK8Tpa/Liebherr_Product.zip), and extract the `images` folder to `./Liebherr_Product/images/`.
2. Collect instance data and store them in `./diversification/instance_data/{class_name}/{instance_name}`, e.g. `./diversification/instance_data/articulated_dump_truck/TA230`.
3. Change the default paths in the following scripts or specify as arguments while running.


### Annotation and review for collected data

1. Annotate collected data with "orignal" and "co-occurring" prompt:
    ```bash
    python ovd/labeling.py -p one
    ```

2.  Annotate collected data with "synonym" and "co-occurring" prompt:
    ```bash
    python ovd/labeling_sep.py -p one
    ```

3. Process labels:
    ```bash
    python ovd/label_processing.py
    ```

4. Identify annotations that need to be processed by GPT-4o:
    ```bash
    jupyter notebook Liebherr_Product/tools/check_anns.ipynb
    ```

5. Review pseudo-labels with GPT-4o:
    ```bash
    python lmm/gpt4.py
    ```

6. Parse GPT-4o's responses:
    ```bash
    jupyter notebook parse_gpt4_response.ipynb
    ```

7. Convert annotations to YOLO format:
    ```bash
    jupyter notebook Liebherr_Product/tools/convert_to_yolo.ipynb
    ```

8. Split data into train/val/test sets:
    ```bash
    jupyter notebook yolo/data_split.ipynb
    ```

### Annotation and review for generated diversified data

1. Generate scripts for DreamBooth training of each instance:
    ```bash
    jupyter notebook diversification/dreambooth/sdxl.ipynb
    ```

2. Run DreamBooth training scripts in bulk:
    ```bash
    python diversification/dreambooth/run_command_bulk.py
    ```

3. Generate data using the trained DreamBooth model in bulk:
    ```bash
    python diversification/dreambooth/data_generation_bulk.py
    ```

4. (Optionally) Generate data using the trained DreamBooth model for specific scenarios:
    ```bash
    python diversification/dreambooth/data_generation_obj_partial_prompts.py
    ```

5. Convert images and create ID to name mapping:
    ```bash
    jupyter notebook diversification/dreambooth/id_to_name.ipynb
    ```

6. Annotate generated data:
    ```bash
    python ovd/labeling_gen.py
    ```

7. Review generated data with InternVL-Chat-V1-5:
    ```bash
    python lmm/InternVL-Chat-V1-5_judge.py
    ```

8. Parse the responses:
    ```bash
    jupyter notebook lmm/parse_lmm_response.ipynb
    ```

9. Process labels for generated data:
    ```bash
    python ovd/label_processing_gen.py
    ```

10. (Optionally) Plot annotations:
    ```bash
    jupyter notebook ovd/annotate_gen.ipynb
    ```

11. Process labels for manually diversified data in the original dataset:
    ```bash
    python label_processing.py --label_dir labels_background --id_types b
    ```

12. Merge labels and stats of generate and original data:
    ```bash
    jupyter notebook Liebherr_Product/tools/merge_labels_stats_dict.ipynb
    ```

13. Convert annotations to YOLO format:
    ```bash
    jupyter notebook Liebherr_Product/tools/convert_to_yolo_gen.ipynb
    ```

14. Split all data into train/val/test sets:
    ```bash
    jupyter notebook yolo/data_split_gen.ipynb
    ```

### Training and fine-tuning

1. Create dataset configs according to experiments:
    ```bash
    # Example: cfg/datasets/train.yaml
    # Example: cfg/datasets/fine-tune.yaml
    ```

2.  Fine-tune hyperparameters:
    ```bash
    python yolo/raytune.py --cfg fine-tune.yaml
    ```
    or 
    ```bash
    python yolo/tune.py --cfg fine-tune.yaml
    ```

3. Train and evaluate the model the with the best hyperparameter set:
    ```bash
    yolo detect train data=cfg/datasets/train_gen_0.75.yaml model=yolov8n.pt epochs=60 imgsz=640 optimizer=AdamW lr0=2e-4 lrf=0.5 warmup_epochs=2 batch=64 cos_lr=True
    ```

### Inference
1. predict based on trained models
    ```bash
    jupyter notebook yolo/predict.ipynb
    ```

## Results

Here are some sample results. Please check out our read our paper for more!
### Object detection results with and without DART on test set images.
![with_or_without_DART_1](/figures/predictions_1.jpg)
![with_or_without_DART_2](/figures/predictions_2.jpg)
### Visualization of data diversification and bounding box annotation
![approved_annotated_generated_images_app](/figures/approved_annotated_generated_images_app.jpg)
### Images annotated by Grounding DINO and approved by GPT-4o
![image_grid_orig_aa](/figures/image_grid_orig_aa.jpg)
## Citation
```
@misc{xinDART2024,
      title={DART: An Automated End-to-End Object Detection Pipeline with Data Diversification, Open-Vocabulary Bounding Box Annotation, Pseudo-Label Review, and Model Training}, 
      author={Chen Xin and Andreas Hartel and Enkelejda Kasneci},
      year={2024},
      eprint={2407.09174},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.09174}, 
}
```