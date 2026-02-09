
# CV Task Workspace Overview

This repository contains multiple tasks and experiments around Colored MNIST, robustness, interpretability, and adversarial attacks. Below is a quick guide to the directory structure and the notebooks.

## Top-level structure

- data/ — Cached MNIST data downloads (raw IDX files). Used by multiple tasks.
- experiments/ — Experimental scripts and outputs for task01 and task2.
- experiments_output/ — Additional experiment outputs.
- relevant_papers/ — Reference materials.
- task0/ — Data generation and dataset exploration.
- task1/ — CNN training on Colored MNIST (lazy model).
- task3/ — Grad-CAM visualizations.
- task4/ — TEA (teacher–student) robust model training.
- task5/ — Adversarial attacks and analysis.

## Task folders and notebooks

### task0

- Notebooks:
 	- [task0/custom-mnist.ipynb](task0/custom-mnist.ipynb): Main notebook used for final dataset generation.
 	- [task0/digit-coloring.ipynb](task0/digit-coloring.ipynb): Creates foreground biased MNISt.
 	- [task0/textured-mnist.ipynb](task0/textured-mnist.ipynb): Explores texture-based background bias.
- Data outputs:
 	- [task0/outputs/colored-mnist/](task0/outputs/colored-mnist/): Generated Colored MNIST dataset (train/test with labels.csv and images).

### task1

- Notebook:
 	- [task1/cnn.ipynb](task1/cnn.ipynb): Trains a 3-layer CNN on Colored MNIST and evaluates bias; includes Grad-CAM utilities for verification and analysis of trained model.
- Saved weights:
 	- [task1/saved_models/](task1/saved_models/): Pretrained CNN checkpoints used as the “lazy” model.

### task3

- Notebook:
 	- [task3/grad-cam.ipynb](task3/grad-cam.ipynb): Grad-CAM visualizations for model explanations.
- Outputs:
 	- [task3/heatmaps/](task3/heatmaps/): Saved heatmaps for train/test samples.

### task4

- Notebook:
 	- [task4/TEA/tea.ipynb](task4/TEA/tea.ipynb): TEA pipeline for teacher–student training and robust model creation.
- Outputs:
 	- [task4/TEA/outputs/](task4/TEA/outputs/): Teacher and student weights.

### task5

- Notebook:
 	- [task5/adversarial.ipynb](task5/adversarial.ipynb): Evaluates clean/FGSM performance and targeted attacks; compares robust vs lazy models and visualizes perturbations.
- Instructions:
 	- [task5/instructions.md](task5/instructions.md)

## Experiments folder

- task01/ — Scripted version of Task 1 training and dataset generation.
 	- [experiments/task01/train.py](experiments/task01/train.py): Training loop for the 3-layer CNN.
 	- [experiments/task01/model.py](experiments/task01/model.py): Model definition.
 	- [experiments/task01/generate_dataset.py](experiments/task01/generate_dataset.py): Dataset generation script.
 	- [experiments/task01/check_bias.py](experiments/task01/check_bias.py): Bias evaluation script.
- task2/ — Optimization and visualization utilities.
 	- [experiments/task2/run_optimization.py](experiments/task2/run_optimization.py)
 	- [experiments/task2/run_advanced_optimization.py](experiments/task2/run_advanced_optimization.py)
 	- [experiments/task2/visualize_kernels.py](experiments/task2/visualize_kernels.py)
 	- [experiments/task2/update_notebook.py](experiments/task2/update_notebook.py)

## Data folder

- [data/MNIST/](data/MNIST/): Raw MNIST IDX files.
- [data/torchvision/MNIST/](data/torchvision/MNIST/): Torchvision MNIST cache.

## Important

- PLEASE read the report first before diving into the notebooks, as it provides crucial context and explanations for the experiments and results. I spent a lot of time writing the report to ensure it is comprehensive and informative, and it will help you understand the notebooks much better. The notebooks are meant to complement the report, not replace it.

## AI Disclosure

- All of my code's core solution idea was prompted by me and AI Generated using copilot. The code for Tasks 0,1,3,4 was reviewed.
- You can find my chat links here [https://chatgpt.com/g/g-p-6970f1f5596081919eaf940aaa185ca2-precog-cv/project]
