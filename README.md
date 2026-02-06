# Partially Observable Cognitive Map Learners
Source code for the paper [Cognitive map formation under uncertainty via local prediction learning](https://www.sciencedirect.com/science/article/pii/S2667305325000778).

## File Descriptions
- `data_visualization.ipynb`: used for figure generation in the paper
- `dataset_generator.ipynb`: used to create pickle files of the dataloaders
- `dataloader.py`: functions for dataset generation (synthetic environment)
- `evaluation.py`: evaluation functions
- `gcml-auto.ipynb`: used for running POCML over different hyperparameters in wandb
- `gcml.ipynb`: used to run POCML on a specific set of hyperparameters, print accuracy, visualize learned representations, loss, etc.
- `model.py`: code for POCML model and LSTM/Transformer benchmarks
- `train_benchmark_auto.ipynb`: used for running benchmarks over different hyperparameters in wandb
- `train_benchmark.ipynb`: used for running benchmarks over different hyperparameters in wandb
- `trainer.py`: code for training POCML and benchmarks
- `two_tunnel.ipynb`: code for generating visualizations for the two tunnel maze experiment
- `utils.py`: misc functions
- `visualizer.py`: PCA visualization code
- `zero_shot.ipynb`: code for zero shot generalization experiment