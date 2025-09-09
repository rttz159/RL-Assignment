# RL-Assignment

This repository contains files and scripts for a reinforcement learning project focused on glucose dynamics modeling and simulation. Below is the file structure with comments describing each item:

## File Structure & Comments

- `.gitignore`  
  Specifies files and directories to be ignored by Git for cleaner version control.

- `CustomGlucoseDynamicsEnv.py`  
  Implements a custom environment for simulating glucose dynamics, likely used for RL training.

- `DigitalTwinModel.py`  
  Contains code for a digital twin model—possibly a virtual representation of the glucose system.

- `GlucoseDynamicsSimulator.py`  
  Provides the simulation logic for glucose dynamics, used for environment or model validation.

- `HybridLookaheadPolicyWrapper.py`  
  Implements a wrapper for hybrid lookahead policy logic, to improve RL agent performance.

- `app.py`  
  Main application entry point—may run experiments or launch an interface.

- `further_enhancement.ipynb`  
  Jupyter notebook for further enhancements, likely containing experimental code or analysis.

- `ppo_best_glucose_model.zip`  
  Zipped file of the best RL model (PPO algorithm) trained on the glucose environment.

- `requirements.txt`  
  Lists Python package dependencies required for running the project.

- `training.ipynb`  
  Jupyter notebook that developing RL agent training procedures and show results.

- `world_model.ipynb`  
  Jupyter notebook for developing further robustness and safety enhancements

- `figures_comparison/`  
  Directory containing figures to compare results, likely for visualization of experiments.

- `figures_comparison_finetuned/`  
  Directory for figures from finetuned models, useful for result comparison.

- `logs/`  
  Folder for output logs generated during training or evaluation.

- `results/`  
  Stores results from experiments, such as evaluation metrics or output files.

- `saved_models/`  
  Directory for saving trained model files for later use or analysis.
