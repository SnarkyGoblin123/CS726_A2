# Diffusion-and-Guidance

You need to edit just ddpm.py file without changing the function names and signatues of functions outside `__main__`. You can add more options to the argparser as required for your experiments. For example, class labels and guidance scales for part 2.

To setup the environment, follow these steps:

```
conda create --name cs726env python=3.8 -y
conda activate cs726env
```
Install the dependencies
```
pip install -r requirements.txt
```
To install torch, you can follow the steps [here](https://pytorch.org/get-started/locally/). You'll need to know the cuda version on the server. Use `nvitop` command to know the version first. If you have cuda version 12.4, you can just do:

```
pip install torch
```

In case multiple GPUs are present in the system, we recommend using the environment variable `CUDA_VISIBLE_DEVICES` when running your scripts. For example, below command ensures that your script runs on 7th GPU. 

```
CUDA_VISIBLE_DEVICES=7 python ddpm.py --mode train --dataset moons
```

CUDA error messages can often be cryptic and difficult to debug. In such cases, the following command can be quite useful:
```
CUDA_VISIBLE_DEVICE=-1 python ddpm.py --mode train --dataset moons
```
This forces the script to run exclusively on the CPU.


# Evaluation and Training Scripts

## Evaluation Script: `eval_qual.py`

The `eval_qual.py` script is used to evaluate the quality of generated samples from the diffusion model. It calculates metrics such as Negative Log-Likelihood (NLL) and Earth Mover's Distance (EMD) for both training and testing splits. Additionally, it provides visualizations of the generated samples compared to the real dataset.

### Usage
To run the evaluation script, use the following command:
```
python eval_qual.py <samples_path>
```

- `<samples_path>`: Path to the `.pth` file containing the generated samples.

### Key Features
- **NLL Calculation**: Computes the NLL of generated samples with respect to training and testing splits.
- **EMD Calculation**: Calculates the Earth Mover's Distance (EMD) for multiple subsamplings and averages the results.
- **Visualization**: Plots the generated samples against the training and testing datasets for comparison.
- **Results Saving**: Saves evaluation results to a `.pth` file for further analysis.

---

## Diffusion Model Script: `ddpm.py`

The `ddpm.py` script implements the Diffusion Probabilistic Model (DDPM) and its conditional variants. It includes training, sampling, and evaluation functionalities for diffusion-based generative models.

### Key Classes and Functions
- **NoiseScheduler**: Handles the noise scheduling for the diffusion process, supporting linear, cosine, and sigmoid schedules.
- **UNet1D and ConditionalUNet1D**: Implements the U-Net architecture for noise prediction, with support for class conditioning.
- **DDPM and ConditionalDDPM**: Core diffusion models for unconditional and conditional generation.
- **Training Functions**:
  - `train`: Trains the DDPM model.
  - `trainConditional`: Trains the ConditionalDDPM model.
  - `trainCFG`: Trains the ConditionalDDPM model with Classifier-Free Guidance (CFG).
- **Sampling Functions**:
  - `sample`: Generates samples using the DDPM model.
  - `sampleConditional`: Generates class-conditional samples.
  - `sampleCFG`: Generates samples with Classifier-Free Guidance.

### Usage
To train or sample from the model, use the following command:
```
python ddpm.py --mode <mode> --dataset <dataset> [other arguments]
```

#### Modes
- `train`: Train the model.
- `sample`: Generate samples from the trained model.
- `1.2.2`: Evaluate guidance scales and generate decision boundary plots.
- `1.2.3`: Compare classification accuracy using DDPM and MLP classifiers.

#### Example Commands
1. **Training**:
   ```
   python ddpm.py --mode train --dataset moons --n_steps 1000 --lbeta 0.0001 --ubeta 0.02 --epochs 50 --batch_size 64 --lr 0.001 --n_dim 2 --model DDPM
   ```

2. **Sampling**:
   ```
   python ddpm.py --mode sample --dataset moons --n_steps 1000 --n_samples 1000 --n_dim 2 --model DDPM
   ```

3. **Guidance Scale Evaluation**:
   ```
   python ddpm.py --mode 1.2.2 --dataset moons --n_steps 1000 --n_samples 1000 --n_dim 2 --model CFG
   ```

4. **Classification Accuracy Comparison**:
   ```
   python ddpm.py --mode 1.2.3 --dataset moons --n_steps 1000 --n_dim 2 --model conditionalDDPM
   ```

### Additional Notes
- Ensure that the dataset is properly loaded and preprocessed before training or evaluation.
- Modify the `argparser` in `ddpm.py` to include additional arguments as needed for your experiments.
- Use the `run_name` parameter to organize and save model checkpoints and results systematically.

---

## Dependencies
Ensure the following dependencies are installed:
- `torch`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`

Install all dependencies using:
```
pip install -r requirements.txt
```

For more details on the diffusion model and its implementation, refer to the comments in `ddpm.py`.

Refer to [CS726_A1](https://github.com/SnarkyGoblin123/CS726_A1) for Message parsing Algorithms on Undirected Graphical Models



