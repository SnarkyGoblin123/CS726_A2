from dataset import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import utils
import torch
import sys

# Load the dataset
data_X, data_y = load_dataset('moons')
# Generate labels for data_y matching the size of data_X
if data_y is None or len(data_y) < len(data_X): 
    data_y = np.arange(len(data_X))
# Split the dataset into training and testing sets
train_X, train_y, test_X, test_y = utils.split_data(data_X, data_y, 0.8)

# Load the samples from the .pth file
if len(sys.argv) != 2:
    print("Usage: python sample_quality.py <samples_path>")
    sys.exit(1)

samples_path = sys.argv[1]
samples = torch.load(samples_path)


# Ensure the samples are on the same device as the dataset
device = train_X.device  # Get the device of the dataset
samples = samples.to(device)

if torch.isnan(samples).any():
    print("Warning: NaN detected in generated samples!")
print(samples.shape)
print(train_X.shape)
# Calculate NLL of samples w.r.t train and test splits
print(f"NLL of samples w.r.t train split: {utils.get_nll(train_X, samples):.3f}")
print(f"NLL of samples w.r.t test split: {utils.get_nll(test_X, samples):.3f}")
print(f"NLL of Test Data w.r.t train split: {utils.get_nll(train_X, test_X):.3f}")  # Should be lower than the above two

# Subsample size for EMD calculation
subsample_size = 600

# Calculate EMD for multiple subsamplings and average the results
train_emd_list = []
test_emd_list = []
for i in range(5):
    subsample_test_X = utils.sample(test_X, size=subsample_size)
    subsample_train_X = utils.sample(train_X, size=subsample_size)
    subsample_samples = utils.sample(samples, size=subsample_size)
    
    test_emd = utils.get_emd(subsample_test_X.cpu().numpy(), subsample_samples.cpu().numpy())
    train_emd = utils.get_emd(subsample_train_X.cpu().numpy(), subsample_samples.cpu().numpy())
    
    print(f'{i} EMD w.r.t test split : {test_emd: .3f}')
    print(f'{i} EMD w.r.t train split: {train_emd: .3f}')
    
    train_emd_list.append(train_emd)
    test_emd_list.append(test_emd)

# Print average EMD results
print(f" ---------------------------------")
print(f"Average EMD w.r.t test split : {np.mean(test_emd_list):.3f} ± {np.std(test_emd_list):.3f}")
print(f"Average EMD w.r.t train split: {np.mean(train_emd_list):.3f} ± {np.std(train_emd_list):.3f}")

# Now, use test_X as perfect samples and repeat the EMD calculation
perfect_samples = test_X  # Perfect samples!
train_emd_list = []
test_emd_list = []
for i in range(5):
    subsample_test_X = utils.sample(test_X, size=subsample_size)
    subsample_train_X = utils.sample(train_X, size=subsample_size)
    subsample_perfect_samples = utils.sample(perfect_samples, size=subsample_size)
    
    test_emd = utils.get_emd(subsample_test_X.cpu().numpy(), subsample_perfect_samples.cpu().numpy())
    train_emd = utils.get_emd(subsample_train_X.cpu().numpy(), subsample_perfect_samples.cpu().numpy())
    
    print(f'{i} EMD w.r.t test split : {test_emd: .3f}')
    print(f'{i} EMD w.r.t train split: {train_emd: .3f}')
    
    train_emd_list.append(train_emd)
    test_emd_list.append(test_emd)

# Print average EMD results for perfect samples
print(f" ---------------------------------")
print(f"Average EMD w.r.t test split : {np.mean(test_emd_list):.3f} ± {np.std(test_emd_list):.3f}")
print(f"Average EMD w.r.t train split: {np.mean(train_emd_list):.3f} ± {np.std(train_emd_list):.3f}")

# Save the evaluation results to a .pth file
evaluation_results = {
    'nll_samples_train': utils.get_nll(train_X, samples),
    'nll_samples_test': utils.get_nll(test_X, samples),
    'nll_test_train': utils.get_nll(train_X, test_X),
    'emd_samples_test': np.mean(test_emd_list),
    'emd_samples_train': np.mean(train_emd_list),
    'emd_perfect_samples_test': np.mean(test_emd_list),
    'emd_perfect_samples_train': np.mean(train_emd_list),
}

torch.save(evaluation_results, 'evaluation_results.pth')
print("Evaluation results saved to evaluation_results.pth")

# Optional: Plot the samples and dataset for visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(train_X.cpu()[:, 0], train_X.cpu()[:, 1], label='Train Data', alpha=0.5)
plt.scatter(samples.cpu()[:, 0], samples.cpu()[:, 1], label='Generated Samples', alpha=0.5)
plt.title("Train Data vs Generated Samples")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(test_X.cpu()[:, 0], test_X.cpu()[:, 1], label='Test Data', alpha=0.5)
plt.scatter(samples.cpu()[:, 0], samples.cpu()[:, 1], label='Generated Samples', alpha=0.5)
plt.title("Test Data vs Generated Samples")
plt.legend()

plt.show()