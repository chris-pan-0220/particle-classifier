import numpy as np 
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def plot2Dhistogram(samples1, samples2, bins=[7, 7]):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist2d(samples1[:, 0], samples1[:, 1], bins=bins, cmap='viridis')
    plt.colorbar(label='Counts')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Histogram of Random Variable1 ')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist2d(samples2[:, 0], samples2[:, 1], bins=bins, cmap='viridis')
    plt.colorbar(label='Counts')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)

    plt.title('2D Histogram of Random Variable2 ')
    plt.show()

def generateEnergy(n: int):
    cov1 = [[1, 0.5], [0.5, 1]]  
    cov2 = [[1, -0.8], [-0.8, 1]]
    mean = [0, 0]
    bins = [7, 7]
    
    # Generating 5000 histograms for each class (c1 and c2) using a loop
    # Each histogram will be a 7x7 grid, resulting in two arrays of shape (5000, 7, 7)

    h1_stacked = np.zeros((5000, 7, 7)) # Array to store h1 histograms
    h2_stacked = np.zeros((5000, 7, 7)) # Array to store h2 histograms

    for i in range(3):
        # Sampling new data for each iteration
        c1 = np.random.multivariate_normal(mean, cov1, 1000)
        c2 = np.random.multivariate_normal(mean, cov2, 1000)

        # Generating histograms
        h1, _, _ = np.histogram2d(c1[:, 0], c1[:, 1], bins=bins)
        h2, _, _ = np.histogram2d(c2[:, 0], c2[:, 1], bins=bins)

        # Stacking the histograms
        h1_stacked[i] = h1
        h2_stacked[i] = h2
        plot2Dhistogram(c1, c2)
    # h1_stacked and h2_stacked are now arrays of shape (5000, 7, 7)
    h1_stacked.shape, h2_stacked.shape

class ParticleDataGenerator:
    def __init__(self, num_data_points:int, num_scalars:int, energy_matrix_size:int, high_res_size:int, seed=None):
        self.num_data_points = num_data_points
        self.num_scalars = num_scalars
        self.energy_matrix_size = energy_matrix_size
        self.high_res_size = high_res_size
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def generate_data(self, category_mean_diff:float=0, std_dev_ratio:float=1, category_ratio:float=1, debug=False):
        if debug:
            print('params: ')
            print('n data: ', self.num_data_points)
            print('n scalar input: ', self.num_scalars)
            print('n energy matrix: ', self.energy_matrix_size)
            print('n high resolution energy matrix: ', self.high_res_size)
            print('category_mean_diff: ', category_mean_diff)
            print('std_dev_ratio: ', std_dev_ratio)
            print('category_ratio: ', category_ratio)
            print()

        num_data_points_category_1 = int(self.num_data_points * category_ratio)
        num_data_points_category_2 = self.num_data_points - num_data_points_category_1
        if debug:
            print('number of data:')
            print('category 1: ', num_data_points_category_1)
            print('category 2: ', num_data_points_category_2)
            print()

        std_dev_category_1 = [1, 1]
        std_dev_category_2 = [std_dev_category_1[0] * std_dev_ratio, std_dev_category_1[1] * std_dev_ratio]
        if debug:
            print('std:')
            print('category 1: ', std_dev_category_1)
            print('category 2: ', std_dev_category_2)
            print()

        scalar_inputs_category_1 = np.random.normal(0 - category_mean_diff / 2, std_dev_category_1[0], (num_data_points_category_1, self.num_scalars))
        scalar_inputs_category_2 = np.random.normal(0 + category_mean_diff / 2, std_dev_category_2[0], (num_data_points_category_2, self.num_scalars))
        scalar_inputs = np.vstack((scalar_inputs_category_1, scalar_inputs_category_2))
        if debug:
            print('scalar input shape:')
            print('category 1: ', scalar_inputs_category_1.shape)
            print('category 2: ', scalar_inputs_category_2.shape)
            print('category 1 & 2: ', scalar_inputs.shape)
            print()

        # generate energy matrix 
        # generate 2D random variable in high resolution, (num_data_points, high_res * high_res)
        # downsampling to 7*7 energy matrix, (num_data_points, 7, 7)
        cov1 = [[1, 0.5], [0.5, 1]]  ## xx, xy, yx, yy
        cov2 = [[cov1[0][0]*std_dev_ratio, cov1[0][1]*std_dev_ratio], [cov1[1][0]*std_dev_ratio, cov1[1][1]*std_dev_ratio]] 
        if debug:
            print('covariance of x, y: ')
            print('category 1: ', cov1)
            print('category 2: ', cov2)
            print()

        mean = [0, 0]
        mean_category_1 = [mean[0] - category_mean_diff / 2, mean[1] - category_mean_diff / 2]
        mean_category_2 = [mean[0] + category_mean_diff / 2, mean[1] + category_mean_diff / 2]
        if debug:
            print('mean of x, y: ')
            print('category 1: ', mean_category_1)
            print('category 2: ', mean_category_2)
            print()

        bins = [self.energy_matrix_size, self.energy_matrix_size]
        if debug:
            print('mean of x, y: ')
            print('category 1: ', mean_category_1)
            print('category 2: ', mean_category_2)
            print()

        # Generating 5000 histograms for each class (c1 and c2) using a loop
        # Each histogram will be a 7x7 grid, resulting in two arrays of shape (5000, 7, 7)
        h1_stacked = np.zeros((num_data_points_category_1, bins[0], bins[1])) # Array to store h1 histograms
        h2_stacked = np.zeros((num_data_points_category_2, bins[0], bins[1])) # Array to store h2 histograms
        # debug: visualize 2D random variable
        if debug:
            c1_stacked = np.zeros((num_data_points_category_1, self.high_res_size, 2))
            c2_stacked = np.zeros((num_data_points_category_2, self.high_res_size, 2))
        
        for i in range(num_data_points_category_1):
            # Sampling 2D data
            c1 = np.random.multivariate_normal(mean_category_1, cov1, self.high_res_size)
            # debug: visualize 2D random variable
            if debug:
                c1_stacked[i] = c1

            # Generating histograms
            h1, _, _ = np.histogram2d(c1[:, 0], c1[:, 1], bins=bins) # use 2D histogram downsampling to 7*7
            # debug: visualize 2D random variable
            h1_stacked[i] = h1

        for i in range(num_data_points_category_2):
            # Sampling 2D data
            c2 = np.random.multivariate_normal(mean_category_2, cov2, self.high_res_size)
            # debug: visualize 2D random variable
            if debug:
                c2_stacked[i] = c2

            # Generating histograms
            h2, _, _ = np.histogram2d(c2[:, 0], c2[:, 1], bins=bins)
            h2_stacked[i] = h2

        # debug: visualize 2D random variable
        if debug:
            plot2Dhistogram(c1_stacked[0], c2_stacked[0], bins=bins)
            # for i in range(3):
            #     plot2Dhistogram(c1_stacked[i], c2_stacked[i])

        energy_distributions = np.vstack((h1_stacked,  h2_stacked))
        if debug:
            print('energy distribution shape: ', energy_distributions.shape) # (n, 7, 7)
        

        labels_category_1 = np.zeros(num_data_points_category_1)
        labels_category_2 = np.ones(num_data_points_category_2)
        labels = np.concatenate((labels_category_1, labels_category_2))
        # check sample

        # Shuffle the dataset
        permutation = np.random.permutation(self.num_data_points)
        scalar_inputs = scalar_inputs[permutation]
        energy_distributions = energy_distributions[permutation]
        labels = labels[permutation]
        # check shuffling

        # Convert numpy arrays to PyTorch tensors
        scalar_inputs_tensor = torch.tensor(np.array(scalar_inputs), dtype=torch.float32)
        energy_distributions_tensor = torch.tensor(np.array(energy_distributions), dtype=torch.float32)
        labels_tensor = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)

        return scalar_inputs_tensor, energy_distributions_tensor, labels_tensor

class ParticleDataset(Dataset):
    """Particle collision dataset."""

    def __init__(self, scalar_inputs, energy_distributions, labels):
        """
        Initialize the dataset with tensors of scalar inputs, energy distributions, and labels.
        :param scalar_inputs: PyTorch Tensor, tensor of scalar inputs
        :param energy_distributions: PyTorch Tensor, tensor of energy distributions
        :param labels: PyTorch Tensor, tensor of labels
        """
        self.scalar_inputs = scalar_inputs
        self.energy_distributions = energy_distributions
        self.labels = labels

    def __len__(self):
        """Return the total number of data points."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve the scalar input, energy distribution, and label for the given index.
        :param idx: int, index of the data point
        :return: tuple, (scalar_input, energy_distribution, label)
        """
        scalar_input = self.scalar_inputs[idx]
        energy_distribution = self.energy_distributions[idx]
        label = self.labels[idx]
        return scalar_input, energy_distribution, label

def test():
    # Use the ParticleDataGenerator to generate data
    n = 1000
    n_scalar = 3
    energy_matrix_size = 15
    high_res_size = 500 # energy high resolution size
    generator = ParticleDataGenerator(num_data_points=n, num_scalars=n_scalar, energy_matrix_size=energy_matrix_size, high_res_size=high_res_size)

    # Generate the data
    category_mean_diff = 1 # difference of mean
    std_dev_ratio = 2 # ratio of std category1 and std category2
    category_ratio = 0.5 # category ratio
    scalar_inputs, energy_distributions, labels = generator.generate_data(category_mean_diff, std_dev_ratio, category_ratio)

    # Set up the figure for 3D plotting
    fig = plt.figure(figsize=(18, 6))

    # 3D plot for the scalar inputs
    ax1 = fig.add_subplot(131, projection='3d')

    # Plot scalar inputs for both categories
    category1_mask = labels.numpy().reshape(-1) == 0
    category2_mask = labels.numpy().reshape(-1) == 1

    ax1.scatter(scalar_inputs[category1_mask, 0], scalar_inputs[category1_mask, 1], scalar_inputs[category1_mask, 2], c='r', label='Category 1')
    ax1.scatter(scalar_inputs[category2_mask, 0], scalar_inputs[category2_mask, 1], scalar_inputs[category2_mask, 2], c='b', label='Category 2')

    # Set labels and title
    ax1.set_xlabel('Scalar Input 1')
    ax1.set_ylabel('Scalar Input 2')
    ax1.set_zlabel('Scalar Input 3')
    ax1.set_title('3D Scatter Plot of Scalar Inputs')
    ax1.legend()

    # 3D plot for the energy matrix of an example from category 1
    ax2 = fig.add_subplot(132, projection='3d')
    example_category_1 = energy_distributions[category1_mask][0]
    x, y = np.meshgrid(np.arange(energy_matrix_size), np.arange(energy_matrix_size))
    z = example_category_1.numpy()
    ax2.plot_surface(x, y, z, cmap='viridis')
    ax2.set_title('Energy Distribution of Category 1')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Energy')

    # 3D plot for the energy matrix of an example from category 2
    ax3 = fig.add_subplot(133, projection='3d')
    example_category_2 = energy_distributions[category2_mask][0]
    z = example_category_2.numpy()
    ax3.plot_surface(x, y, z, cmap='plasma')
    ax3.set_title('Energy Distribution of Category 2')
    ax3.set_xlabel('X-axis')
    ax3.set_ylabel('Y-axis')
    ax3.set_zlabel('Energy')

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # # Instantiate the generator with a specified mean difference, standard deviation ratio, and category ratio
    # category_mean_diff = 0.25
    # std_dev_ratio = 2
    # category_ratio = 0.3
    # generator = ParticleDataGenerator(num_data_points=5000, num_scalars=3, energy_matrix_size=7, high_res_size=500, seed=42)

    # # Generate the data
    # scalar_inputs, energy_distributions, labels = generator.generate_data(category_mean_diff, std_dev_ratio, category_ratio)

    # # Create the dataset
    # particle_dataset = ParticleDataset(scalar_inputs, energy_distributions, labels)

    # # Create a DataLoader instance
    # batch_size = 64
    # particle_dataloader = DataLoader(particle_dataset, batch_size=batch_size, shuffle=True)

    # # We can iterate over the DataLoader
    # # Here's how we can get the first batch of data
    # first_batch = next(iter(particle_dataloader))

    # # Output the shapes of the data in the first batch
    # print(first_batch[0].shape, first_batch[1].shape, first_batch[2].shape)
    test()
