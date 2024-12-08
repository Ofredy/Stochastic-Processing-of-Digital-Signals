import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import cv2

# Create a directory for saving figures
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

############### Problem 1 ###############

# Function to generate multivariate Gaussian data
def generate_gaussian_samples(mean, cov, n_samples):
    return np.random.multivariate_normal(mean, cov, n_samples)

# Parameters for (a)
mean_a = [0, 0, 0]
cov_a = np.array([[2, 0, 0],
                  [0, 1.5, 0],
                  [0, 0, 2.5]])
samples_a = generate_gaussian_samples(mean_a, cov_a, 1000)

# Parameters for (b)
mean_b = [0, 0, 0]
cov_b = np.array([[2, 0.35, 0.62],
                  [0.35, 0.5, 0.33],
                  [0.62, 0.33, 1]])
samples_b = generate_gaussian_samples(mean_b, cov_b, 1000)

# Scatter plot for (a)
fig_a = plt.figure()
ax_a = fig_a.add_subplot(111, projection='3d')
ax_a.scatter(samples_a[:, 0], samples_a[:, 1], samples_a[:, 2], alpha=0.6, s=10)
ax_a.set_title("Scatter Plot of Gaussian Samples (a)")
ax_a.set_xlabel("X-axis")
ax_a.set_ylabel("Y-axis")
ax_a.set_zlabel("Z-axis")
plt.savefig(os.path.join(output_dir, "scatter_plot_gaussian_samples_a.png"))

# Scatter plot for (b)
fig_b = plt.figure()
ax_b = fig_b.add_subplot(111, projection='3d')
ax_b.scatter(samples_b[:, 0], samples_b[:, 1], samples_b[:, 2], alpha=0.6, s=10)
ax_b.set_title("Scatter Plot of Gaussian Samples (b)")
ax_b.set_xlabel("X-axis")
ax_b.set_ylabel("Y-axis")
ax_b.set_zlabel("Z-axis")
plt.savefig(os.path.join(output_dir, "scatter_plot_gaussian_samples_b.png"))


############### Problem 2 ###############

# Function to project data onto principal components
def project_data(X, d2):
    """
    Projects data X onto d2 principal components.

    Parameters:
        X (numpy.ndarray): Input data matrix of size (n, d).
        d2 (int): Number of dimensions for projection (d2 << d).

    Returns:
        Y (numpy.ndarray): Projected data matrix of size (n, d2).
        T (numpy.ndarray): Transformation matrix of size (d, d2).
    """
    # Transpose X to ensure shape (d, n)
    X = X.T

    # Estimate covariance matrix
    covariance_matrix = np.cov(X)

    # Perform eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the first d2 eigenvectors for transformation matrix
    T = eigenvectors[:, :d2]

    # Project data onto lower-dimensional space
    Y = np.dot(X.T, T)

    return Y, T

# Project data from 3D to 2D for (a)
Y_a, T_a = project_data(samples_a, d2=2)
# Covariance matrix of projected data
cov_Y_a = np.cov(Y_a.T)

# Plot 2D projected data for (a)
plt.figure()
plt.scatter(Y_a[:, 0], Y_a[:, 1], alpha=0.6, s=10)
plt.title("2D Scatter Plot of Projected Data (1a)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig(os.path.join(output_dir, "scatter_plot_projected_data_a.png"))

# Project data from 3D to 2D for (b)
Y_b, T_b = project_data(samples_b, d2=2)
# Covariance matrix of projected data
cov_Y_b = np.cov(Y_b.T)

# Plot 2D projected data for (b)
plt.figure()
plt.scatter(Y_b[:, 0], Y_b[:, 1], alpha=0.6, s=10)
plt.title("2D Scatter Plot of Projected Data (1b)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig(os.path.join(output_dir, "scatter_plot_projected_data_b.png"))
plt.show()

# Output covariance matrices for comparison
print("Covariance matrix of projected data (1a):\n", cov_Y_a)
print("Covariance matrix of projected data (1b):\n", cov_Y_b)


############### Problem 3: Geospatial Image ###############

# Load the geospatial image from MATLAB file
file1_path = 'geospatialImage.mat'
data1 = sio.loadmat(file1_path)
image1 = data1['Geospatial']

# Display the original geospatial image
plt.figure(figsize=(6, 6))
plt.imshow(image1, cmap='gray')
plt.title('Original Geospatial Image')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "original_geospatial_image.png"))

# Perform histogram equalization on the geospatial image
image1_eq = cv2.equalizeHist(image1.astype(np.uint8))

# Display the equalized geospatial image
plt.figure(figsize=(6, 6))
plt.imshow(image1_eq, cmap='gray')
plt.title('Equalized Geospatial Image')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "equalized_geospatial_image.png"))

# Plot histograms for the geospatial image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(image1.ravel(), bins=256, range=(0, 256), density=True, color='gray')
plt.title('Original Histogram (Geospatial)')

plt.subplot(1, 2, 2)
plt.hist(image1_eq.ravel(), bins=256, range=(0, 256), density=True, color='gray')
plt.title('Equalized Histogram (Geospatial)')
plt.savefig(os.path.join(output_dir, "histograms_geospatial.png"))
plt.tight_layout()
plt.show()

############### Problem 3: MRI Image ###############

# Load the MRI image from MATLAB file
file2_path = 'GrayScaleMRI.mat'
data2 = sio.loadmat(file2_path)
image2 = data2['GrayScaleMRI']

# Display the original MRI image
plt.figure(figsize=(6, 6))
plt.imshow(image2, cmap='gray')
plt.title('Original MRI Image')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "original_mri_image.png"))

# Perform histogram equalization on the MRI image
image2_eq = cv2.equalizeHist(image2.astype(np.uint8))

# Display the equalized MRI image
plt.figure(figsize=(6, 6))
plt.imshow(image2_eq, cmap='gray')
plt.title('Equalized MRI Image')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "equalized_mri_image.png"))

# Plot histograms for the MRI image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(image2.ravel(), bins=256, range=(0, 256), density=True, color='gray')
plt.title('Original Histogram (MRI)')

plt.subplot(1, 2, 2)
plt.hist(image2_eq.ravel(), bins=256, range=(0, 256), density=True, color='gray')
plt.title('Equalized Histogram (MRI)')
plt.savefig(os.path.join(output_dir, "histograms_mri.png"))
plt.tight_layout()
plt.show()
