import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Parameters
num_samples = 10  # Number of samples per experiment
num_experiments = 100000  # Number of experiments for statistical analysis
lambda_param = 1.0  # Rate parameter of the exponential distribution
x_min, x_max = 0, 10  # Range for the histogram

# Choose two different numbers of bins for comparison
num_bins1 = 12  # First number of bins
num_bins2 = 24  # Second number of bins

# Define a function to compute histogram statistics for a given number of bins
def compute_histogram_statistics(num_bins):
    bins = np.linspace(x_min, x_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers
    histograms = np.zeros((num_experiments, num_bins))

    for i in range(num_experiments):
        # Sample from the exponential distribution
        samples = np.random.exponential(1 / lambda_param, size=num_samples)

        # Compute the histogram (density estimate)
        histogram, _ = np.histogram(samples, bins=bins, density=True)
        histograms[i, :] = histogram

    # Compute the mean and variance
    mean_estimator = np.mean(histograms, axis=0)
    variance_estimator = np.var(histograms, axis=0)

    # Compute relative variance (variance / mean^2), avoiding division by zero
    relative_variance = np.divide(
        variance_estimator,
        mean_estimator**2,
        out=np.zeros_like(variance_estimator),
        where=mean_estimator != 0,
    )

    return bins, bin_centers, mean_estimator, variance_estimator, relative_variance

# Compute statistics for both numbers of bins
bins1, centers1, mean1, var1, rel_var1 = compute_histogram_statistics(num_bins1)
bins2, centers2, mean2, var2, rel_var2 = compute_histogram_statistics(num_bins2)

# True PDF for comparison
x = np.linspace(x_min, x_max, 1000)
true_pdf = lambda_param * np.exp(-lambda_param * x)

# Define a function to calculate and display the slope in log-log space
def calculate_slope(bin_centers, values, label):
    log_centers = np.log(bin_centers)  # Log of bin centers
    log_values = np.log(values)  # Log of the estimator values

    # Perform linear regression
    slope, intercept, _, _, _ = linregress(log_centers, log_values)

    print(f"Slope of the {label}: {slope:.2f}")
    return slope

# Plot interpolated results
plt.figure(figsize=(12, 12))

# Subplot 1: Mean estimators with slopes (interpolated)
plt.subplot(3, 1, 1)
plt.plot(centers1, mean1, label=f"Mean (Bins={num_bins1})", color="blue", marker="o")
plt.plot(centers2, mean2, label=f"Mean (Bins={num_bins2})", color="orange", marker="x")
plt.plot(x, true_pdf, label="True PDF", color="green", linestyle="--")

# Calculate and display slopes
calculate_slope(centers1, mean1, f"Mean Estimator (Bins={num_bins1})")
calculate_slope(centers2, mean2, f"Mean Estimator (Bins={num_bins2})")

#plt.yscale("log")  # Logarithmic y-axis
#plt.xscale("log")  # Logarithmic x-axis
plt.xlabel("x")
plt.ylabel("Mean Density (Log-Log)")
plt.title("Mean of Histogram-Based PDF Estimator (Interpolated, Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Subplot 2: Variance estimators with slopes (interpolated)
plt.subplot(3, 1, 2)
plt.plot(centers1, var1, label=f"Variance (Bins={num_bins1})", color="blue", marker="o")
plt.plot(centers2, var2, label=f"Variance (Bins={num_bins2})", color="orange", marker="x")

# Calculate and display slopes
calculate_slope(centers1, var1, "Variance Estimator (Bins=20)")
calculate_slope(centers2, var2, "Variance Estimator (Bins=50)")

#plt.yscale("log")  # Logarithmic y-axis
#plt.xscale("log")  # Logarithmic x-axis
plt.xlabel("x")
plt.ylabel("Variance (Log-Log)")
plt.title("Variance of Histogram-Based PDF Estimator (Interpolated, Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Subplot 3: Relative variance estimators with slopes (interpolated)
plt.subplot(3, 1, 3)
plt.plot(centers1, rel_var1, label=f"Relative Variance (Bins={num_bins1})", color="blue", marker="o")
plt.plot(centers2, rel_var2, label=f"Relative Variance (Bins={num_bins2})", color="orange", marker="x")

# Calculate and display slopes
calculate_slope(centers1, rel_var1, "Relative Variance Estimator (Bins=20)")
calculate_slope(centers2, rel_var2, "Relative Variance Estimator (Bins=50)")

#plt.yscale("log")  # Logarithmic y-axis
#plt.xscale("log")  # Logarithmic x-axis
plt.xlabel("x")
plt.ylabel("Relative Variance (Log-Log)")
plt.title("Relative Variance of Histogram-Based PDF Estimator (Interpolated, Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()

# Plot step results (piecewise constant)
plt.figure(figsize=(12, 12))

# Subplot 1: Mean estimators (piecewise constant)
plt.subplot(3, 1, 1)
plt.step(bins1[:-1], mean1, where="post", label=f"Mean (Bins={num_bins1})", color="blue")
plt.step(bins2[:-1], mean2, where="post", label=f"Mean (Bins={num_bins2})", color="orange")
plt.plot(x, true_pdf, label="True PDF", color="green", linestyle="--")
plt.yscale("log")  # Logarithmic y-axis
plt.xlabel("x")
plt.ylabel("Logarithmic Mean Density")
plt.title("Mean of Histogram-Based PDF Estimator (Piecewise Constant, Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Subplot 2: Variance estimators (piecewise constant)
plt.subplot(3, 1, 2)
plt.step(bins1[:-1], var1, where="post", label=f"Variance (Bins={num_bins1})", color="blue")
plt.step(bins2[:-1], var2, where="post", label=f"Variance (Bins={num_bins2})", color="orange")
plt.yscale("log")  # Logarithmic y-axis
plt.xlabel("x")
plt.ylabel("Logarithmic Variance")
plt.title("Variance of Histogram-Based PDF Estimator (Piecewise Constant, Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Subplot 3: Relative variance estimators (piecewise constant)
plt.subplot(3, 1, 3)
plt.step(bins1[:-1], rel_var1, where="post", label=f"Relative Variance (Bins={num_bins1})", color="blue")
plt.step(bins2[:-1], rel_var2, where="post", label=f"Relative Variance (Bins={num_bins2})", color="orange")
plt.yscale("log")  # Logarithmic y-axis
plt.xlabel("x")
plt.ylabel("Logarithmic Relative Variance")
plt.title("Relative Variance of Histogram-Based PDF Estimator (Piecewise Constant, Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.show()
