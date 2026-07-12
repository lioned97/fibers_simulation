import numpy as np
import matplotlib.pyplot as plt
def check_poisson_lambda(lam=10.0, n_samples=1_000_000, seed=0):
    rng = np.random.default_rng(seed)

    # Draw Poisson samples with parameter lambda
    x = rng.poisson(lam, size=n_samples)

    # Empirical statistics
    sample_mean = x.mean()
    sample_std = x.std(ddof=0)   # population variance
    return np.abs(np.sqrt(sample_mean) / sample_std - 1 ) * 100
    # print("shot-noise:", np.sqrt(sample_mean), "variance:", sample_std)
    # print("ratio:", np.abs(np.sqrt(sample_mean) / sample_std - 1 ) * 100)

a =[]
for i in range(1, 100):
    a.append(check_poisson_lambda(lam=1200, n_samples=i*10))
plt.figure()
plt.plot(np.arange(1, 100)*10, a)
plt.xlabel("Number of samples")
plt.ylabel("Percentage Error")
plt.show()
