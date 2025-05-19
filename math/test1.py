import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 1 # 均值和标准差
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2)), label='Gaussian Distribution')

plt.title('Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()