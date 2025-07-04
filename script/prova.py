import numpy as np
import matplotlib.pyplot as plt

# Genera una contact map binaria di esempio
contact_map = np.random.choice([0, 1], size=(1500, 1500))

# Aumenta il contrasto
plt.imshow(contact_map, cmap='binary', interpolation='none', vmin=0, vmax=1)

# Sovrapposizione dei contorni
plt.contour(contact_map, colors='red', levels=[0.5], linewidths=0.1)

# Mostra la contact map
plt.show()

