import matplotlib.pyplot as plt

# Epochs
epochs = [1, 2, 3, 4, 5, 6]

# === Données ===

# SGD naïf
loss_naive = [0.226, 0.092, 0.074, 0.061, 0.052, 0.045]
acc_naive = [96.79, 97.84, 97.87, 97.92, 98.01, 98.04]
time_naive = [75.52, 68.25, 62.16, 58.33, 55.74, 53.21]

# Mini-batch SGD
loss_batch = [0.636, 0.328, 0.308, 0.272, 0.241, 0.218]
acc_batch = [90.64, 91.57, 91.89, 92.35, 92.81, 93.24]
time_batch = [45.06, 45.33, 45.56, 45.80, 46.05, 46.34]

# Mini-batch + parallélisme
loss_parallel = [0.636, 0.328, 0.308, 0.273, 0.245, 0.222]
acc_parallel = [90.64, 91.57, 91.88, 92.31, 92.76, 93.18]
time_parallel = [19.05, 18.97, 20.15, 19.85, 19.71, 19.68]

# === 1. Courbe du Loss ===
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_naive, 'o-', label='SGD naïf')
plt.plot(epochs, loss_batch, 's-', label='Mini-batch SGD')
plt.plot(epochs, loss_parallel, '^-', label='Mini-batch + parallèle')
plt.title('Évolution du Loss par epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_vs_epoch.png')

# === 2. Courbe de l'Accuracy ===
plt.figure(figsize=(8, 5))
plt.plot(epochs, acc_naive, 'o-', label='SGD naïf')
plt.plot(epochs, acc_batch, 's-', label='Mini-batch SGD')
plt.plot(epochs, acc_parallel, '^-', label='Mini-batch + parallèle')
plt.title('Évolution de la précision (test accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_epoch.png')

# === 3. Courbe du Temps d'exécution ===
plt.figure(figsize=(8, 5))
plt.plot(epochs, time_naive, 'o-', label='SGD naïf')
plt.plot(epochs, time_batch, 's-', label='Mini-batch SGD')
plt.plot(epochs, time_parallel, '^-', label='Mini-batch + parallèle')
plt.title("Temps d'exécution par epoch")
plt.xlabel('Epoch')
plt.ylabel('Temps (secondes)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('time_vs_epoch.png')
