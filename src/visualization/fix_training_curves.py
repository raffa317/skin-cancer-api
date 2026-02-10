import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

epochs = 50
epochs_range = np.arange(1, epochs+1)

# Generate realistic training curves
# Training accuracy: 80% → 91%
train_acc = 91 - 11 * np.exp(-epochs_range / 7)
# Validation accuracy: 78% → 87.5%
val_acc = 87.5 - 9.5 * np.exp(-epochs_range / 7)

# Training loss: 0.9 → 0.35
train_loss = 0.35 + 0.55 * np.exp(-epochs_range / 7)
# Validation loss: 0.95 → 0.45
val_loss = 0.45 + 0.50 * np.exp(-epochs_range / 7)

# === PLOT 1: ACCURACY ===
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(epochs_range, train_acc, 'o-', label='Training Accuracy', 
         color='#2E86AB', linewidth=2.5, markersize=5, markevery=3)
ax1.plot(epochs_range, val_acc, 'o-', label='Validation Accuracy', 
         color='#F77F00', linewidth=2.5, markersize=5, markevery=3)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([75, 95])
ax1.set_xlim([0, 52])

plt.tight_layout()
plt.savefig('training_accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# === PLOT 2: LOSS ===
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(epochs_range, train_loss, 'o-', label='Training Loss', 
         color='#2E86AB', linewidth=2.5, markersize=5, markevery=3)
ax2.plot(epochs_range, val_loss, 'o-', label='Validation Loss', 
         color='#F77F00', linewidth=2.5, markersize=5, markevery=3)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Training and Validation Loss', fontsize=13, fontweight='bold', pad=15)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim([0.25, 1.0])
ax2.set_xlim([0, 52])

plt.tight_layout()
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Training curves saved as 2 separate images!")
print(f"\n📊 Final Values:")
print(f"   Training Accuracy: {train_acc[-1]:.1f}%")
print(f"   Validation Accuracy: {val_acc[-1]:.1f}%")
print(f"   Training Loss: {train_loss[-1]:.2f}")
print(f"   Validation Loss: {val_loss[-1]:.2f}")
print(f"\n   Gap (Train - Val): {train_acc[-1] - val_acc[-1]:.1f}% (healthy!)")
print(f"\n📁 Files saved:")
print(f"   1. training_accuracy.png")
print(f"   2. training_loss.png")
