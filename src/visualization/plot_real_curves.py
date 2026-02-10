"""
Plot Real Training Curves from Saved History
Run this AFTER training_with_logging.py completes
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load training history
with open('training_history.json', 'r') as f:
    history = json.load(f)

epochs = history['epochs']
train_acc = history['train_acc']
val_acc = history['val_acc']
train_loss = history['train_loss']
val_loss = history['val_loss']

print("="*60)
print("PLOTTING REAL TRAINING CURVES")
print("="*60)
print(f"Epochs trained: {len(epochs)}")
print(f"Final Training Acc: {train_acc[-1]:.2f}%")
print(f"Final Validation Acc: {val_acc[-1]:.2f}%")
print(f"Final Training Loss: {train_loss[-1]:.4f}")
print(f"Final Validation Loss: {val_loss[-1]:.4f}")
print("="*60)

# === PLOT 1: ACCURACY ===
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(epochs, train_acc, 'o-', label='Training Accuracy', 
         color='#2E86AB', linewidth=2.5, markersize=5, markevery=max(1, len(epochs)//15))
ax1.plot(epochs, val_acc, 'o-', label='Validation Accuracy', 
         color='#F77F00', linewidth=2.5, markersize=5, markevery=max(1, len(epochs)//15))
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Training and Validation Accuracy (Real Data)', fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('training_accuracy_REAL.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# === PLOT 2: LOSS ===
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(epochs, train_loss, 'o-', label='Training Loss', 
         color='#2E86AB', linewidth=2.5, markersize=5, markevery=max(1, len(epochs)//15))
ax2.plot(epochs, val_loss, 'o-', label='Validation Loss', 
         color='#F77F00', linewidth=2.5, markersize=5, markevery=max(1, len(epochs)//15))
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Training and Validation Loss (Real Data)', fontsize=13, fontweight='bold', pad=15)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('training_loss_REAL.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✅ Real training curves saved!")
print(f"   1. training_accuracy_REAL.png")
print(f"   2. training_loss_REAL.png")
print("\nThese are 100% REAL curves from actual training! 🎯")
