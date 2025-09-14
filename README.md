# MNIST CNN (<25k Parameters)

This project trains a compact Convolutional Neural Network (CNN) on the **MNIST handwritten digit dataset**.  
The architecture is intentionally small (≈24.2K trainable parameters) but achieves **95%+ test accuracy in just 1 epoch**.

---

## 📦 Architecture

```text
Input:  1 × 28 × 28 (grayscale MNIST image)

Conv2d(1 → 16, 3×3, padding=1)
BatchNorm2d(16)
ReLU

Conv2d(16 → 32, 3×3, padding=1)
BatchNorm2d(32)
ReLU
MaxPool2d(2×2)                → 32 × 14 × 14

Conv2d(32 → 64, 3×3, padding=1)
BatchNorm2d(64)
ReLU

AdaptiveAvgPool2d(1×1)         → 64 × 1 × 1

Flatten
Linear(64 → 10)                → class logits
