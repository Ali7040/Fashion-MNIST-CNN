# 👕 Fashion-MNIST CNN Classifier with PyTorch

A deep learning project built on **PyTorch** using **Convolutional Neural Networks (CNNs)** to classify fashion images from the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). This notebook was designed, trained, and evaluated using **Kaggle Notebooks**.

<p align="center">
  <img width="400" alt="Training Metrics" src="https://github.com/user-attachments/assets/4187f6f0-51c7-48aa-8e0c-5292aac90d36" />
  <img width="389" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/febaa3c7-847d-4785-b051-13cb21557973" />
  <img width="387" alt="Predictions" src="https://github.com/user-attachments/assets/88dad381-013c-45e2-89a4-eca7d07c68c6" />
</p>

---

## 📌 Key Highlights

✅ Built a custom CNN architecture using `nn.Module`  
✅ Trained using PyTorch's `DataLoader` and `CrossEntropyLoss`  
✅ Visualized performance with matplotlib (Loss, Accuracy, Predictions)  
✅ Evaluated with a confusion matrix and prediction analysis  

---

## 📂 Dataset Overview

Fashion-MNIST consists of 28x28 grayscale images across **10 categories**:

- T-shirt/top, Trouser, Pullover, Dress, Coat  
- Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## 🧠 Model Architecture

The CNN includes:

- **Convolution Layers** with ReLU and MaxPooling
- **Fully Connected (Dense) Layers**
- **Softmax Output** with 10 units for classification

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(32 * 7 * 7, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
```

---

## 🏋️‍♂️ Training Setup

- Optimizer: `Adam`
- Loss Function: `CrossEntropyLoss`
- Batch Size: `32`
- Device: `GPU` (if available)

```python
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Training loop with live loss/accuracy tracking and validation.

---

## 📊 Visualizations

- 📈 Training Loss & Accuracy Curve  
- 📉 Confusion Matrix  
- 👕 Sample Predictions with True Labels  

These help in understanding both overfitting and class-wise performance.

---

## ✅ Evaluation

- Accuracy: **~90.81%%**
- Most confused classes: [Shirt ↔ T-shirt], [Sneaker ↔ Ankle boot]
- Generalized well on unseen data

---

## 📎 How to Use

> You can run this notebook directly in Kaggle:

1. Upload `fashion-mnist-cnn.ipynb` to your Kaggle account
2. Ensure GPU is enabled (`Settings > Accelerator`)
3. Run all cells to train and evaluate the model

---

## 📁 Project Structure

```
fashion-mnist-cnn/
│
├── fashion-mnist-cnn.ipynb       # Notebook with model, training, evaluation
├── README.md                     # Project description
```

---

## 📚 Dependencies

```bash
pip install torch torchvision matplotlib seaborn
```

---

## 🙋‍♂️ Author

**Ali Haider**  
Final Year CS Student | Frontend Developer → ML Explorer  
[GitHub](https://github.com/Ali7040) • [LinkedIn](https://www.linkedin.com/in/ali-haider7/)

---

## 🌟 Show Your Support

If you like this project, consider giving it a ⭐ on GitHub or sharing it with others!
