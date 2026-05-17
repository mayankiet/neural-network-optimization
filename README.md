# neural-network-optimization

FashionMNIST Neural Network Classifier

A deep learning project built using PyTorch for classifying clothing images from the FashionMNIST dataset.

This project demonstrates:

Dataset loading
Data preprocessing
Neural network creation
Model training
Evaluation using classification metrics
📚 Dataset

The project uses the FashionMNIST dataset from TorchVision.

FashionMNIST contains grayscale images of fashion products belonging to 10 classes.

Classes
Label	Class
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
🛠️ Tech Stack
Python 3.11
PyTorch
TorchVision
NumPy
Matplotlib
Scikit-learn
📦 Installation
Clone Repository
git clone https://github.com/mayankiet/neural-network-optimization.git

cd neural-network-optimization
Create Virtual Environment
Mac/Linux
python3.11 -m venv .venv

source .venv/bin/activate
Windows
python -m venv .venv

.venv\Scripts\activate
Install Dependencies
pip install torch torchvision matplotlib scikit-learn numpy
🚀 Project Workflow
Load FashionMNIST dataset
Create DataLoader
Visualize sample images
Build neural network
Train model
Evaluate model
Generate classification report
📥 Load Dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
📊 DataLoader
batch_size = 64

train_loader = DataLoader(training_data, batch_size=batch_size)

test_loader = DataLoader(test_data, batch_size=batch_size)
🖼️ Example Batch Shape
torch.Size([64, 1, 28, 28])
torch.Size([64])
Meaning
Shape	Description
[64, 1, 28, 28]	64 grayscale images
[64]	64 labels
🧠 Neural Network Architecture
class ClothsClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(28*28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)
🏗️ Architecture Explanation
Input Image (1x28x28)

↓ Flatten

784 Features

↓ Linear Layer (784 → 128)

↓ ReLU

↓ Linear Layer (128 → 64)

↓ ReLU

↓ Linear Layer (64 → 10)

↓ Output Classes
⚡ Device Configuration
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

Supports:

NVIDIA GPU (CUDA)
Apple Silicon GPU (MPS)
CPU fallback
🎯 Loss Function
loss_fn = nn.CrossEntropyLoss()

Used for multi-class classification.

🚀 Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

Adam optimizer helps update weights efficiently during training.

🏋️ Training Loop
epochs = 2

model.train()

for epoch in range(epochs):

    for batch, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)

        loss = loss_fn(pred, labels)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"Batch: {batch}, Loss: {loss.item()}")
📉 Example Training Output
Batch: 0, Loss: 2.29
Batch: 100, Loss: 0.76
Batch: 200, Loss: 0.45

Loss decreases during training, meaning the model is learning.

🧪 Model Evaluation
model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
📋 Classification Report
from sklearn.metrics import classification_report

report = classification_report(all_labels, all_predicted)

print(report)

Provides:

Precision
Recall
F1-score
Accuracy
📈 Sample Output
              precision    recall  f1-score   support

     T-shirt       0.85      0.82      0.83
     Trouser       0.98      0.97      0.98
     Pullover      0.79      0.75      0.77
🔥 Key Deep Learning Concepts Used
Artificial Neural Networks (ANN)
Forward Propagation
Backpropagation
Gradient Descent
Activation Functions
Batch Processing
Optimization Algorithms
Model Evaluation
📌 Future Improvements
Add CNN architecture
Use dropout regularization
Add learning rate scheduler
TensorBoard integration
Hyperparameter tuning
GPU optimization
