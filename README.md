

car-damage-classification/
│
├── .gitignore
├── requirements.txt
├── README.md
│
├── src/
│   ├── init.py
│   ├── model.py         # حاوی کلاس معماری شبکه (CarDamageCNN)
│   ├── utils.py         # کدهای مربوط به تقسیم‌بندی تصاویر (Dataset Splitter)
│   └── predict.py       # اسکریپت بارگذاری مدل و پیش‌بینی تصویر جدید
│
└── main.py              # اسکریپت اصلی برای بارگذاری داده، آموزش و تست مدل



### بخش دوم: تفکیک و اصلاح کدها

کدهای شما را به بخش‌های استاندارد تقسیم کرده‌ام تا در فایل‌های مربوطه قرار دهید:

#### ۱. فایل src/model.py (تعریف معماری مدل)
این فایل فقط حاوی کلاس مدل شبکه عصبی شماست.

python
# src/model.py
import torch.nn as nn
import torch.nn.functional as F

class CarDamageCNN(nn.Module):
    def init(self):
        super(CarDamageCNN, self).init()
        # لایه کانولوشن اول: ورودی 3 کانال (RGB)، خروجی 32 کانال، فیلتر 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # لایه کانولوشن دوم: ورودی 32 کانال، خروجی 64 کانال
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # لایه کانولوشن سوم
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # لایه‌های کاملاً متصل (Dense)
        # ابعاد ورودی بر اساس عکس ورودی 224x224 و سه لایه Pool متوالی: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 4) # ۴ کلاس آسیب خودرو
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#### ۲. فایل src/predict.py (اسکریپت برای پیش‌بینی روی تصاویر تک)
این کد مدل ذخیره شده را لود کرده و کلاس تصویر ورودی را پیش‌بینی می‌کند.

`python
# src/predict.py
import torch
from torchvision import transforms
from PIL import Image
from model import CarDamageCNN

def predict_image(image_path, model_path="best_car_damage_model.pth", class_names=None):
    if class_names is None:
        class_names = ['scratch', 'dent', 'broken_glass', 'intact'] # کلاس‌های خود را اینجا بنویسید
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # لود کردن مدل
    model = CarDamageCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # آماده‌سازی تصویر
    predict_transform = transforms.Compose([
        transforms.Resize((224, 224)),
transforms.ToTensor()
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = predict_transform(img).unsqueeze(0).to(device) # افزودن بعد Batch
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        
    class_idx = predicted.item()
    return class_names[class_idx]

if name == "main":
    # مثال برای اجرا
    # result = predict_image("path_to_test_image.jpg")
    # print("Predicted Class:", result)
    pass


#### ۳. فایل `main.py` (اسکریپت اصلی آموزش و ارزیابی)
این فایل تمام بخش‌های پروژه را به هم متصل می‌کند. برای اینکه در سرور یا محیط‌های بدون مانیتور (مانند گیت‌هاب اکشنز یا برخی سرورها) اجرای بخش بصری با خطا مواجه نشود، نمایش تصاویر با `plt.show()` را در بخش آموزش غیرفعال یا اختیاری کنید و مسیر دیتابیس را به جای مسیرهای ثابت سیستم خودتان (مانند `C:/Users/...`) به صورت مسیر نسبی آدرس‌دهی کنید.


python
# main.py
import os
import random
import shutil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.model import CarDamageCNN

# تنظیمات مسیرها به صورت محلی در پروژه
SOURCE_DIR = "./dataset"
DEST_DIR = "./dataset_split"

def split_dataset():
    if os.path.exists(DEST_DIR):
        print("Dataset already split.")
        return
        
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)
        
    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)
            total_images = len(images)
            
            train_end = int(total_images * train_ratio)
            val_end = int(total_images * (train_ratio + val_ratio))
            
            train_images = images[:train_end]
            val_images = images[train_end:val_end]
            test_images = images[val_end:]
            
            for split in ["train", "val", "test"]:
                os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)
                
            for img in train_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(DEST_DIR, "train", class_name, img))
            for img in val_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(DEST_DIR, "val", class_name, img))
            for img in test_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(DEST_DIR, "test", class_name, img))
    print("Dataset splitting completed successfully.")

def main():
    split_dataset()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.ImageFolder(root=os.path.join(DEST_DIR, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(DEST_DIR, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(DEST_DIR, "test"), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarDamageCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float('inf')
    num_epochs = 5
    
    print(f"Training on device: {device}")
for epoch in range(num_epochs):
        # چرخه آموزش
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        
        # چرخه ولیدیشن
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_car_damage_model.pth")
            print("=> Saved best model weights!")
            
    # رسم و ذخیره نمودار پیشرفت آموزش
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss History")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy History")
    plt.legend()
    plt.savefig("training_curves.png") # ذخیره نمودار به صورت عکس برای استفاده در گیت‌هاب
    print("Training curves saved as training_curves.png")

    # تست نهایی
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    print(f"Test Accuracy: {100 * test_correct / test_total:.2f}%")

if name == "main":
    main()
`
