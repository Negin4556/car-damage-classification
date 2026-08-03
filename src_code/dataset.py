import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import shutil
from torchvision import transforms
from torchvision import datasets 
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

PHOTO_CAR = "C:/Users/hp/Desktop/python_ai/car_damage/dataset"
SOURCE_DIR = PHOTO_CAR   # ✅ اصلاح شد
PHOTO_CAR_split = "C:/Users/hp/Desktop/python_ai/car_damage/dataset_split"
DEST_DIR = PHOTO_CAR_split   # ✅ اصلاح شد

# تعداد تصاویر هر کلاس
for data in os.listdir(PHOTO_CAR):
    folder_path = os.path.join(PHOTO_CAR, data)

    if os.path.isdir(folder_path):
        images_count = len(os.listdir(folder_path))
        print(f"{data} : {images_count}")

count_folder = []

for data in os.listdir(PHOTO_CAR):

    folder_path = os.path.join(PHOTO_CAR, data)

    if os.path.isdir(folder_path):

        images = os.listdir(folder_path)

        img_path = os.path.join(folder_path, images[0])
        img = Image.open(img_path)

        width, height = img.size
        average_size = (width + height) / 2
        count_folder.append(average_size)

        plt.figure(figsize=(15,3))

        for i in range(5):

            img_path = os.path.join(folder_path, images[i])
            img = Image.open(img_path)

            plt.subplot(1,5,i+1)
            plt.imshow(img)
            plt.title(data)
            plt.axis("off")

        plt.show()

# نسبت‌ها
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# ساخت فولدرهای train, val, test
for split in ["train", "val", "test"]:
    split_path = os.path.join(PHOTO_CAR_split , split)
    os.makedirs(split_path, exist_ok=True)


# رفتن روی هر کلاس
for class_name in os.listdir(SOURCE_DIR):

    class_path = os.path.join(SOURCE_DIR, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)

        # مخلوط کردن تصاویر
        random.shuffle(images)
        total_images = len(images)

        train_end = int(total_images * train_ratio)
        val_end = int(total_images * (train_ratio + val_ratio))

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # ساخت فولدر کلاس داخل هر بخش
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)

        # کپی تصاویر train
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(DEST_DIR, "train", class_name, img)
            shutil.copy(src, dst)

        # کپی تصاویر val
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(DEST_DIR, "val", class_name, img)
            shutil.copy(src, dst)

        # کپی تصاویر test
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(DEST_DIR, "test", class_name, img)
            shutil.copy(src, dst)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root = os.path.join(PHOTO_CAR_split , "train"),
    transform = transform
)

val_dataset = datasets.ImageFolder(
    root = os.path.join(PHOTO_CAR_split , "val"),
    transform = transform
)

test_dataset = datasets.ImageFolder(
    root = os.path.join(PHOTO_CAR_split , "test"),
    transform = transform
)

# batch with DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle = True
)

val_loader = DataLoader(
    val_dataset,
    batch_size = 32,
    shuffle = False
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 32,
    shuffle = False
)

images , labels = next(iter(train_loader))

class CarDamageCNN(nn.Module):
    def __init__(self):
        super(CarDamageCNN, self).__init__()
        
        # لایه کانولوشن اول: ورودی 3 کانال (RGB)، خروجی 32 کانال، فیلتر 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # لایه کاهش ابعاد (MaxPool) با فیلتر 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # لایه کانولوشن دوم: ورودی 32 کانال، خروجی 64 کانال
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # لایه کانولوشن سوم برای استخراج ویژگی‌های پیچیده‌تر
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # لایه‌های کاملاً متصل (Dense / Fully Connected)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        
        # لایه نهایی خروجی به تعداد کلاس‌ها (۴ نوع آسیب خودرو)
        self.fc2 = nn.Linear(512, 4)
        
    def forward(self, x):
        # عبور داده از لایه اول + فعال‌ساز ReLU + لایه پولینگ
        x = self.pool(F.relu(self.conv1(x)))
        # عبور داده از لایه دوم
        x = self.pool(F.relu(self.conv2(x)))
        # عبور داده از لایه سوم
        x = self.pool(F.relu(self.conv3(x)))
        
        # تبدیل ماتریس سه بعدی ویژگی‌ها به یک بردار تک‌بعدی (Flatten)
        x = x.view(-1, 128 * 28 * 28)
        
        # عبور از لایه‌های خطی
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ۱. ساخت یک نمونه از مدل
model = CarDamageCNN()
print(model)

# ۲. گرفتن یک Batch داده از دیتالودر (که از قبل نوشتی)
images, labels = next(iter(train_loader))

# ۳. دادن تصاویر به مدل برای پیش‌بینی اولیه (قبل از آموزش)
outputs = model(images)

# ۴. چاپ ابعاد خروجی مدل
print("Output shape:", outputs.shape)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# مشخص کردن دستگاه محاسباتی (استفاده از GPU در صورت موجود بودن)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# تعریف لیست‌ها برای ذخیره تاریخچه آموزش (مفید برای رسم نمودار)
history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": []
}

best_val_loss = float('inf') # مقدار اولیه بی‌نهایت برای ذخیره بهترین مدل
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) # انتقال داده‌ها به GPU/CPU
        
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

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

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

    # ذخیره در تاریخچه
    history["train_loss"].append(epoch_train_loss)
    history["train_acc"].append(epoch_train_acc)
    history["val_loss"].append(epoch_val_loss)
    history["val_acc"].append(epoch_val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train --> Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
    print(f"Val   --> Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")
    
    # ذخیره بهترین مدل بر اساس کمترین Loss ولیدیشن
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_car_damage_model.pth")
        print("=> Saved best model weights!")
        
    print("-" * 50)

# رسم نمودارهای خطی آموزش و ولیدیشن پس از پایان اپوک‌ها
plt.figure(figsize=(12, 4))

# نمودار Loss
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# نمودار Accuracy
plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.title("Accuracy History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.show()

# ارزیابی مدل نهایی روی تست لودر
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_epoch_loss = test_loss / len(test_loader)
test_epoch_acc = 100 * test_correct / test_total
print(f"Test --> Loss: {test_epoch_loss:.4f} | Acc: {test_epoch_acc:.2f}%")
# for images, labels in train_loader:
# روی batchها حرکت می‌کند
# outputs = model(images)
# تصاویر را به مدل می‌دهد
# loss = criterion(outputs, labels)
# میزان خطا را حساب می‌کند
# optimizer.zero_grad()
# گرادیان قبلی را پاک می‌کند
# loss.backward()
# مشتق‌ها را حساب می‌کند
# optimizer.step()
# وزن‌ها را آپدیت می‌کند
# break
# epoch یعنی چیست؟
# اگر کل دیتاست train یک بار کامل از مدل عبور کند، می‌گوییم:
# یک epoch انجام شده
# loss هر epoch را حساب کنیم
# accuracy هر epoch را حساب کنیم
# # ببینیم مدل دارد یاد می‌گیرد ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

# for epoch in range(num_epochs):
# حلقه اصلی آموزش

# model.train()
# مدل را در حالت آموزش قرار می‌دهد

# running_loss = 0.0
# جمع lossهای batchها

# correct = 0
# تعداد پیش‌بینی درست

# total = 0
# کل نمونه‌هایی که مدل دیده

# outputs = model(images)
# پیش‌بینی مدل

# loss = criterion(outputs, labels)
# محاسبه خطا

# optimizer.zero_grad()
# پاک کردن گرادیان قبلی

# loss.backward()
# محاسبه مشتق‌ها

# optimizer.step()
# آپدیت وزن‌ها

# running_loss += loss.item()
# ذخیره loss هر batch
# torch.max(outputs, 1)
# بیشترین امتیاز هر ردیف را پیدا می‌کند

# یعنی مدل می‌گوید این تصویر متعلق به کدام کلاس است

# epoch_loss
# میانگین loss کل epoch

# epoch_accuracy
# دقت کل epoch

# مرحله 1: گرفتن اسم کلاس‌ها از dataset
class_names = train_dataset.classes
print("Classes:", class_names)

# مرحله 2: transform مخصوص prediction
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# مرحله 3: لود کردن مدل آموزش‌دیده
def load_trained_model(model_path="best_car_damage_model.pth"):
    model = CarDamageCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

trained_model = load_trained_model()
def predict_image(image_path, model, show_image=True):
    # ۱. باز کردن تصویر و تبدیل به ۳ کانال رنگی
    image = Image.open(image_path).convert("RGB")
    
    # ۲. اعمال تبدیل‌ها و آماده‌سازی ابعاد (اضافه کردن بعد Batch با unsqueeze)
    input_tensor = predict_transform(image).unsqueeze(0).to(device)
    
    # ۳. دادن عکس به مدل در حالت بدون محاسبه گرادیان
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)
        
    # ۴. استخراج نام کلاس و درصد اطمینان
    predicted_class = class_names[predicted_index.item()]
    confidence_percent = confidence.item() * 100
    
    # ۵. نمایش تصویر به همراه پیش‌بینی مدل
    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f"Prediction: {predicted_class} ({confidence_percent:.2f}%)")
        plt.axis("off")
        plt.show()
        
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence_percent:.2f}%")
    
    return predicted_class, confidence_percent

# ۵. دیکشنری هزینه‌های تقریبی برای هر آسیب (می‌توانی مبالغ را تغییر دهی)
# ساختار: { Class_Name: (Min_Cost, Max_Cost, Average_Cost) }
repair_costs = {
    'Broken_glass': (100, 400, 250),        # تعویض یا ترمیم شیشه
    'Brokenz_headlight': (80, 350, 180),     # تعویض چراغ
    'Dent': (150, 800, 400),                 # صافکاری و نقاشی قری
    'Scratch': (50, 300, 150)                # لیسه‌گیری و پولیش خط و خش
}

def estimate_repair_cost(predicted_class, confidence):
    """
    بر اساس کلاس تشخیص داده شده و درصد اطمینان مدل، هزینه‌ای تخمینی ارائه می‌دهد.
    """
    if predicted_class in repair_costs:
        min_c, max_c, avg_c = repair_costs[predicted_class]
        
        print("\n==============================================")
        print("           REPAIR COST ESTIMATION             ")
        print("==============================================")
        print(f"Detected Damage: {predicted_class}")
        print(f"Confidence Level: {confidence:.2f}%")
        print(f"Estimated Cost Range: ${min_c} - ${max_c}")
        print(f"Average Expected Cost: ${avg_c}")
        
        # یک پیشنهاد هوشمند بر اساس میزان اطمینان مدل
        if confidence < 50:
            print("Note: Model confidence is low. A physical inspection is highly recommended.")
        else:
            print("Note: Standard estimation. Actual cost may vary based on damage severity.")
        print("==============================================")
    else:
        print(f"Error: No cost rules found for class '{predicted_class}'")
def predict_and_estimate(image_path, model):
    """
    تابع اصلی برای دریافت تصویر، تشخیص نوع آسیب و محاسبه هزینه
    """
    # پیش‌بینی نوع آسیب روی عکس
    predicted_class, confidence = predict_image(image_path, model, show_image=True)
    
    # محاسبه هزینه تعمیرات
    estimate_repair_cost(predicted_class, confidence)

# خطوط تست نهایی پروژه
test_dent_folder = os.path.join(PHOTO_CAR_split, "test", "Dent")
if os.path.exists(test_dent_folder) and len(os.listdir(test_dent_folder)) > 0:
    sample_file = os.listdir(test_dent_folder)[0]
    sample_image_path = os.path.join(test_dent_folder, sample_file)
    
    print("\n--- Running Final Integrated System (Prediction + Cost) ---")
    predict_and_estimate(sample_image_path, trained_model)
else:
    print("\nNo test images found in Dent folder to run final integrated test.")
