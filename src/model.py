import torch.nn as nn
import torch.nn.functional as F

class CarDamageCNN(nn.Module):
    def __init__(self):
        super(CarDamageCNN, self).__init__()
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

if __name__ == "__main__":
    # مثال برای اجرا
    # result = predict_image("path_to_test_image.jpg")
    # print("Predicted Class:", result)
    pass
