
import albumentations as A
import cv2
class Baseline:
    def __init__(self,):
        self.transform = A.Compose([
            A.Rotate(limit=20),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.HorizontalFlip(),
        ])

    def distort(self,img):
        img = self.transform(image=img)['image']
        return img