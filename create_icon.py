from PIL import Image
import os

def convert_icon():
    if not os.path.exists('app_icon.png'):
        print("Error: app_icon.png not found")
        return
        
    img = Image.open('app_icon.png')
    # Save as ICO (containing multiple sizes for Windows)
    img.save('icon.ico', format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print("Converted app_icon.png to icon.ico")

if __name__ == "__main__":
    convert_icon()
