
from PIL import Image

from yolo import YOLO

yolo = YOLO()

while True:
    # img = input('Input image filename:')
    img = "VOCdevkit/VOC2007/JPEGImages/xxxx.jpg"
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
        image.save("./input/images-optional/xxx.jpg")
        break
