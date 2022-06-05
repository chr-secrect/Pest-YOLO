import sys
import os
import glob
import xml.etree.ElementTree as ET
from collections import Counter
'''
！！！！！！！！！！！！！注意事项！！！！！！！！！！！！！
# 这一部分是当xml有无关的类的时候，下方有代码可以进行筛选！
'''
#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

image_ids = open('VOCdevkit/Pest24/ImageSets/Main/val.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_id in image_ids:
    with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
        root = ET.parse("VOCdevkit/Pest24/Annotations/"+image_id+".xml").getroot()
        a = []
        for obj in root.findall('object'):
            difficult_flag = False
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    difficult_flag = True
            obj_name = obj.find('name').text
            a.append(obj_name)
            '''
            ！！！！！！！！！！！！注意事项！！！！！！！！！！！！
            # 这一部分是当xml有无关的类的时候，可以取消下面代码的注释
            # 利用对应的classes.txt来进行筛选！！！！！！！！！！！！
            '''
            # classes_path = 'model_data/voc_classes.txt'
            # class_names = get_classes(classes_path)
            # if obj_name not in class_names:
            #     continue

            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            #######计数
            
            #########
            if difficult_flag:
                new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
            else:
                # new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                count = Counter(a)
        new_f.write("%s,%s\n" % (count,image_id))
                

print("Conversion completed!")
