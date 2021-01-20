import platform
import xml.etree.ElementTree as ET 

sysstr = platform.system()
if sysstr == 'Windows':
    user_root_path = 'D:'
    print('Call Windows tasks')
elif sysstr == 'Linux':
    user_root_path = '/media/weixing'
    print('Call Linux tasks')
else:
    user_root_path = '/media/weixing'
    print('Call Other platform tasks')

voc_root_path = user_root_path + '/dataset/voc'

#sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')] #('2012', 'train'), ('2012', 'test')
sets = [('2012', 'val')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", 
           "bus", "car", "cat", "chair", "cow", 
           "diningtable", "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def print_info():
    print(sets)
    print(classes)
print_info()

def convert_annotation(year, image_id, list_file):
    try:
        in_file = open(voc_root_path + '/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    except FileNotFoundError:
        print(voc_root_path + '/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
        return

    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print(image_id, cls)
            continue
        difficult = obj.find('difficult').text
        if int(difficult)==1:
            #print(image_id, difficult)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

for year, image_set in sets:
    voc_path = voc_root_path + '/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)
    image_ids = open(voc_path).read().strip().split()
    list_file = open('%s/%s_%s.txt'%(voc_root_path, year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(voc_root_path, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
    print(year, image_set)
