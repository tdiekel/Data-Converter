import os
from types import SimpleNamespace
import os
import sys
import time
import xml.etree.ElementTree as ET
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from util.util import *


def get_subfolder_name(class_id, args):
    return str(class_id) + '_' + args.id2cat[class_id].replace(' ', '_')


def create_subfolders(args):
    """ Create a subfolder for each class in args.include
    """

    for class_id in args.include:
        subfolder = get_subfolder_name(class_id, args)
        subfolder_path = os.path.join(args.output_path, subfolder)

        create_dir(subfolder_path)


def write_cuts_to_subfolder(objects, args):
    """ Writes cut out objects to disk in a subfolder per class
    :param objects:
    :param args:
    :return:
    """

    for i, class_id in enumerate(objects['class_ids']):
        subfolder = get_subfolder_name(class_id, args)
        subfolder_path = os.path.join(args.output_path, subfolder)

        filename, ext = os.path.splitext(objects['filename'])
        filename += '_' + str(i) + ext

        file_path = os.path.join(subfolder_path, filename)

        cv2.imwrite(file_path, objects['cuts'][i])


def cut_objects(objects, args):
    """ Cuts out the labeled object from the image.
    :param objects: Dict with filename, class_ids and coords
    :return: Object as nd array
    """

    image_path = os.path.join(args.image_path, objects['filename'])

    image = cv2.imread(image_path)
    height, width, channels = image.shape
    assert height == objects['height']
    assert width == objects['width']

    for xmin, ymin, xmax, ymax in objects['coords']:
        objects['cuts'].append(image[ymin:ymax, xmin:xmax, :])


def create_dataset(args):
    image_list = os.listdir(args.image_path)
    label_list = os.listdir(args.label_path)

    assert len(image_list) >= len(label_list)

    # Create output folders
    if args.target_format == 'subfolders':
        create_subfolders(args)

    print('\nIterating label files and writing cutouts to disk...')
    time.sleep(0.5)

    for label in tqdm(label_list, unit='label files', desc='Progress'):
        label_file = os.path.join(args.label_path, label)

        xml_tree = ET.parse(label_file).getroot()

        filename = xml_tree.find('filename').text
        width = int(xml_tree.find('size')[0].text)
        height = int(xml_tree.find('size')[1].text)

        objects = {
            'filename': filename,
            'width': width,
            'height': height,
            'class_ids': [],
            'coords': [],
            'cuts': []
        }

        for member in xml_tree.findall('object'):
            class_id = int(member[0].text)

            if class_id not in args.include:
                continue

            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)

            area = np.float((xmax - xmin) * (ymax - ymin))

            if area <= args.min_size:
                continue

            objects['class_ids'].append(class_id)
            objects['coords'].append((xmin, ymin, xmax, ymax))

        if len(objects['class_ids']) > 0:
            cut_objects(objects, args)

            if args.target_format == 'subfolders':
                write_cuts_to_subfolder(objects, args)


traffic_signs_prohibitory = [
    # Verwendete Klassen
    2, 3, 4, 5, 6, 8, 9, 10, 11, 16, 17, 122, 123, 124, 125, 126, 127, 128, 129, 130, 145,
    # Nicht verwendete Klassen
    # 1, 46, 48, 51, 55, 56, 57, 98, 100, 101, 102, 103, 104, 131, 136, 137, 138, 139, 140, 141, 176, 177
]
traffic_signs_danger = [
    # Verwendete Klassen
    12, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 53,
    # Nicht verwendete Klassen
    # 30, 31, 99
]
traffic_signs_mandatory = [
    # Verwendete Klassen
    34, 35, 36, 37, 38, 39, 41, 109, 110, 111, 112,
    # Nicht verwendete Klassen
    # 40, 105, 106, 107, 108, 113, 148, 149
]
traffic_signs_other = [
    # Verwendete Klassen
    13, 14, 15, 18, 33, 45, 54, 58, 59, 60, 61, 74, 85, 89, 91, 92, 93, 96, 116, 118, 119, 132,
    133, 155,
    # Nicht verwendete Klassen
    # 7, 42, 43, 44, 47, 49, 50, 52, 86, 87, 88, 90, 94, 95, 97, 114, 115, 117, 120, 121, 134, 135, 142, 143, 144,
    # 146, 147, 150, 151, 152, 174, 175
]
traffic_signs_digital = [
    # Verwendete Klassen
    62, 66, 70, 75, 81,
    # Nicht verwendete Klassen
    # 63, 64, 65, 67, 68, 69, 71, 72, 73, 76, 77, 78, 79, 80, 82, 83, 84, 153
]
traffic_signals = [
    # Verwendete Klassen
    158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
    # Nicht verwendete Klassen
    # 157
]
objects = [
    # Verwendete Klassen
    178, 179, 180, 181, 182, 183, 186, 187, 188, 189,
    # Nicht verwendete Klassen
    # 184
]
roadsurface = [
    # Verwendete Klassen
    193, 194, 195, 198, 199, 200,
    # Nicht verwendete Klassen
    # 192, 196, 197, 201, 202, 203, 204, 205, 206
]


def main():
    args = SimpleNamespace()

    args.image_path = '/home/osm/Schreibtisch/01_Datasets/2019_Mai/01_Rawdata/Images/'
    args.label_path = '/home/osm/Schreibtisch/01_Datasets/2019_Mai/01_Rawdata/Labels/'
    args.label_map = './label_map.json'
    args.output_path = '/home/osm/Schreibtisch/01_Datasets/2019_Mai/Classification_Supercategories_Used/roadsurface'
    args.target_format = 'subfolders'

    # Classes with '(danger)' but w/o '(digital)'
    # args.include = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 52, 98]

    # Classes with '(object)'
    # args.include = [177, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188]

    # Classes with '(traffic signal)'
    # args.include = list(range(155, 173))

    # All classes
    # args.include = list(range(206))

    args.include = [class_id - 1 for class_id in roadsurface]

    args.min_size = 0

    categories = json.load(open(args.label_map, 'r')).get('classes')
    args.categories = [{'id': cat['id'], 'name': cat['name']} for cat in categories]
    args.cat2id = {cat['name']: cat['id'] for cat in args.categories}
    args.id2cat = {cat['id']: cat['name'] for cat in args.categories}

    create_dataset(args)

    print('Written to {}'.format(args.output_path))


if __name__ == '__main__':
    main()
