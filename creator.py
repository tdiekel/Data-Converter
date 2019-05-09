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


def main():
    args = SimpleNamespace()

    args.image_path = '/home/osm/Schreibtisch/01_Datasets/2019_Mai/01_Rawdata/Images/'
    args.label_path = '/home/osm/Schreibtisch/01_Datasets/2019_Mai/01_Rawdata/Labels/'
    args.label_map = './label_map.json'
    args.output_path = '/home/osm/Schreibtisch/01_Datasets/2019_Mai/Classification'
    args.target_format = 'subfolders'

    # Classes with '(danger)' but w/o '(digital)'
    # args.include = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 52, 98]

    # Classes with '(object)'
    # args.include = [177, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188]

    # Classes with '(traffic signal)'
    # args.include = list(range(155, 173))

    # All classes
    args.include = list(range(206))

    args.min_size = 0

    categories = json.load(open(args.label_map, 'r')).get('classes')
    args.categories = [{'id': cat['id'], 'name': cat['name']} for cat in categories]
    args.cat2id = {cat['name']: cat['id'] for cat in args.categories}
    args.id2cat = {cat['id']: cat['name'] for cat in args.categories}

    create_dataset(args)


if __name__ == '__main__':
    main()
