import json
import os
import random
import shutil
import time
from datetime import datetime
from tabulate import tabulate

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd


class BaseConverter:
    def __init__(self, image_path, image_src_type, image_dest_type, label_path, label_map, file_lists, output_path,
                 excluded_classes):
        self.info = {
            'description': 'IfF 2018 Dataset',
            'url': 'http://www.iff.tu-bs.de',
            'version': '1.0',
            'year': 2018,
            'contributor': 'IfF',
            'date_created': datetime.today().strftime('%Y/%m/%d')
        }

        self.image_path = image_path
        self.image_src_type = image_src_type
        self.image_dest_type = image_dest_type
        self.label_path = label_path
        self.file_lists = file_lists
        self.output_path = output_path
        self.excluded_classes = excluded_classes

        self.images_copied = False
        self.images_split = False

        self.categories = json.load(open(label_map, 'r')).get('classes')

        self._check_for_excluded_classes()

        self.cat2id = {cat['name']: cat['id'] for cat in self.categories}
        self.id2cat = {cat['id']: cat['name'] for cat in self.categories}

        self.gt_boxes = {cat['id']: {'name': cat['name'], 'num_gt_boxes': {}} for cat in self.categories}

        # Get included ids
        self.included_ids = []
        for cat_id in self.id2cat:
            self.included_ids.append(cat_id)

        self.images = {}
        self.label = {}
        self.image_sets = []

        self._fill_lists()

        assert self._validate_match(), 'Image and label files do not match.'

        self.img_mean = []
        self.img_var = []
        self.img_std = []

        # Create output folder
        self._create_dir(self.output_path)

    def _check_for_excluded_classes(self):
        self._create_dir(self.output_path)

        if not len(self.excluded_classes) == 0:
            with open(os.path.join(self.output_path, "excluded_classes.txt"), 'w') as file:
                remaining_categories = []

                for cat in self.categories:
                    if not cat['id'] in self.excluded_classes:
                        remaining_categories.append(cat)
                    else:
                        file.write("{}\t: {}\n".format(cat['id'], cat['name']))

            self.categories = remaining_categories

        with open(os.path.join(self.output_path, "included_classes.txt"), 'w') as file:
            for cat in self.categories:
                file.write("{}\t: {}\n".format(cat['id'], cat['name']))

    def _fill_lists(self):
        if self.file_lists is not None:
            for file_list in self.file_lists:
                lines = list(open(file_list))

                set_name = lines.pop(0).split('=')[1].rstrip("\n\r")
                self.image_sets.append(set_name)

                self.images[set_name] = ['{}.{}'.format(filename.rstrip("\n\r"), self.image_src_type) for filename in
                                         lines]
                self.label[set_name] = ['{}.xml'.format(filename.rstrip("\n\r")) for filename in lines]

            self.images_split = True
        else:
            self.image_sets = ['images']

            self.images['images'] = [file for file in os.listdir(self.image_path) if file.endswith(self.image_src_type)]
            self.label['images'] = [file for file in os.listdir(self.label_path) if file.endswith('.xml')]

    def _validate_match(self):
        valid = True

        for s in self.image_sets:
            image_list = [file[:-4] for file in self.images[s]]
            label_list = [file[:-4] for file in self.label[s]]

            valid = valid and sorted(image_list) == sorted(label_list)

        return valid

    def calc_statistics(self):
        print("Calculating dataset statistics ...")

        mean_per_image = []

        for s in self.image_sets:
            time.sleep(0.1)
            print("\tCalculating mean and variance for images in {} ...".format(s))
            time.sleep(0.1)

            for i, image_filename in enumerate(tqdm(self.images[s], desc="\tProgress:", unit="files")):
                image_path = os.path.join(self.image_path, image_filename)
                image_arr = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

                mean_per_image.append(np.mean(image_arr, axis=(0, 1)))
                self.img_var.append((np.var(image_arr, axis=(0, 1))))

        self.img_mean = np.mean(mean_per_image, axis=0)
        self.img_std = np.sqrt(np.divide(np.sum(self.img_var, axis=0), i))
        print("Mean (RGB) = {}, {}, {}\nStandard deviation = {}, {}, {}".format(self.img_mean[0], self.img_mean[1],
                                                                                self.img_mean[2], self.img_std[0],
                                                                                self.img_std[1],
                                                                                self.img_std[2]))

        self._create_dir(self.output_path)
        with open(os.path.join(self.output_path, "image_stats.txt")) as file:
            file.write("Image statistics per RGB Channel")
            file.write("mean = [{}, {}, {}]\n".format(self.img_mean[0], self.img_mean[1], self.img_mean[2]))
            file.write("std = [{}, {}, {}]\n".format(self.img_std[0], self.img_std[1], self.img_std[2]))

    def split(self, sets, set_sizes, shuffle):
        if self.images_split:
            return

        print('\nSplitting data...')

        sets = [s + str(self.info['year']) for s in sets]
        images_per_set = [int(len(self.images['images']) * size / sum(set_sizes)) for size in set_sizes]

        # Add remainder to first set
        while sum(images_per_set) < len(self.images['images']):
            images_per_set[0] += 1

        if shuffle:
            self._shuffle()

        for s, images_per_set in zip(sets, images_per_set):
            time.sleep(0.1)
            print('\tCreating {} with {} images...'.format(s, images_per_set))
            time.sleep(0.1)

            # Make set output dir
            set_dir = os.path.join(self.output_path, s)
            if not os.path.exists(set_dir):
                os.makedirs(set_dir)
            else:
                # Check if images already split and copied
                if len(os.listdir(set_dir)) == images_per_set:
                    self.images_copied = True

            self.image_sets.append(s)
            self.images[s] = []
            self.label[s] = []

            for i in tqdm(range(images_per_set), unit='file(s)', desc='\tProgress:'):
                self.images[s].append(self.images['images'].pop())
                self.label[s].append(self.label['images'].pop())

                if not self.images_copied:
                    self._save_image(i, s)

        self.images.pop('images')
        self.label.pop('images')
        self.image_sets.remove('images')

        if self.file_lists is None:
            self._write_file_list()

        self.images_copied = True
        self.images_split = True
        self.image_src_type = self.image_dest_type

    def _shuffle(self):
        zipped = list(zip(self.images['images'], self.label['label']))

        random.shuffle(zipped)

        images, label = zip(*zipped)
        self.images['images'], self.label['label'] = list(images), list(label)

    def _write_file_list(self):
        for s in self.image_sets:
            with open(os.path.join(self.output_path, s + '_file_list.txt'), 'w')as file:
                file.write("Set={}\n".format(s))

                for filename in self.images[s]:
                    file.write(filename[:-4] + '\n')

    def _copy_all_images(self):
        time.sleep(0.1)
        print("\nCopying all images ...")

        for s in self.image_sets:
            time.sleep(0.1)
            print("\tRunning copy operation for {} ...".format(s))
            time.sleep(0.1)

            for idx in tqdm(range(len(self.images[s])), desc='\tProgress', unit='files'):
                self._save_image(idx, s)

        self.images_copied = True

    def _save_image(self, idx, image_set):
        image_path = os.path.join(self.image_path, self.images[image_set][idx])
        assert os.path.isfile(image_path), "File not found: {}".format(image_path)

        output_path = self._create_dir(os.path.join(self.output_path, image_set))

        if not self.image_src_type == self.image_dest_type:
            image = Image.open(image_path).convert('RGB')
            self.images[image_set][idx] = os.path.basename(image_path).replace('.' + self.image_src_type,
                                                                               '.' + self.image_dest_type)

            image_out_path = os.path.join(output_path, self.images[image_set][idx])
            image.save(image_out_path)
        else:
            image_out_path = os.path.join(output_path, os.path.basename(image_path))

            if not os.path.isfile(image_out_path):
                shutil.copyfile(image_path, image_out_path)

    @staticmethod
    def _create_dir(path):
        if not os.path.isdir(path):
            os.makedirs(path)

        return path

    def print_class_distribution(self):
        data = dict()

        data['class id'] = [class_id for class_id in self.gt_boxes]
        data['class'] = [self.gt_boxes[class_id]['name'] for class_id in self.gt_boxes]
        data['#bbox'] = [0 for class_id in self.gt_boxes]

        columns = ['#bbox']

        for image_set in self.image_sets:
            column = '#bbox in {}'.format(image_set)
            columns.append(column)
            # data[column] = []

            for i, class_id in enumerate(self.gt_boxes):
                num = self.gt_boxes[class_id]['num_gt_boxes'].get(image_set, 0)

                # data[column].append(num)
                data['#bbox'][i] += num

        df = pd.DataFrame(data=data)
        df.to_csv(os.path.join(self.output_path, 'class_distribution.csv'), index=None)

        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

        time.sleep(1)
        self._print_warning_for_empty_classes(data)

    @staticmethod
    def _print_warning_for_empty_classes(data):
        emtpy_classes = []

        for i, num_bbox in enumerate(data['#bbox']):
            if num_bbox == 0:
                emtpy_classes.append(str(data['class id'][i]))

        print('Recommended to exclude the following class ids with 0 bboxes: {}'.format(' '.join(emtpy_classes)))
