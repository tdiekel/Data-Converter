import json
import os
import random
import shutil
import time
from datetime import datetime
from tempfile import mkdtemp

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm

import conv


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
        self.label_map = label_map
        self.file_lists = file_lists
        self.output_path = output_path
        self.excluded_classes = excluded_classes

        self.images_copied = False
        self.images_split = False

        self.categories = json.load(open(self.label_map, 'r')).get('classes')

        self.categories = [{'id': cat['id'] + 1,
                            'name': cat['name']
                            } for cat in self.categories]

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

    def calc_img_statistics(self):
        print("Calculating image statistics ...")

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
                                                                                self.img_std[1], self.img_std[2]))

        self._create_dir(self.output_path)
        with open(os.path.join(self.output_path, "image_stats.txt")) as file:
            file.write("Image statistics per RGB Channel")
            file.write("mean = [{}, {}, {}]\n".format(self.img_mean[0], self.img_mean[1], self.img_mean[2]))
            file.write("std = [{}, {}, {}]\n".format(self.img_std[0], self.img_std[1], self.img_std[2]))

    def calc_label_statistics(self):
        print("Creating dummy csv dataset ...")

        converter = conv.CSVConverter(self.image_path, self.image_src_type, self.image_dest_type, self.label_path,
                                      self.label_map, self.file_lists, mkdtemp(), self.excluded_classes)

        print("Calculating label statistics ...")
        dataframes = []

        for i, s in enumerate(self.image_sets):
            time.sleep(0.1)
            print("\tCalculating for images in {} ...".format(s))
            time.sleep(0.1)

            dataframes.append(converter.get_dataframe(s))
            self._print_label_stats(dataframes[i], s)

        if len(dataframes) != 1:
            df = pd.concat(dataframes)
            self._print_label_stats(df, set_title='full')

    def _print_label_stats(self, df, set_title):
        time.sleep(0.1)

        # General stats
        print('\nGeneral stats for \'{}\' set.'.format(set_title))

        df_general = self._get_general_stats(df)
        df_general.to_csv(os.path.join(self.output_path, '{}_general_stats.csv'.format(set_title)), index=None)

        print(tabulate(df_general, headers='keys', tablefmt='psql', showindex=False))

        # Class stats
        print('\nClass stats for \'{}\' set.'.format(set_title))

        df_class = self._get_class_stats(df)
        df_class.to_csv(os.path.join(self.output_path, '{}_class_stats.csv'.format(set_title)), index=None)

        columns_to_print = ['class_id', 'class', 'examples', 'num_bboxes_greater_than_32px',
                            'avg_x_center', 'avg_y_center', 'avg_bbox_w', 'avg_bbox_h',
                            'min_bbox_w', 'min_bbox_h', 'max_bbox_w', 'max_bbox_h']

        print(tabulate(df_class[columns_to_print], headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f"))

    def _get_general_stats(self, df):
        general_stats = ['images',
                         'avg. width', 'min. width', 'max. width',
                         'avg. height', 'min. height', 'max. height']

        data = [(df['filename'].count(),
                 df['width'].mean(), df['width'].min(), df['width'].max(),
                 df['height'].mean(), df['height'].min(), df['height'].max())]

        return pd.DataFrame(data, columns=general_stats)

    def _get_class_stats(self, df):
        class_stats = ['class_id', 'class', 'examples',
                       'avg_xmin', 'avg_ymin', 'avg_xmax', 'avg_ymax',
                       'min_x_center', 'min_y_center', 'min_bbox_w', 'min_bbox_h',
                       'avg_x_center', 'avg_y_center', 'avg_bbox_w', 'avg_bbox_h',
                       'max_x_center', 'max_y_center', 'max_bbox_w', 'max_bbox_h',
                       'num_bboxes_greater_than_32px',
                       'avg_rel_x_center', 'avg_rel_y_center', 'avg_rel_bbox_w', 'avg_rel_bbox_h'
                       ]

        class_list = []

        for class_id in sorted(df['class'].unique()):
            if class_id in self.excluded_classes:
                continue

            filtered_df = df[df['class'] == class_id]

            # Convert to center values
            x_center = filtered_df['xmax'] - (filtered_df['xmax'] - filtered_df['xmin']) / 2
            y_center = filtered_df['ymax'] - (filtered_df['ymax'] - filtered_df['ymin']) / 2
            bbox_w = filtered_df['xmax'] - filtered_df['xmin']
            bbox_h = filtered_df['ymax'] - filtered_df['ymin']

            # Convert to relative values
            rel_x_center = x_center / filtered_df['width'].mean()
            rel_y_center = y_center / filtered_df['height'].mean()
            rel_bbox_w = bbox_w / filtered_df['width'].mean()
            rel_bbox_h = bbox_h / filtered_df['height'].mean()

            class_name = self.id2cat[class_id]

            avg_xmin = filtered_df['xmin'].mean()
            avg_ymin = filtered_df['ymin'].mean()
            avg_xmax = filtered_df['xmax'].mean()
            avg_ymax = filtered_df['ymax'].mean()

            min_x_center = x_center.min()
            min_y_center = y_center.min()
            min_bbox_width = bbox_w.min()
            min_bbox_height = bbox_h.min()

            avg_x_center = x_center.mean()
            avg_y_center = y_center.mean()
            avg_bbox_width = bbox_w.mean()
            avg_bbox_height = bbox_h.mean()

            max_x_center = x_center.max()
            max_y_center = y_center.max()
            max_bbox_width = bbox_w.max()
            max_bbox_height = bbox_h.max()

            bbox_area = pd.DataFrame({'bbox_w': bbox_w, 'bbox_h': bbox_h})
            bboxes_greater_than_32px = bbox_area[(bbox_area['bbox_w'] >= 32) & (bbox_area['bbox_h'] >= 32)]
            num_bboxes_greater_than_32px = bboxes_greater_than_32px.count()['bbox_h']

            avg_rel_x_center = rel_x_center.mean() * 100
            avg_rel_y_center = rel_y_center.mean() * 100
            avg_rel_bbox_w = rel_bbox_w.mean() * 100
            avg_rel_bbox_h = rel_bbox_h.mean() * 100

            class_list.append((class_id, class_name, filtered_df['class'].count(),
                               avg_xmin, avg_ymin, avg_xmax, avg_ymax,
                               min_x_center, min_y_center, min_bbox_width, min_bbox_height,
                               avg_x_center, avg_y_center, avg_bbox_width, avg_bbox_height,
                               max_x_center, max_y_center, max_bbox_width, max_bbox_height,
                               num_bboxes_greater_than_32px,
                               avg_rel_x_center, avg_rel_y_center, avg_rel_bbox_w, avg_rel_bbox_h
                               ))

        return pd.DataFrame(class_list, columns=class_stats)

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
