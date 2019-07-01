import copy
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from shutil import copyfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm

import converters
from util.util import create_dir, validate_match, print_label_stats, print_warning_for_empty_classes, \
    check_label_names_for_duplicates, find_value


class BaseConverter:
    def __init__(self, args):
        self.info = {
            'description': 'IfF {} Dataset'.format(args.year),
            'url': 'http://www.iff.tu-bs.de',
            'version': '1.0',
            'year': args.year,
            'contributor': 'IfF',
            'date_created': datetime.today().strftime('%Y/%m/%d')
        }

        self.args = args
        self.image_path = args.image_path
        self.image_src_filetype = args.image_src_filetype
        self.image_dest_filetype = args.image_dest_filetype
        self.label_path = args.label_path
        self.label_map = args.label_map
        self.file_lists = args.file_lists
        self.output_path = args.output_path
        if 'exclude' in args:
            self.excluded_classes = args.exclude
        if 'include' in args:
            self.included_classes = args.include
        self.remap_labels = args.remap_labels

        self.images_copied = args.no_copy
        self.images_split = False
        self.skip_images_without_label = args.skip_images_without_label

        '''Create class parameter
        '''
        self.categories = {}
        self.org_categories = {}

        self.included_ids = []

        self.label_id_mapping = {}
        self.label_rearrange_mapping = {}

        self.cat2id = {}
        self.id2cat = {}
        self.gt_boxes = {}

        self.images = {}
        self.label = {}
        self.image_sets = []

        self.not_verified_label_files = []

        self.img_mean = []
        self.img_var = []
        self.img_std = []

    def init(self):
        # Create output folder
        create_dir(self.output_path)

        # Copy label map
        copyfile(self.label_map, os.path.join(self.output_path, self.label_map))

        self.categories = json.load(open(self.label_map, 'r')).get('classes')
        self.categories = [{'id': cat['id'] + 1, 'name': cat['name']} for cat in self.categories]
        self.org_categories = copy.deepcopy(self.categories)

        if self.included_classes is None:
            self._check_for_excluded_classes()
        else:
            self._check_for_included_classes()

        # Get included ids
        self.included_ids = [cat['id'] for cat in self.categories]

        if self.remap_labels:
            self._remap_labels()

        if self.args.rearrange_ids:
            self._rearrange_ids()

        self._write_label_map()

        if check_label_names_for_duplicates(self.categories):
            print('\nExiting! Please fix label map.')
            sys.exit(-1)

        self.cat2id = {cat['name']: cat['id'] for cat in self.categories}
        self.id2cat = {cat['id']: cat['name'] for cat in self.categories}

        self.gt_boxes = {cat['id']: {'name': cat['name'], 'num_gt_boxes': {}} for cat in self.categories}

        self._fill_lists()

        assert validate_match(self.image_sets, self.images, self.label), 'Image and label files do not match.'

    def _check_for_excluded_classes(self):
        create_dir(self.output_path)

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

    def _check_for_included_classes(self):
        create_dir(self.output_path)

        self.excluded_classes = []

        with open(os.path.join(self.output_path, "excluded_classes.txt"), 'w') as file:
            remaining_categories = []

            for cat in self.categories:
                if cat['id'] in self.included_classes:
                    remaining_categories.append(cat)
                else:
                    self.excluded_classes.append(cat['id'])
                    file.write("{}\t: {}\n".format(cat['id'], cat['name']))

        self.categories = remaining_categories

        with open(os.path.join(self.output_path, "included_classes.txt"), 'w') as file:
            for cat in self.categories:
                file.write("{}\t: {}\n".format(cat['id'], cat['name']))

    def combine_by_id(self):
        new_categories = []
        labels_merged_per_id = {}

        if 'ids_from_org_list' not in self.args.mapping:
            use_org_ids = True
        else:
            use_org_ids = self.args.mapping['ids_from_org_list']

        # Get all remapped ids for faster processing
        remapped_ids = set()
        for new_labels in self.args.mapping['new_labels']:
            for old_id in new_labels['old_id']:
                remapped_ids.add(old_id)

            remapped_ids.add(new_labels['new_id'])
        remapped_ids = sorted(remapped_ids)

        for old_cat in self.categories:

            if old_cat['id'] in remapped_ids:
                for new_cat in self.args.mapping['new_labels']:

                    assert isinstance(new_cat['new_id'], int), 'New ID must be int. Got {} of type {}'.format(
                        new_cat['new_id'], type(new_cat['new_id']))

                    assert isinstance(new_cat['old_id'], int) \
                           or isinstance(new_cat['old_id'], list), 'Old ID(s) must be int or list.' \
                                                                   ' Got type {}'.format(type(new_cat['old_id']))

                    assert 'new_name' in new_cat, 'No new name set for new ID {}.'.format(new_cat['new_id'])

                    if use_org_ids:
                        new_cat['new_id'] += 1
                        new_cat['old_id'] = [i + 1 for i in new_cat['old_id']]

                    if old_cat['id'] in new_cat['old_id']:
                        rename = len(new_cat['old_id']) == 1

                        if not rename:
                            # Avoid double entries
                            if not len(list(filter(lambda cat: cat['id'] == new_cat['new_id'], new_categories))) == 0:
                                # Save merging statistics
                                labels_merged_per_id[new_cat['new_id']] += 1
                                continue

                        new_categories.append({'id': new_cat['new_id'], 'name': new_cat['new_name']})

                        for old_id in new_cat['old_id']:
                            self.label_id_mapping[old_id] = new_cat['new_id']

                        # Save merging statistics
                        labels_merged_per_id[new_cat['new_id']] = 1

                        # Remove old ids from included
                        old_ids = copy.deepcopy(new_cat['old_id'])
                        if new_cat['new_id'] in old_ids:
                            old_ids.remove(new_cat['new_id'])

                        for old_id in old_ids:
                            if old_id in self.included_ids:
                                self.included_ids.remove(old_id)
                                self.excluded_classes.append(old_id)

                        new_id = new_cat['new_id']
                        if new_id not in self.included_ids:
                            self.included_ids.append(new_cat['new_id'])
                        if new_id in self.excluded_classes:
                            self.excluded_classes.remove(new_id)

                        break

            else:
                new_categories.append(old_cat)

        new_id2new_cat = {cat['id']: cat['name'] for cat in new_categories}

        return new_categories, new_id2new_cat, labels_merged_per_id

    def combine_by_substring(self):
        new_categories = [{'id': new_label['new_id'],
                           'name': new_label['new_name'],
                           'supercategory': new_label['substring'].replace('(', '').replace(')', '')}
                          for new_label in self.args.mapping['new_labels']]
        new_id2new_cat = {cat['id']: cat['name'] for cat in new_categories}

        labels_merged_per_id = {new_cat['id']: 0 for new_cat in new_categories}

        for old_cat in self.categories:
            old_id = old_cat['id']
            old_name = old_cat['name']

            for new_cat in self.args.mapping['new_labels']:
                new_id = new_cat['new_id']
                substring = new_cat['substring']

                exclude = None
                if 'exclude' in new_cat:
                    exclude = new_cat['exclude']

                if substring in old_name:
                    if exclude is not None and exclude in old_cat['name']:
                        continue

                    self.label_id_mapping[old_id] = new_id
                    labels_merged_per_id[new_id] += 1
                    break

        i = len(self.included_ids) - 1
        while i >= 0:
            found = False

            for old_id in self.label_id_mapping:
                if old_id == self.included_ids[i]:
                    found = True
                    break

            if not found:
                self.excluded_classes.append(self.included_ids[i])
                del self.included_ids[i]

            i -= 1

        return new_categories, new_id2new_cat, labels_merged_per_id

    def _remap_labels(self):
        print('Remapping labels ...')

        if self.args.mapping['type'] == 'combine_by_id':
            new_categories, new_id2new_cat, labels_merged_per_id = self.combine_by_id()
        elif self.args.mapping['type'] == 'combine_by_substring':
            new_categories, new_id2new_cat, labels_merged_per_id = self.combine_by_substring()
        else:
            return

        if not len(self.categories) == len(new_categories):
            print('\tReduced from {} to {} class(es).'.format(len(self.categories), len(new_categories)))
            for label_id in labels_merged_per_id:
                if labels_merged_per_id[label_id] == 0:
                    print('\tNo matching classes found for class {} with id {}. Ignoring class.'.format(
                        new_id2new_cat[label_id], label_id))

                    i = len(new_categories) - 1
                    while i >= 0:
                        if new_categories[i]['id'] == label_id:
                            del new_categories[i]
                            break
                        i -= 1

                else:
                    print('\tMapped {} classes to {} with id {}.'.format(labels_merged_per_id[label_id],
                                                                         new_id2new_cat[label_id], label_id))

        self.categories = new_categories
        self._overwrite_in_and_excluded_classes_files()
        # Get included ids
        self.included_ids = [cat['id'] for cat in self.categories]

    def _rearrange_ids(self):
        old_id2cat = {cat['id']: cat['name'] for cat in self.categories}
        new_categories = []
        new_included = []

        for new_id, cat_id in enumerate(sorted([cat['id'] for cat in self.categories]), start=1):
            self.label_rearrange_mapping[cat_id] = new_id
            new_categories.append({'id': new_id, 'name': old_id2cat[cat_id]})
            new_included.append(new_id)

            if new_id in self.excluded_classes:
                self.excluded_classes.remove(new_id)

        for cat_id in [cat['id'] for cat in self.categories]:
            if cat_id not in new_included:
                self.excluded_classes.append(cat_id)

        '''Overwrite lists'''
        self.categories = new_categories
        self.included_ids = new_included

        # Remove double entries
        self.excluded_classes = list(set(self.excluded_classes))

    def _overwrite_in_and_excluded_classes_files(self):
        id2cat = {cat['id']: cat['name'] for cat in self.org_categories}

        info_str = '###########################################\n' \
                   '###     File contains old label IDs     ###\n' \
                   '###########################################\n'

        with open(os.path.join(self.output_path, "excluded_classes.txt"), 'w') as file:
            file.write(info_str)

            for class_id in self.excluded_classes:
                if class_id not in self.label_id_mapping:
                    file.write("{}\t: {}\n".format(class_id, id2cat[class_id]))

        with open(os.path.join(self.output_path, "included_classes.txt"), 'w') as file:
            file.write(info_str)

            # Get all in sorted list
            included = [class_id for class_id in self.included_ids]
            included.extend([class_id for class_id in self.label_id_mapping])
            included = sorted(included)

            for class_id in included:
                file.write("{}\t: {}\n".format(class_id, id2cat[class_id]))

    def _write_label_map(self):
        print('Saving label map and id mapping to output folder.')

        label_map_file = os.path.join(self.output_path, 'label_map.json')
        with open(label_map_file, 'w') as f:
            json.dump({'classes': self.categories}, f, sort_keys=True, indent=4)

        label_id_mapping_file = os.path.join(self.output_path, 'label_id_mapping.json')

        label_id_mapping = {}
        if not len(self.label_rearrange_mapping) == 0:
            for new_id in self.included_ids:
                old_id = find_value(self.label_id_mapping, find_value(self.label_rearrange_mapping, new_id))
                if old_id is None:
                    old_id = find_value(self.label_rearrange_mapping, new_id)

                label_id_mapping[old_id] = new_id
        else:
            label_id_mapping = self.label_id_mapping

        with open(label_id_mapping_file, 'w') as f:
            json.dump({'old_id_to_new_id': label_id_mapping}, f, sort_keys=True, indent=4)

    def _fill_lists(self):
        if self.file_lists is not None:
            for file_list in self.file_lists:
                lines = list(open(file_list))

                set_name = lines.pop(0).split('=')[1].rstrip("\n\r")
                self.image_sets.append(set_name)

                self.images[set_name] = ['{}.{}'.format(filename.rstrip("\n\r"), self.image_src_filetype) for filename
                                         in
                                         lines]
                self.label[set_name] = ['{}.xml'.format(filename.rstrip("\n\r")) for filename in lines]

            self.images_split = True
        else:
            self.image_sets = ['images']

            self.images['images'] = [file for file in os.listdir(self.image_path) if
                                     file.endswith(self.image_src_filetype)]
            self.label['images'] = [file for file in os.listdir(self.label_path) if file.endswith('.xml')]

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

        create_dir(self.output_path)
        with open(os.path.join(self.output_path, "image_stats.txt"), 'w') as file:
            file.write("Image statistics per RGB Channel")
            file.write("mean = [{}, {}, {}]\n".format(self.img_mean[0], self.img_mean[1], self.img_mean[2]))
            file.write("std = [{}, {}, {}]\n".format(self.img_std[0], self.img_std[1], self.img_std[2]))

    def calc_label_statistics(self, max_classes=206):
        time.sleep(0.1)
        print("\n\nCreating dummy csv dataset ...")

        converter = converters.CSVConverter(self.args)
        converter.images = self.images
        converter.label = self.label
        converter.id2cat = self.id2cat
        converter.excluded_classes = self.excluded_classes
        converter.included_ids = self.included_ids
        converter.label_id_mapping = self.label_id_mapping
        converter.label_rearrange_mapping = self.label_rearrange_mapping

        print("Calculating label statistics ...")
        dataframes = []

        for i, s in enumerate(self.image_sets):
            time.sleep(0.1)
            print("\tCalculating for images in {} ...".format(s))
            time.sleep(0.1)

            dataframes.append(converter.get_dataframe(s))
            print_label_stats(self.output_path, self.id2cat, max_classes, self.excluded_classes, dataframes[i], s,
                              tablefmt=self.args.tablefmt)

        if len(dataframes) != 1:
            df = pd.concat(dataframes)
            print_label_stats(self.output_path, self.id2cat, max_classes, self.excluded_classes, df, set_title='full',
                              tablefmt=self.args.tablefmt)

    def split(self, sets, set_sizes, shuffle):
        if self.images_split:
            return

        print('\nSplitting data...')

        sets = [s + str(self.info['year']) for s in sets]
        num_images = len(self.images['images'])
        images_per_set = [int(num_images * size / sum(set_sizes)) for size in set_sizes]

        # Add remainder to first set
        while sum(images_per_set) < num_images:
            images_per_set[0] += 1

        print('Resulting distribution:', '\n' + tabulate(
            tabular_data=[{'Set': s, 'Fraction [%]': images_per_set[i] / num_images * 100} for i, s in enumerate(sets)],
            headers='keys', tablefmt=self.args.tablefmt, showindex=False, floatfmt=".3f"
        ))

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
        self.image_src_filetype = self.image_dest_filetype

    def _shuffle(self):
        zipped = list(zip(self.images['images'], self.label['images']))

        random.shuffle(zipped)

        images, label = zip(*zipped)
        self.images['images'], self.label['images'] = list(images), list(label)

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

        output_path = create_dir(os.path.join(self.output_path, image_set))

        if not self.image_src_filetype == self.image_dest_filetype:
            image = Image.open(image_path).convert('RGB')
            self.images[image_set][idx] = os.path.basename(image_path).replace('.' + self.image_src_filetype,
                                                                               '.' + self.image_dest_filetype)

            image_out_path = os.path.join(output_path, self.images[image_set][idx])
            image.save(image_out_path)
        else:
            image_out_path = os.path.join(output_path, os.path.basename(image_path))

            if not os.path.isfile(image_out_path):
                shutil.copyfile(image_path, image_out_path)

    def print_class_distribution(self):
        print('\nPrinting class distribution for image sets...')

        data = dict()
        data['class id'] = [class_id for class_id in sorted(self.gt_boxes)]
        data['class'] = [self.gt_boxes[class_id]['name'] for class_id in sorted(self.gt_boxes)]
        data['#bbox'] = [0 for _ in self.gt_boxes]

        columns = ['#bbox']

        for image_set in self.image_sets:
            column_num = '#bbox in {}'.format(image_set)
            columns.append(column_num)
            data[column_num] = []

            column_frac = 'fraction {} [%]'.format(image_set)
            columns.append(column_frac)
            data[column_frac] = []

            column_targ = 'target delta {} [%]'.format(image_set)
            columns.append(column_targ)
            data[column_targ] = []

            for i, class_id in enumerate(sorted(self.gt_boxes)):
                num = self.gt_boxes[class_id]['num_gt_boxes'].get(image_set, 0)

                data['#bbox'][i] += num
                data[column_num].append(num)

        for s, image_set in enumerate(self.image_sets):
            column_num = '#bbox in {}'.format(image_set)
            column_frac = 'fraction {} [%]'.format(image_set)
            column_targ = 'target delta {} [%]'.format(image_set)

            for i in range(len(data['class id'])):
                if data['#bbox'][i] == 0:
                    data[column_frac].append(0)
                    data[column_targ].append(0)
                else:
                    fraction = data[column_num][i] / data['#bbox'][i] * 100

                    target_fraction = self.args.set_sizes[s]
                    fraction_delta = target_fraction - fraction

                    data[column_frac].append(fraction)
                    data[column_targ].append(fraction_delta)

        df = pd.DataFrame(data=data)
        df.to_csv(os.path.join(self.output_path, 'class_distribution.csv'), index=None)

        print(tabulate(df, headers='keys', tablefmt=self.args.tablefmt, showindex=False, floatfmt=".2f"))

        time.sleep(1)
        print_warning_for_empty_classes(data)
