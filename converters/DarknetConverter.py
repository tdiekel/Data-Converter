import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

from converters.BaseConverter import BaseConverter
from util.util import create_dir, warning_not_verified_label_files


class DarknetConverter(BaseConverter):

    def __init__(self, args):
        super().__init__(args)

        self.rel_output_path = args.rel_output_path
        self.dataset_name = args.dataset_name
        self.skipped_labels = []

    def convert(self):
        time.sleep(0.1)
        print("\nCreating darknet dataset...")

        for image_set in self.image_sets:
            time.sleep(0.1)
            print("\n\tCreating {} set...".format(image_set))
            print("\t\tWriting label files ...")
            time.sleep(0.1)

            set_file_list = self._create_label_files(image_set)

            set_file = os.path.join(self.output_path, "{}.txt".format(image_set))
            with open(set_file, 'w') as file:
                for line in set_file_list:
                    file.write("{}\n".format(line))

            print("\t\tSkipped {} bboxes.".format(len(self.skipped_labels)))
            for class_id, var, value, xml_file in self.skipped_labels:
                print("\t\tSkipped class with id {class_id}, because {var} was {value} in label file: {xml_file}".format(
                    class_id=class_id, var=var, value=value, xml_file=xml_file))

            self.skipped_labels = []

        print("\nWriting config files ...")
        self._create_cfg_files()

        if not self.images_copied:
            self._copy_all_images()

        if self.args.show_not_verified:
            warning_not_verified_label_files(self.not_verified_label_files)

    def _create_label_files(self, image_set):
        label_target_folder = create_dir(os.path.join(self.output_path, image_set))
        set_file_list = []

        for xml_filename in tqdm(self.label[image_set], unit="files", desc='\t\tProgress:'):
            xml_file = os.path.join(self.label_path, xml_filename)

            xml_tree = ET.parse(xml_file).getroot()

            if "verified" not in xml_tree.attrib:
                self.not_verified_label_files.append(xml_file)

            label_file = os.path.join(label_target_folder, xml_filename.replace('.xml', '.txt'))
            with open(label_file, 'w') as file:

                # Get image width and height
                width = int(xml_tree.find('size')[0].text)
                height = int(xml_tree.find('size')[1].text)

                for member in xml_tree.findall('object'):
                    if not str(member[0].text).isdigit():
                        print(
                            '\nError: Class ID \'{}\' not convertible to integer. Found in label file: {}'.format(
                                member[0].text, xml_file))
                        sys.exit(-1)

                    class_id = int(member[0].text) + 1

                    if class_id in self.label_id_mapping:
                        class_id = self.label_id_mapping[class_id]

                    if self.args.rearrange_ids:
                        if class_id in self.label_rearrange_mapping:
                            class_id = self.label_rearrange_mapping[class_id]
                        else:
                            continue

                    if class_id in self.excluded_classes:
                        continue

                    if class_id not in self.included_ids:
                        print(
                            'Error: Class ID {} not in label map or not included. Found in label file: {}'.format(
                                str(class_id), xml_file))
                        sys.exit(-1)

                    x_min = int(member[4][0].text)
                    y_min = int(member[4][1].text)
                    x_max = int(member[4][2].text)
                    y_max = int(member[4][3].text)

                    area = np.float((x_max - x_min) * (y_max - y_min))

                    if self.args.exclude_area is not None:
                        if area <= self.args.exclude_area:
                            continue

                    if class_id in self.gt_boxes:
                        if image_set in self.gt_boxes[class_id]['num_gt_boxes']:
                            self.gt_boxes[class_id]['num_gt_boxes'][image_set] += 1
                        else:
                            self.gt_boxes[class_id]['num_gt_boxes'][image_set] = 1

                    # Convert to center values
                    x_center = x_max - (x_max - x_min) / 2
                    y_center = y_max - (y_max - y_min) / 2
                    bbox_w = x_max - x_min
                    bbox_h = y_max - y_min

                    # Convert to relative values
                    x = x_center / width
                    y = y_center / height
                    w = bbox_w / width
                    h = bbox_h / height

                    # Check boundaries
                    if not 0.0 < x <= 1.0:
                        self.skipped_labels.append((class_id, "x", x, xml_file))
                        continue
                    if not 0.0 < y <= 1.0:
                        self.skipped_labels.append((class_id, "y", y, xml_file))
                        continue
                    if not 0.0 < w <= 1.0:
                        self.skipped_labels.append((class_id, "w", w, xml_file))
                        continue
                    if not 0.0 < h <= 1.0:
                        self.skipped_labels.append((class_id, "h", h, xml_file))
                        continue

                    # Write to file
                    file.write(
                        '{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n'.format(class_id=class_id - 1, x=x, y=y, w=w, h=h))

            # Add file to set list
            filename = xml_tree.find('filename').text
            set_file_list.append(os.path.join(self.rel_output_path, image_set, filename))

        return set_file_list

    def _create_cfg_files(self):
        # .data file
        data_str = "classes = {class_num}\n" \
                   "{sets}" \
                   "names = {rel_path}/{dataset_name}.names\n" \
                   "backup = ./training_diekel/{dataset_name}"

        if len(self.image_sets) == 1:
            image_sets = "train = {rel_path}/{set}.txt\n".format(rel_path=self.rel_output_path, set=self.image_sets[0])
        else:
            image_sets = "train = {rel_path}/{set1}.txt\n" \
                         "valid = {rel_path}/{set2}.txt\n".format(rel_path=self.rel_output_path, set1=self.image_sets[0], set2=self.image_sets[1])

        data_str = data_str.format(class_num=len(self.included_ids),
                                   sets=image_sets,
                                   rel_path=self.rel_output_path,
                                   dataset_name=self.dataset_name)

        data_file = os.path.join(self.output_path, self.dataset_name + '.data')

        with open(data_file, 'w') as f:
            f.write(data_str)

        # .names file
        name_str = '\n'.join([cat['name'] for cat in self.categories])

        name_file = os.path.join(self.output_path, self.dataset_name + '.names')

        with open(name_file, 'w') as f:
            f.write(name_str)
