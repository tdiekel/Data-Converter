import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm import tqdm

import converters
from util.util import warning_not_verified_label_files


class CSVConverter(converters.BaseConverter):

    def __init__(self, args):
        super().__init__(args)

        self.column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    def convert(self):
        time.sleep(0.1)
        print("\nCreating csv dataset...")

        for image_set in self.image_sets:
            time.sleep(0.1)
            print("\tCreating {} set...".format(image_set))
            time.sleep(0.1)

            df = self._xml_to_dataframe(image_set)
            df.to_csv(os.path.join(self.output_path, '{}_labels.csv'.format(image_set)), index=None)

        if not self.images_copied:
            self._copy_all_images()

        if self.args.show_not_verified:
            warning_not_verified_label_files(self.not_verified_label_files)

    def get_dataframe(self, image_set):
        return self._xml_to_dataframe(image_set)

    def _xml_to_dataframe(self, image_set):
        xml_list = []

        for xml_filename in tqdm(self.label[image_set], unit="files", desc='\tProgress:'):
            xml_file = os.path.join(self.label_path, xml_filename)

            xml_tree = ET.parse(xml_file).getroot()

            if "verified" not in xml_tree.attrib:
                self.not_verified_label_files.append(xml_file)

            filename = xml_tree.find('filename').text
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

                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)

                area = np.float((xmax - xmin) * (ymax - ymin))

                if self.args.exclude_area is not None:
                    if area <= self.args.exclude_area:
                        continue

                if class_id in self.gt_boxes:
                    if image_set in self.gt_boxes[class_id]['num_gt_boxes']:
                        self.gt_boxes[class_id]['num_gt_boxes'][image_set] += 1
                    else:
                        self.gt_boxes[class_id]['num_gt_boxes'][image_set] = 1

                xml_list.append((filename, width, height, class_id, xmin, ymin, xmax, ymax))

        return pd.DataFrame(xml_list, columns=self.column_names)
