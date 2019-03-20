import os
import sys
import time
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm

from utils.BaseConverter import BaseConverter


class CSVConverter(BaseConverter):

    def __init__(self, image_path, image_src_type, image_dest_type, label_path, label_map, output_path):
        super().__init__(image_path, image_src_type, image_dest_type, label_path, label_map, output_path)

        self.column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    def convert(self, indent=""):
        time.sleep(0.1)
        print("\n" + indent + "Creating csv dataset...")

        for image_set in self.image_sets:
            time.sleep(0.1)
            print(indent + "\tCreating {} set...".format(image_set))
            time.sleep(0.1)

            df = self._xml_to_dataframe(image_set, indent)
            df.to_csv(os.path.join(self.output_path, '{}_labels.csv'.format(image_set)), index=None)

    def _xml_to_dataframe(self, image_set, indent):
        xml_list = []

        for xml_filename in tqdm(self.label[image_set], unit="files", desc=indent + '\tProgress:'):
            xml_file = os.path.join(self.label_path, xml_filename)

            xml_tree = ET.parse(xml_file).getroot()

            for member in xml_tree.findall('object'):
                filename = xml_tree.find('filename').text
                width = int(xml_tree.find('size')[0].text)
                height = int(xml_tree.find('size')[1].text)
                class_id = member[0].text
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)

                if int(class_id) > self.max_id:
                    UserWarning(
                        'Error: Class ID {} greater than max ID {} in label file: {}'.format(class_id, self.max_id,
                                                                                             xml_file))
                    sys.exit(-1)

                xml_list.append((filename, width, height, class_id, xmin, ymin, xmax, ymax))

        return pd.DataFrame(xml_list, columns=self.column_names)
