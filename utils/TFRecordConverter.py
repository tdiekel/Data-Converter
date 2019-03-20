import time
import os
from utils.BaseConverter import BaseConverter
from utils.CSVConverter import CSVConverter
from utils.generate_tfrecord import generate_tfrecord


class TFRecordConverter(BaseConverter):

    def __init__(self, image_path, image_src_type, image_dest_type, label_path, label_map, output_path,
                 create_csv=False):
        super().__init__(image_path, image_src_type, image_dest_type, label_path, label_map, output_path)

        self.create_csv = create_csv

        if self.create_csv:
            self.csv_converter = CSVConverter(image_path, image_src_type, image_dest_type, label_path, label_map,
                                              output_path)

    def convert(self):
        print('\nConverting ...')
        if self.create_csv:
            self._copy_values_to_csv_converter()
            self.csv_converter.convert('\t')

        time.sleep(0.1)
        for image_set in self.image_sets:
            print('\n\tCreating tfrecord files for {} ...'.format(image_set))

            csv_path = os.path.join(self.output_path, '{}_labels.csv'.format(image_set))
            tfrecord_file = os.path.join(self.output_path, '{}.record'.format(image_set))

            generate_tfrecord(self.image_path, csv_path, self.id2cat, tfrecord_file)

        self._create_label_map_pbtxt()

    def _copy_values_to_csv_converter(self):
        self.csv_converter.images_copied = self.images_copied
        self.csv_converter.images_split = self.images_split

        self.csv_converter.images = self.images
        self.csv_converter.label = self.label

        self.csv_converter.image_sets = self.image_sets

    def _create_label_map_pbtxt(self):
        print('\nCreating label map file ...')

        label_map_file = os.path.join(self.output_path, 'label_map.pbtxt')
        item = 'item {{\n' \
               '  id: {}\n' \
               '  name: \'{}\'\n' \
               '}}\n' \
               '\n'

        with open(label_map_file, 'w') as f:
            for cat_id in self.id2cat:
                f.write(item.format(cat_id, self.id2cat[cat_id]))

