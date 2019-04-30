import os
import time

import conv


class TFRecordConverter(conv.BaseConverter):

    def __init__(self, args):
        super().__init__(args)

        self.csv_converter = conv.CSVConverter(args)

    def convert(self):
        time.sleep(0.1)
        self._copy_values_to_csv_converter()
        self.csv_converter.convert()
        self._copy_values_from_csv_converter()

        time.sleep(0.1)
        for image_set in self.image_sets:
            print('\nCreating tfrecord files for {} ...'.format(image_set))
            self.gt_boxes = conv.generate_tfrecord(self.image_path, self.output_path, image_set, self.id2cat, self.gt_boxes)

        self._create_label_map_pbtxt()

        if not self.images_copied:
            self._copy_all_images()

        self._warning_not_verfied_label_files()

    def _copy_values_to_csv_converter(self):
        self.csv_converter.images_copied = self.images_copied
        self.csv_converter.images_split = self.images_split

        self.csv_converter.images = self.images
        self.csv_converter.label = self.label

        self.csv_converter.image_sets = self.image_sets

    def _copy_values_from_csv_converter(self):
        self.images_copied = self.csv_converter.images_copied
        self.images_split = self.csv_converter.images_split

        self.images = self.csv_converter.images
        self.label = self.csv_converter.label

        self.image_sets = self.csv_converter.image_sets

    def _create_label_map_pbtxt(self):
        label_map_file = os.path.join(self.output_path, 'label_map.pbtxt')
        item = 'item {{\n' \
               '  id: {}\n' \
               '  name: \'{}\'\n' \
               '}}\n' \
               '\n'

        with open(label_map_file, 'w') as f:
            for cat_id in self.id2cat:
                f.write(item.format(cat_id, self.id2cat[cat_id]))
        print('\nCreated label map file.')
