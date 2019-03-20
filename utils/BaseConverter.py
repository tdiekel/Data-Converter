import json
import os
import random
import shutil
import time
from datetime import datetime

from PIL import Image
from tqdm import tqdm


class BaseConverter:
    def __init__(self, image_path, image_src_type, image_dest_type, label_path, label_map, output_path):
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
        self.output_path = output_path

        self.images_copied = False
        self.images_split = False

        self.categories = json.load(open(label_map, 'r')).get('classes')
        self.cat2id = {cat['name']: cat['id'] for cat in self.categories}
        self.id2cat = {cat['id']: cat['name'] for cat in self.categories}

        # Get max id
        self.max_id = -1
        for cat_id in self.id2cat:
            if cat_id > self.max_id:
                self.max_id = cat_id

        self.images = {}
        self.label = {}
        self._fill_lists()

        assert self._validate_match(), 'Image and label files do not match.'

        self.image_sets = ["images"]

    def _fill_lists(self):
        self.images = {'images': [file for file in os.listdir(self.image_path) if file.endswith(self.image_src_type)]}
        self.label = {'label': [file for file in os.listdir(self.label_path) if file.endswith('.xml')]}

    def _validate_match(self):
        image_list = [file[:-4] for file in self.images['images']]
        label_list = [file[:-4] for file in self.label['label']]

        return sorted(image_list) == sorted(label_list)

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
                self.label[s].append(self.label['label'].pop())

                if not self.images_copied:
                    self._save_image(i, s)

        self.images.pop('images')
        self.label.pop('label')
        self.image_sets.remove('images')

        self.images_copied = True
        self.images_split = True
        self.image_src_type = self.image_dest_type

    def _shuffle(self):
        zipped = list(zip(self.images['images'], self.label['label']))

        random.shuffle(zipped)

        images, label = zip(*zipped)
        self.images['images'], self.label['label'] = list(images), list(label)

    def _save_image(self, idx, image_set):
        image_path = os.path.join(self.image_path, self.images[image_set][idx])

        if not self.image_src_type == self.image_dest_type:
            image = Image.open(image_path).convert('RGB')
            self.images[image_set][idx] = os.path.basename(image_path).replace('.' + self.image_src_type,
                                                                               '.' + self.image_dest_type)

            image_out_path = os.path.join(self.output_path, image_set, self.images[image_set][idx])
            image.save(image_out_path)
        else:
            image_out_path = os.path.join(self.output_path, image_set, os.path.basename(image_path))
            shutil.copyfile(image_path, image_out_path)
