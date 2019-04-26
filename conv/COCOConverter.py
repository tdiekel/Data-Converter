import json
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
from pycocotools import coco as cocoapi
from tqdm import tqdm

import conv


class COCOConverter(conv.BaseConverter):
    def __init__(self, image_path, image_src_type, image_dest_type, label_path, label_map, file_lists, output_path,
                 excluded_classes, included_classes):
        super().__init__(image_path, image_src_type, image_dest_type, label_path, label_map, file_lists,
                         output_path, excluded_classes, included_classes)

        self.licenses = [{'id': 1,
                          'name': 'IfF',
                          'url': 'http://www.iff.tu-bs.de'
                          }]

        for item in self.categories:
            name = item.get('name')
            idx = name.rfind('(')
            item['supercategory'] = name[idx + 1:-1]

        self.annotation_id = 1

    def convert(self):
        time.sleep(0.1)
        print("\nCreating dataset...")

        # Make annotations output dir
        annotations_dir = os.path.join(self.output_path, "annotations")
        self._create_dir(annotations_dir)

        for image_set in self.image_sets:
            time.sleep(0.1)
            print("\tCreating {} set...".format(image_set))
            time.sleep(0.1)

            # Make image_set output dir
            image_set_dir = os.path.join(self.output_path, image_set)
            self._create_dir(image_set_dir)

            images, annotations = self._get_images_and_annotations(image_set)

            json_data = {
                "info": self.info,
                "licenses": self.licenses,
                "images": images,
                "annotations": annotations,
                "categories": self.categories
            }

            annotation_file = os.path.join(self.output_path, "annotations", "instances_" + image_set + ".json")
            with open(annotation_file, "w") as jsonfile:
                json.dump(json_data, jsonfile, sort_keys=True, indent=4)

        for image_set in self.image_sets:
            print('\nTesting dataset {} ...'.format(image_set))

            annotation_file = os.path.join(self.output_path, "annotations", "instances_" + image_set + ".json")
            self._test_dataset(annotation_file)

    def _test_dataset(self, annotation_file):
        c = cocoapi.COCO(annotation_file)

    def _get_images_and_annotations(self, image_set):
        images = []
        annotations = []

        for image_id, image_filename in enumerate(tqdm(self.images[image_set], desc='\tProgress', unit='files')):
            label_path = os.path.join(self.label_path, image_filename.replace('.' + self.image_src_type, '.xml'))
            assert os.path.isfile(label_path), "File not found: {}".format(label_path)

            annotation_list, im_width, im_height = self._get_annotations(image_set=image_set, image_id=image_id,
                                                                         label_path=label_path)
            for annotation in annotation_list:
                annotations.append(annotation)

            if not self.images_copied:
                self._save_image(image_id, image_set)

            images.append({
                "license": 1,
                "file_name": image_filename,
                "height": im_height,
                "width": im_width,
                "id": image_id + 1
            })

        return images, annotations

    def _get_annotations(self, image_set, image_id, label_path):
        xml_tree = ET.parse(label_path).getroot()

        if "verified" not in xml_tree.attrib:
            print("Label file not verified: {}".format(label_path))

        annotation_list = []
        width = int(xml_tree.find('size')[0].text)
        height = int(xml_tree.find('size')[1].text)

        for member in xml_tree.findall('object'):
            category_id = int(member[0].text) + 1

            if category_id in self.excluded_classes:
                continue

            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)

            bbox = [xmin, ymax, xmax - xmin, ymax - ymin]
            area = np.float(bbox[2] * bbox[3])

            annotation_list.append({
                # https://github.com/facebookresearch/Detectron/issues/48#issuecomment-361028870
                "segmentation": [],
                "area": area,
                "iscrowd": 0,
                "image_id": image_id + 1,
                "bbox": bbox,
                "category_id": category_id,
                "id": self.annotation_id
            })

            if category_id in self.gt_boxes:
                if image_set in self.gt_boxes[category_id]['num_gt_boxes']:
                    self.gt_boxes[category_id]['num_gt_boxes'][image_set] += 1
                else:
                    self.gt_boxes[category_id]['num_gt_boxes'][image_set] = 1

            self.annotation_id += 1

        return annotation_list, width, height
