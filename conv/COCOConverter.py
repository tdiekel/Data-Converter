import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from pycocotools import coco as cocoapi
from tqdm import tqdm

import conv
from conv.util import create_dir, warning_not_verified_label_files, check_label_names_for_duplicates


class COCOConverter(conv.BaseConverter):
    def __init__(self, args):
        super().__init__(args)

        self.licenses = [{'id': 1,
                          'name': 'IfF',
                          'url': 'http://www.iff.tu-bs.de'
                          }]

        if 'supercategory' not in self.categories[0]:
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
        create_dir(annotations_dir)

        for image_set in self.image_sets:
            time.sleep(0.1)
            print("\tCreating {} set...".format(image_set))
            time.sleep(0.1)

            # Make image_set output dir
            image_set_dir = os.path.join(self.output_path, image_set)
            create_dir(image_set_dir)

            images, annotations = self._get_images_and_annotations(image_set)

            json_data = {
                "info": self.info,
                "licenses": self.licenses,
                "images": images,
                "annotations": annotations,
                "categories": self.categories
            }

            time.sleep(0.1)
            print('\tWriting annotations to disk...\n')
            time.sleep(0.1)

            annotation_file = os.path.join(self.output_path, "annotations", "instances_" + image_set + ".json")
            with open(annotation_file, "w") as jsonfile:
                json.dump(json_data, jsonfile, indent=4)

        for image_set in self.image_sets:
            print('\nTesting dataset {} ...'.format(image_set))

            annotation_file = os.path.join(self.output_path, "annotations", "instances_" + image_set + ".json")
            self._test_dataset(annotation_file)

        warning_not_verified_label_files(self.not_verified_label_files)

    @staticmethod
    def _test_dataset(annotation_file):
        coco = cocoapi.COCO(annotation_file)

        # Load categories from annotation file
        cats = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
        classes = ['__background__'] + cats
        num_classes = len(classes)
        _class_to_ind = dict(zip(classes, range(num_classes)))
        _class_to_coco_ind = dict(zip(cats, coco.getCatIds()))
        _coco_ind_to_class_ind = dict([(_class_to_coco_ind[cls], _class_to_ind[cls])
                                       for cls in classes[1:]])

    def _get_images_and_annotations(self, image_set):
        images = []
        annotations = []

        for image_id, image_filename in enumerate(tqdm(self.images[image_set], desc='\tProgress', unit='files')):
            label_path = os.path.join(self.label_path, image_filename.replace('.' + self.image_src_filetype, '.xml'))
            assert os.path.isfile(label_path), "File not found: {}".format(label_path)

            annotation_list, im_width, im_height = self._get_annotations(image_set=image_set, image_id=image_id,
                                                                         label_path=label_path)
            for annotation in annotation_list:
                annotations.append(annotation)

            if not self.images_copied:
                if self.skip_images_without_label:
                    if len(annotation_list) > 0:
                        self._save_image(image_id, image_set)
                else:
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
            self.not_verified_label_files.append(label_path)

        annotation_list = []
        width = int(xml_tree.find('size')[0].text)
        height = int(xml_tree.find('size')[1].text)

        for member in xml_tree.findall('object'):
            category_id = int(member[0].text) + 1

            if category_id in self.excluded_classes:
                continue

            if category_id not in self.included_ids:
                print(
                    'Error: Class ID {} not in label map or not included. Found in label file: {}'.format(
                        str(category_id), label_path))
                sys.exit(-1)

            if category_id in self.label_id_mapping:
                category_id = self.label_id_mapping[category_id]

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
