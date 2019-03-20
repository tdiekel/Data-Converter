"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import os
import sys
import time
from collections import namedtuple

import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# Check if running from tf folder, else append sys path
tf_research_path = os.path.join(os.environ['TENSORFLOW_PATH'], 'models', 'research')
if not os.getcwd() == tf_research_path:
    sys.path.append(tf_research_path)

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('label_path', '', 'Path to the json label map')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# # TO-DO replace this with label map
# def class_text_to_int(row_label):
#     if row_label == "speed limit 20 (prohibitory)": return 1
#     elif row_label == "speed limit 30 (prohibitory)": return 2
#     elif row_label == "speed limit 50 (prohibitory)": return 3
#     elif row_label == "speed limit 60 (prohibitory)": return 4
#     elif row_label == "speed limit 70 (prohibitory)": return 5
#     elif row_label == "speed limit 80 (prohibitory)": return 6
#     elif row_label == "restriction ends 80 (other)": return 7
#     elif row_label == "speed limit 100 (prohibitory)": return 8
#     elif row_label == "speed limit 120 (prohibitory)": return 9
#     elif row_label == "no overtaking (prohibitory)": return 10
#     elif row_label == "no overtaking (trucks) (prohibitory)": return 11
#     elif row_label == "priority at next intersection (danger)": return 12
#     elif row_label == "priority road (other)": return 13
#     elif row_label == "give way (other)": return 14
#     elif row_label == "stop (other)": return 15
#     elif row_label == "no traffic both ways (prohibitory)": return 16
#     elif row_label == "no trucks (prohibitory)": return 17
#     elif row_label == "no entry (other)": return 18
#     elif row_label == "danger (danger)": return 19
#     elif row_label == "bend left (danger)": return 20
#     elif row_label == "bend right (danger)": return 21
#     elif row_label == "bend (danger)": return 22
#     elif row_label == "uneven road (danger)": return 23
#     elif row_label == "slippery road (danger)": return 24
#     elif row_label == "road narrows (danger)": return 25
#     elif row_label == "construction (danger)": return 26
#     elif row_label == "traffic signal (danger)": return 27
#     elif row_label == "pedestrian crossing (danger)": return 28
#     elif row_label == "school crossing (danger)": return 29
#     elif row_label == "cycles crossing (danger)": return 30
#     elif row_label == "snow (danger)": return 31
#     elif row_label == "animals (danger)": return 32
#     elif row_label == "restriction ends (other)": return 33
#     elif row_label == "go right (mandatory)": return 34
#     elif row_label == "go left (mandatory)": return 35
#     elif row_label == "go straight (mandatory)": return 36
#     elif row_label == "go right or straight (mandatory)": return 37
#     elif row_label == "go left or straight (mandatory)": return 38
#     elif row_label == "keep right (mandatory)": return 39
#     elif row_label == "keep left (mandatory)": return 40
#     elif row_label == "roundabout (mandatory)": return 41
#     elif row_label == "restriction ends (overtaking) (other)": return 42
#     elif row_label == "restriction ends (overtaking (trucks)) (other)": return 43
#     elif row_label == "restriction ends 60 (other)": return 44
#     elif row_label == "restriction ends 70 (other)": return 45
#     elif row_label == "speed limit 90 (prohibitory)": return 46
#     elif row_label == "restriction ends 90 (other)": return 47
#     elif row_label == "speed limit 110 (prohibitory)": return 48
#     elif row_label == "restriction ends 110 (other)": return 49
#     elif row_label == "restriction ends 120 (other)": return 50
#     elif row_label == "speed limit 130 (prohibitory)": return 51
#     elif row_label == "restriction ends 130 (other)": return 52
#     elif row_label == "bend double right (danger)": return 53
#     elif row_label == "highway turn (left) (other)": return 54
#     elif row_label == "maximum width (prohibitory)": return 55
#     elif row_label == "maximum height (prohibitory)": return 56
#     elif row_label == "minimum truck distance (prohibitory)": return 57
#     elif row_label == "highway exit 200 (other)": return 58
#     elif row_label == "highway exit 100 (other)": return 59
#     elif row_label == "right lane merging (other)": return 60
#     elif row_label == "warning beacon roadwork (other)": return 61
#     elif row_label == "speed limit 60 (digital) (prohibitory)": return 62
#     elif row_label == "restriction ends 60 (digital) (other)": return 63
#     elif row_label == "speed limit 70 (digital) (prohibitory)": return 64
#     elif row_label == "restriction ends 70 (digital) (other)": return 65
#     elif row_label == "speed limit 80 (digital) (prohibitory)": return 66
#     elif row_label == "restriction ends 80 (digital) (other)": return 67
#     elif row_label == "restriction ends 80 (digital) (other)": return 68
#     elif row_label == "restriction ends 90 (digital) (other)": return 69
#     elif row_label == "speed limit 100 (digital) (prohibitory)": return 70
#     elif row_label == "restriction ends 100 (digital) (other)": return 71
#     elif row_label == "speed limit 110 (digital) (prohibitory)": return 72
#     elif row_label == "restriction ends 110 (digital) (other)": return 73
#     elif row_label == "left lane merging (other)": return 74
#     elif row_label == "speed limit 120 (digital) (prohibitory)": return 75
#     elif row_label == "restriction ends 120 (digital) (other)": return 76
#     elif row_label == "speed limit 130 (digital) (prohibitory)": return 77
#     elif row_label == "restriction ends 130 (digital) (other)": return 78
#     elif row_label == "no overtaking (digital) (prohibitory)": return 79
#     elif row_label == "restriction ends 130 (digital) (other)": return 80
#     elif row_label == "no overtaking (trucks) (digital) (prohibitory)": return 81
#     elif row_label == "restriction ends (overtaking (trucks)) (digital) (other)": return 82
#     elif row_label == "construction (digital) (danger)": return 83
#     elif row_label == "traffic jam (digital) (danger)": return 84
#     elif row_label == "highway exit (other)": return 85
#     elif row_label == "traffic jam (other)": return 86
#     elif row_label == "restriction distance (other)": return 87
#     elif row_label == "restriction time (other)": return 88
#     elif row_label == "highway exit 300m (other)": return 89
#     elif row_label == "restriction ends 100 (other)": return 90
#     elif row_label == "andreaskreuz (other)": return 91
#     elif row_label == "one way street (left) (other)": return 92
#     elif row_label == "one way street (right) (other)": return 93
#     elif row_label == "beginning of highway (other)": return 94
#     elif row_label == "end of highway (other)": return 95
#     elif row_label == "busstop (other)": return 96
#     elif row_label == "tunnel (other)": return 97
#     elif row_label == "no cars (prohibitory)": return 98
#     elif row_label == "train crossing (danger)": return 99
#     elif row_label == "no bicycles (prohibitory)": return 100
#     elif row_label == "no motorbikes (prohibitory)": return 101
#     elif row_label == "no mopeds (prohibitory)": return 102
#     elif row_label == "no horses (prohibitory)": return 103
#     elif row_label == "no cars & motorbikes (prohibitory)": return 104
#     elif row_label == "busses only (mandatory)": return 105
#     elif row_label == "pedestrian zone (mandatory)": return 106
#     elif row_label == "bicycle boulevard (mandatory)": return 107
#     elif row_label == "end of bicycle boulevard (mandatory)": return 108
#     elif row_label == "bicycle path (mandatory)": return 109
#     elif row_label == "pedestrian path (mandatory)": return 110
#     elif row_label == "pedestrian and bicycle path (mandatory)": return 111
#     elif row_label == "separated path for bicycles and and pedestrians (right) (mandatory)": return 112
#     elif row_label == "separated path for bicycles and and pedestrians (left) (mandatory)": return 113
#     elif row_label == "play street (other)": return 114
#     elif row_label == "end of play street (other)": return 115
#     elif row_label == "beginning of motorway (other)": return 116
#     elif row_label == "end of motorway (other)": return 117
#     elif row_label == "crosswalk (zebra) (other)": return 118
#     elif row_label == "dead-end street (other)": return 119
#     elif row_label == "one way street (straight) (other)": return 120
#     elif row_label == "priority road (other)": return 121
#     elif row_label == "no stopping (prohibitory)": return 122
#     elif row_label == "no stopping (beginning) (prohibitory)": return 123
#     elif row_label == "no stopping (middle) (prohibitory)": return 124
#     elif row_label == "no stopping (end) (prohibitory)": return 125
#     elif row_label == "no parking (beginning) (prohibitory)": return 126
#     elif row_label == "no parking (end) (prohibitory)": return 127
#     elif row_label == "no parking (middle) (prohibitory)": return 128
#     elif row_label == "no parking (prohibitory)": return 129
#     elif row_label == "no parking zone (prohibitory)": return 130
#     elif row_label == "end of no parking zone (prohibitory)": return 131
#     elif row_label == "city limit (in) (other)": return 132
#     elif row_label == "city limit (out) (other)": return 133
#     elif row_label == "direction to village (other)": return 134
#     elif row_label == "rural road exit (other)": return 135
#     elif row_label == "speed limit 20 zone (prohibitory)": return 136
#     elif row_label == "end speed limit 20 zone (prohibitory)": return 137
#     elif row_label == "speed limit 30 zone (prohibitory)": return 138
#     elif row_label == "end speed limit 30 zone (prohibitory)": return 139
#     elif row_label == "speed limit 5 (prohibitory)": return 140
#     elif row_label == "speed limit 10 (prohibitory)": return 141
#     elif row_label == "restriction ends 10 (other)": return 142
#     elif row_label == "restriction ends 20 (other)": return 143
#     elif row_label == "restriction ends 30 (other)": return 144
#     elif row_label == "speed limit 40 (prohibitory)": return 145
#     elif row_label == "restriction ends 40 (other)": return 146
#     elif row_label == "restriction ends 50 (other)": return 147
#     elif row_label == "go left (now) (mandatory)": return 148
#     elif row_label == "go right (now) (mandatory)": return 149
#     elif row_label == "train crossing in 300m (other)": return 150
#     elif row_label == "train crossing in 200m (other)": return 151
#     elif row_label == "train crossing in 100m (other)": return 152
#     elif row_label == "danger (digital) (danger)": return 153
#     elif row_label == "restriction ends 100 (other)": return 154
#     elif row_label == "highway turn (right) (other)": return 155
#     else:
#         return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, label_dict, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(int(row['xmin']) / width)
        xmaxs.append(int(row['xmax']) / width)
        ymins.append(int(row['ymin']) / height)
        ymaxs.append(int(row['ymax']) / height)
        classes_text.append(_get_class_text(label_dict, row['class']))
        classes.append(row['class'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def _get_class_text(label_dict, label_id):
    default = -1

    return_value = label_dict.get(label_id, default)

    if return_value == default:
        UserWarning('Error: Class ID {} not found in label dict!'.format(label_id))
        sys.exit(-1)

    return return_value.encode('utf8')


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_path = FLAGS.label_path
    categories = json.load(open(label_path, 'r')).get('classes')
    id2cat = {cat['id']: cat['name'] for cat in categories}

    image_dir = FLAGS.image_dir

    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, id2cat, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


def generate_tfrecord(image_dir, csv_input, id2cat, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)

    print('\t\tReading csv file ...')
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')

    print('\t\tWriting record file ...')
    for group in tqdm(grouped, desc='\t\tProgress:', unit='files'):
        tf_example = create_tf_example(group, id2cat, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)

    time.sleep(0.1)
    print('\t\tSuccessfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
