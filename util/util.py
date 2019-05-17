import json
import os
import time

import pandas as pd
from tabulate import tabulate


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

    return path


def validate_match(image_sets, images, label):
    valid = True

    for s in image_sets:
        image_list = [file[:-4] for file in images[s]]
        label_list = [file[:-4] for file in label[s]]

        valid = valid and sorted(image_list) == sorted(label_list)

    return valid


def print_label_stats(output_path, id2cat, excluded_classes, df, set_title, tablefmt):
    time.sleep(0.1)

    # General stats
    print('\nGeneral stats for \'{}\' set.'.format(set_title))

    df_general = _get_general_stats(df)
    df_general.to_csv(os.path.join(output_path, '{}_general_stats.csv'.format(set_title)), index=None)

    print(tabulate(df_general, headers='keys', tablefmt=tablefmt, showindex=False))

    # Class stats
    print('\nClass stats for \'{}\' set.'.format(set_title))

    df_class = _get_class_stats(id2cat, excluded_classes, df)
    df_class.to_csv(os.path.join(output_path, '{}_class_stats.csv'.format(set_title)), index=None)

    columns_to_print = ['class_id', 'class', 'examples',
                        'bbox_area_tiny', 'fraction_tiny_bbox_%',
                        'bbox_area_small', 'fraction_small_bbox_%',
                        'bbox_area_medium', 'fraction_medium_bbox_%',
                        'bbox_area_large', 'fraction_large_bbox_%',
                        'avg_x_center', 'avg_y_center', 'avg_bbox_w', 'avg_bbox_h']

    print(tabulate(df_class[columns_to_print], headers='keys', tablefmt=tablefmt, showindex=False, floatfmt=".2f"))


def _get_general_stats(df):
    general_stats = ['images',
                     'avg. width', 'min. width', 'max. width',
                     'avg. height', 'min. height', 'max. height']

    data = [(len(df['filename'].unique()),
             df['width'].mean(), df['width'].min(), df['width'].max(),
             df['height'].mean(), df['height'].min(), df['height'].max())]

    return pd.DataFrame(data, columns=general_stats)


def _get_class_stats(id2cat, excluded_classes, df):
    class_stats = ['class_id', 'class', 'examples',
                   'bbox_area_tiny', 'fraction_tiny_bbox_%',
                   'bbox_area_small', 'fraction_small_bbox_%',
                   'bbox_area_medium', 'fraction_medium_bbox_%',
                   'bbox_area_large', 'fraction_large_bbox_%',
                   'avg_xmin', 'avg_ymin', 'avg_xmax', 'avg_ymax',
                   'min_x_center', 'min_y_center', 'min_bbox_w', 'min_bbox_h',
                   'avg_x_center', 'avg_y_center', 'avg_bbox_w', 'avg_bbox_h',
                   'max_x_center', 'max_y_center', 'max_bbox_w', 'max_bbox_h',
                   'avg_bbox_area', 'min_bbox_area', 'max_bbox_area',
                   'avg_rel_x_center', 'avg_rel_y_center', 'avg_rel_bbox_w', 'avg_rel_bbox_h']

    class_list = []

    # TODO
    #  - über range(classes) iterieren
    #  - wenn class_id nicht in df['class'].unique() leere zeile mit nur id und namen zurück geben

    unique = df['class'].unique()

    for class_id in range(1, len(id2cat) + 1):
        if class_id in excluded_classes:
            continue

        if class_id not in unique:
            row = (class_id, id2cat[class_id], 0,)
            row += tuple(-1 for _ in range(len(class_stats) - 3))

            class_list.append(row)
            continue

        filtered_df = df[df['class'] == class_id]
        class_name = id2cat[class_id]
        examples = filtered_df['class'].count()

        # Convert to center values
        x_center = filtered_df['xmax'] - (filtered_df['xmax'] - filtered_df['xmin']) / 2
        y_center = filtered_df['ymax'] - (filtered_df['ymax'] - filtered_df['ymin']) / 2
        bbox_w = filtered_df['xmax'] - filtered_df['xmin']
        bbox_h = filtered_df['ymax'] - filtered_df['ymin']

        # Convert to relative values
        rel_x_center = x_center / filtered_df['width'].mean()
        rel_y_center = y_center / filtered_df['height'].mean()
        rel_bbox_w = bbox_w / filtered_df['width'].mean()
        rel_bbox_h = bbox_h / filtered_df['height'].mean()

        avg_xmin = filtered_df['xmin'].mean()
        avg_ymin = filtered_df['ymin'].mean()
        avg_xmax = filtered_df['xmax'].mean()
        avg_ymax = filtered_df['ymax'].mean()

        min_x_center = x_center.min()
        min_y_center = y_center.min()
        min_bbox_width = bbox_w.min()
        min_bbox_height = bbox_h.min()

        avg_x_center = x_center.mean()
        avg_y_center = y_center.mean()
        avg_bbox_width = bbox_w.mean()
        avg_bbox_height = bbox_h.mean()

        max_x_center = x_center.max()
        max_y_center = y_center.max()
        max_bbox_width = bbox_w.max()
        max_bbox_height = bbox_h.max()

        bbox_area = pd.DataFrame({'bbox_w': bbox_w, 'bbox_h': bbox_h, 'area': bbox_w * bbox_h})
        avg_bbox_area = bbox_area['area'].mean()
        min_bbox_area = bbox_area['area'].min()
        max_bbox_area = bbox_area['area'].max()

        bbox_area_tiny = bbox_area['area'][(bbox_area['area'] <= 16 * 16)].count()
        bbox_area_small = bbox_area['area'][(bbox_area['area'] > 16 * 16) & (bbox_area['area'] <= 32 * 32)].count()
        bbox_area_medium = bbox_area['area'][(bbox_area['area'] > 32 * 32) & (bbox_area['area'] <= 96 * 96)].count()
        bbox_area_large = bbox_area['area'][(bbox_area['area'] > 96 * 96)].count()

        fraction_tiny_bbox = bbox_area_tiny / examples * 100
        fraction_small_bbox = bbox_area_small / examples * 100
        fraction_medium_bbox = bbox_area_medium / examples * 100
        fraction_large_bbox = bbox_area_large / examples * 100

        avg_rel_x_center = rel_x_center.mean() * 100
        avg_rel_y_center = rel_y_center.mean() * 100
        avg_rel_bbox_w = rel_bbox_w.mean() * 100
        avg_rel_bbox_h = rel_bbox_h.mean() * 100

        class_list.append((class_id, class_name, examples,
                           bbox_area_tiny, fraction_tiny_bbox,
                           bbox_area_small, fraction_small_bbox,
                           bbox_area_medium, fraction_medium_bbox,
                           bbox_area_large, fraction_large_bbox,
                           avg_xmin, avg_ymin, avg_xmax, avg_ymax,
                           min_x_center, min_y_center, min_bbox_width, min_bbox_height,
                           avg_x_center, avg_y_center, avg_bbox_width, avg_bbox_height,
                           max_x_center, max_y_center, max_bbox_width, max_bbox_height,
                           avg_bbox_area, min_bbox_area, max_bbox_area,
                           avg_rel_x_center, avg_rel_y_center, avg_rel_bbox_w, avg_rel_bbox_h))

    return pd.DataFrame(class_list, columns=class_stats)


def print_warning_for_empty_classes(data):
    emtpy_classes = []

    for i, num_bbox in enumerate(data['#bbox']):
        if num_bbox == 0:
            emtpy_classes.append(str(data['class id'][i]))

    if len(emtpy_classes) > 0:
        print('Recommended to exclude the following class ids with 0 bboxes: {}'.format(' '.join(emtpy_classes)))


def warning_not_verified_label_files(not_verified_label_files):
    if len(not_verified_label_files) > 0:
        print('\nNot verified label files found in folder {}.'.format(
            os.path.dirname(not_verified_label_files[0])))

        print('\tFile list:')
        for label_file in not_verified_label_files:
            print('\t' + os.path.basename(label_file))


def write_label_map(output_path, categories, label_id_mapping):
    print('Saving label map and id mapping to output folder.')

    label_map_file = os.path.join(output_path, 'label_map.json')
    with open(label_map_file, 'w') as f:
        json.dump({'classes': categories}, f, sort_keys=True, indent=4)

    label_id_mapping_file = os.path.join(output_path, 'label_id_mapping.json')
    with open(label_id_mapping_file, 'w') as f:
        json.dump({'old_id_to_new_id': label_id_mapping}, f, sort_keys=True, indent=4)


def check_label_names_for_duplicates(categories):
    found_duplicates = False

    cat_names = []
    for cat in categories:
        if cat['name'] not in cat_names:
            cat_names.append(cat['name'])
        else:
            found_duplicates = True

            print(cat['name'], 'exists more than once with IDs:')
            for c in categories:
                if c['name'] == cat['name']:
                    print(c['id'])

    return found_duplicates
