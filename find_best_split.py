import os
from argparse import Namespace
from copy import deepcopy

from convert import main

# args = Namespace(dataset_name=None, exclude=[], exclude_area=None, exclude_starts_at_one=False, file_list_path=None,
#                  file_lists=None, image_dest_filetype='png',
#                  image_path='/home/osm/Schreibtisch/01_Datasets/2019_Juli/01_Rawdata/Images/', image_src_filetype='png',
#                  include=None, include_starts_at_one=False, label_map='./label_map.json',
#                  label_path='/home/osm/Schreibtisch/01_Datasets/2019_Juli/01_Rawdata/Labels/',
#                  mapping={'type': 'combine_by_id', 'ids_from_org_list': False,
#                           'new_labels': [{'new_name': 'restriction ends 100 (other)', 'new_id': 90, 'old_id': [154]}]},
#                  mapping_id=1, no_copy=True,
#                  output_path='/media/osm/ml/01_Datasets/Master_Diekel/00_split_70_30',
#                  rearrange_ids=False, rel_output_path=None, remap_labels=True, set_sizes=[70.0, 30.0],
#                  sets=['train', 'val'],
#                  show_not_verified=False, shuffle=True, skip_images_without_label=False, stats=False, stats_img=False,
#                  stats_label=False, tablefmt='psql', target_format='csv', year=2019)

args = Namespace(dataset_name=None,
                 exclude=[1, 7, 30, 31, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 57, 63, 64, 65, 67, 68, 69, 71, 72, 73,
                          76, 77, 78, 79, 80, 82, 83, 84, 86, 88, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
                          107, 108, 113, 114, 115, 117, 120, 121, 131, 136, 137, 138, 139, 140, 141, 142, 143, 144, 146,
                          147, 148, 149, 150, 151, 152, 153, 156, 176, 177, 185, 191, 196, 197, 201, 202, 204, 205, 206,
                          53, 29, 155, 74, 93, 123, 124, 125, 129, 128, 127, 130, 154, 162, 166, 170, 163, 167, 171,
                          164, 168, 172, 165, 169, 173, 179, 182, 189, 198, 200], exclude_area=256,
                 exclude_starts_at_one=True, file_list_path=None, file_lists=None, image_dest_filetype='png',
                 image_path='/home/osm/Schreibtisch/01_Datasets/2019_Juli/01_Rawdata/Images/', image_src_filetype='png',
                 include=None, include_starts_at_one=False, label_map='./label_map.json',
                 label_path='/home/osm/Schreibtisch/01_Datasets/2019_Juli/01_Rawdata/Labels/',
                 mapping={'type': 'combine_by_id', 'ids_from_org_list': False,
                          'new_labels': [{'new_name': 'pedestrian crossing (danger)', 'new_id': 28, 'old_id': [28, 29]},
                                         {'new_name': 'bend double (danger)', 'new_id': 22, 'old_id': [22, 53]},
                                         {'new_name': 'highway turn (other)', 'new_id': 54, 'old_id': [54, 155]},
                                         {'new_name': 'lane merging (other)', 'new_id': 60, 'old_id': [74, 60]},
                                         {'new_name': 'one way street (other)', 'new_id': 92, 'old_id': [92, 93]},
                                         {'new_name': 'no stopping (prohibitory)', 'new_id': 122,
                                          'old_id': [122, 123, 124, 125]},
                                         {'new_name': 'no parking (prohibitory)', 'new_id': 126,
                                          'old_id': [129, 126, 128, 127, 130]},
                                         {'new_name': 'red (traffic signal)', 'new_id': 159,
                                          'old_id': [159, 163, 167, 171]},
                                         {'new_name': 'green (traffic signal)', 'new_id': 160,
                                          'old_id': [160, 164, 168, 172]},
                                         {'new_name': 'yellow (traffic signal)', 'new_id': 161,
                                          'old_id': [161, 165, 169, 173]},
                                         {'new_name': 'red and yellow (traffic signal)', 'new_id': 158,
                                          'old_id': [158, 162, 166, 170]},
                                         {'new_name': 'truck (object)', 'new_id': 178, 'old_id': [178, 179]},
                                         {'new_name': 'transporter (object)', 'new_id': 180, 'old_id': [180, 182]},
                                         {'new_name': 'motorbike (object)', 'new_id': 186, 'old_id': [186, 189]},
                                         {'new_name': 'arrow straight (roadsurface)', 'new_id': 193,
                                          'old_id': [193, 198]},
                                         {'new_name': 'arrow straight and turn (roadsurface)', 'new_id': 199,
                                          'old_id': [199, 200]},
                                         {'new_name': 'restriction ends 100 (other)', 'new_id': 90,
                                          'old_id': [90, 154]}]}, mapping_id=3, no_copy=True,
                 output_path='/home/osm/Schreibtisch/01_Datasets/Master_Diekel/00_split_70_30_reduced',
                 rearrange_ids=False, rel_output_path=None, remap_labels=True, set_sizes=[70.0, 30.0],
                 sets=['train', 'val'], show_not_verified=False, shuffle=True, skip_images_without_label=False,
                 stats=False, stats_img=False, stats_label=False, tablefmt='psql', target_format='csv', year=2019)

current_min_delta = 15
dst_name = '00_split_70_30_reduced({})'

while True:
    max_delta = round(main(deepcopy(args)), 2)

    if max_delta >= 0 and max_delta < current_min_delta:
        current_min_delta = max_delta
        src = args.output_path
        dst = os.path.join(os.path.dirname(args.output_path), dst_name.format(max_delta))

        os.rename(src, dst)
