"""
Mapping settings for labels
"""

'''Example 1 - Combine by ID'''
# <Int_ID>: {                            # <Int_ID> is ID of mapping setting
#            'type': 'combine_by_id',
#            'ids_from_org_list': False, # Will assume Label IDs are starting at 0 if true, else start from 1.
#            'new_labels': [             # List of dicts with information for new label
#                           {
#                            'new_id': <Int>,
#                            'new_name': <String>,
#                            'old_id': <List_Of_Int(s)> # Single ID for renaming, multiple for merging
#                           },
#                          ]
#           }

'''Example 2 - Combine by Substring'''
# <Int_ID>: {                            # <Int_ID> is ID of mapping setting
#            'type': 'combine_by_substring',
#            'new_labels': [             # List of dicts with information for new label
#                           {
#                            'new_id': <Int>,                   # Optional, when not given will append at end
#                            'new_name': <String>,
#                            'substring': <String>          # All label containing <String> will be merged
#                            'exclude': <String>            # Optional, exclude all containing <String>
#                           },
#                          ]
#           }


mapping_settings = {
    # INFO Default setting
    1: {
        'type': 'combine_by_id',
        'ids_from_org_list': False,
        'new_labels': [
            {
                'new_name': 'restriction ends 100 (other)',
                'new_id': 90,
                'old_id': [154]

            },
        ]
    },
    # INFO Supercategories by name
    2: {
        'type': 'combine_by_substring',
        'new_labels': [
            {
                'new_id': 1,
                'new_name': 'traffic_signs_prohibitory',
                'substring': '(prohibitory)',
                'exclude': '(digital)'
            },
            {
                'new_id': 2,
                'new_name': 'traffic_signs_danger',
                'substring': '(danger)',
                'exclude': '(digital)'
            },
            {
                'new_id': 3,
                'new_name': 'traffic_signs_mandatory',
                'substring': '(mandatory)',
                'exclude': '(digital)'
            },
            {
                'new_id': 4,
                'new_name': 'traffic_signs_other',
                'substring': '(other)',
                'exclude': '(digital)'
            },
            {
                'new_id': 5,
                'new_name': 'traffic_signs_digital',
                'substring': '(digital)'
            },
            {
                'new_id': 6,
                'new_name': 'traffic_signals',
                'substring': '(traffic signal)'
            },
            {
                'new_id': 7,
                'new_name': 'object',
                'substring': '(object)'
            },
            {
                'new_id': 8,
                'new_name': 'structure',
                'substring': '(structure)'
            },
            {
                'new_id': 9,
                'new_name': 'roadsurface',
                'substring': '(roadsurface)'
            },
        ]
    },
    # INFO Merging relevant classes
    3: {
        'type': 'combine_by_id',
        'ids_from_org_list': False,
        'new_labels': [
            {
                'new_name': 'pedestrian crossing (danger)',
                'new_id': 28,
                'old_id': [28, 29]
            },
            {
                'new_name': 'bend double (danger)',
                'new_id': 22,
                'old_id': [22, 53]
            },
            {
                'new_name': 'highway turn (other)',
                'new_id': 54,
                'old_id': [54, 155]
            },
            {
                'new_name': 'lane merging (other)',
                'new_id': 60,
                'old_id': [74, 60]
            },
            {
                'new_name': 'one way street (other)',
                'new_id': 92,
                'old_id': [92, 93]
            },
            {
                'new_name': 'no stopping (prohibitory)',
                'new_id': 122,
                'old_id': [122, 123, 124, 125]
            },
            {
                'new_name': 'no parking (prohibitory)',
                'new_id': 126,
                'old_id': [129, 126, 128, 127, 130]
            },
            {
                'new_name': 'red (traffic signal)',
                'new_id': 159,
                'old_id': [159, 163, 167, 171]
            },
            {
                'new_name': 'green (traffic signal)',
                'new_id': 160,
                'old_id': [160, 164, 168, 172]
            },
            {
                'new_name': 'yellow (traffic signal)',
                'new_id': 161,
                'old_id': [161, 165, 169, 173]
            },
            {
                'new_name': 'red and yellow (traffic signal)',
                'new_id': 158,
                'old_id': [158, 162, 166, 170]
            },
            {
                'new_name': 'truck (object)',
                'new_id': 178,
                'old_id': [178, 179]
            },
            {
                'new_name': 'transporter (object)',
                'new_id': 180,
                'old_id': [180, 182]
            },
            {
                'new_name': 'motorbike (object)',
                'new_id': 186,
                'old_id': [186, 189]
            },
            {
                'new_name': 'arrow straight (roadsurface)',
                'new_id': 193,
                'old_id': [193, 198]
            },
            {
                'new_name': 'arrow straight and turn (roadsurface)',
                'new_id': 199,
                'old_id': [199, 200]
            },
            # {
            #     'new_name': 'restriction ends 100 (other)',
            #     'new_id': 90,
            #     'old_id': [90, 154]
            # },
        ]
    },
    # INFO Supercategories by id
    4: {
        'type': 'combine_by_id',
        'ids_from_org_list': False,
        'new_labels': [
            {
                'new_id': 1,
                'new_name': 'traffic_signs_prohibitory',
                'old_id': [1, 46, 48, 51, 55, 56, 57, 98, 100, 101, 102, 103, 104, 131, 136, 137, 138, 139, 140, 141,
                           176, 177, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16, 17, 122, 123, 124, 125, 126, 127, 128, 129, 130,
                           145]
            },
            {
                'new_id': 2,
                'new_name': 'traffic_signs_danger',
                'old_id': [12, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 53, 30, 31, 86, 99]
            },
            {
                'new_id': 3,
                'new_name': 'traffic_signs_mandatory',
                'old_id': [34, 35, 36, 37, 38, 39, 41, 109, 110, 111, 112, 40, 105, 106, 107, 108, 113, 148, 149]
            },
            {
                'new_id': 4,
                'new_name': 'traffic_signs_other_restriction_ends',
                'old_id': [7, 33, 42, 43, 44, 45, 47, 49, 50, 52, 90, 142, 143, 144, 156, 147]
            },
            {
                'new_id': 5,
                'new_name': 'traffic_signs_other_direction',
                'old_id': [85, 134, 135]
            },
            {
                'new_id': 6,
                'new_name': 'traffic_signs_other_lane_restriction',
                'old_id': [54, 61, 155]
            },
            {
                'new_id': 7,
                'new_name': 'traffic_signs_other_exit_distance',
                'old_id': [58, 59, 89]
            },
            {
                'new_id': 8,
                'new_name': 'traffic_signs_other_beginning_or_end_of',
                'old_id': [94, 95, 114, 115, 116, 117, 132, 133]
            },
            {
                'new_id': 9,
                'new_name': 'traffic_signs_other',
                'old_id': [13, 14, 15, 18, 60, 74, 91, 92, 93, 96, 118, 119]
            },
            {
                'new_id': 10,
                'new_name': 'traffic_signs_digital',
                'old_id': [62, 66, 70, 75, 81, 63, 64, 65, 67, 68, 69, 71, 72, 73, 76, 77, 78, 79, 80, 82, 83, 84, 153]
            },
            {
                'new_id': 11,
                'new_name': 'roadsurface',
                'old_id': [193, 194, 195, 198, 199, 200]
            },
            {
                'new_id': 12,
                'new_name': 'traffic_signals',
                'old_id': [158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173]
            },
            {
                'new_id': 13,
                'new_name': 'vehicle',
                'old_id': [178, 179, 180, 181, 182, 183]
            },
            {
                'new_id': 14,
                'new_name': 'two_wheeler',
                'old_id': [186, 187, 189]
            },
            {
                'new_id': 15,
                'new_name': 'pedestrian',
                'old_id': [188]
            },
        ]
    },
}
