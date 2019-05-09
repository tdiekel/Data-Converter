mapping_settings = {
    1: {
        'combine_by_id': True,
        'ids_from_org_list': False,  # Default: True
        # Example
        # 'new_labels': [
        #         {
        #             'new_id': 1,
        #             'new_name': 'traffic_signs', # Optional, leave empty or delete when old name should be kept
        #             'old_id': int() or list()
        #         },
        #     ]
        'new_labels': [
            {
                'new_id': 90,
                'old_id': 154

            },
        ]
    },
    2: {
        'combine_by_substring': True,
        # Example
        # 'new_labels': [
        #     {
        #         'new_id': 1,
        #         'new_name': 'traffic_signs_prohibitory',
        #         'substring': 'prohibitory',
        #         'exclude': 'digital' # Optional, leave empty or delete when not needed
        #     },
        # ]
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
    }
}
