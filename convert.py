import argparse
import os
import sys

import converters
from label_mapping import mapping_settings


def check_args(args):
    valid_formats = ['png', 'jpg']

    assert os.path.isdir(args.image_path), 'Image dir not found at: {}'.format(args.image_path)

    assert (args.image_src_filetype in valid_formats
            ), 'Image source filetype not valid, please choose on of {}.'.format(valid_formats)

    if args.image_dest_filetype is not None:
        assert (args.image_dest_filetype in valid_formats
                ), 'Image destination filetype not valid, please choose on of {}.'.format(valid_formats)
    else:
        args.image_dest_filetype = args.image_src_filetype

    assert os.path.isdir(args.label_path), 'Label dir not found at: {}'.format(args.label_path)

    assert os.path.isfile(args.label_map), 'Label map file not found at: {}.'.format(args.label_map)

    if args.target_format == 'darknet':
        assert args.rel_output_path is not None, 'When using Darknet target format \"--rel-output-path\" have to be set.'
        assert args.dataset_name is not None, 'When using Darknet target format \"--dataset-name\" have to be set.'
        assert 1 <= len(args.sets) <= 2, 'When using Darknet target format \"--sets\" have to be between 1 and 2.'

    if args.file_list_path is not None:
        assert os.path.isdir(args.file_list_path), 'File list dir not found at: {}'.format(args.file_list_path)
        assert args.file_lists is not None, 'No file list received. At least one file list must be given.'

        for i, file_list in enumerate(args.file_lists):
            file_list_path = os.path.join(args.file_list_path, file_list)
            assert os.path.isfile(file_list_path), "File list not found at: {}".format(file_list_path)

            args.file_lists[i] = file_list_path

        assert not args.shuffle, "Shuffling is not possible when using file lists."
    else:
        assert (len(args.sets) == len(args.set_sizes)
                ), 'Number of set sizes does not fit the number of sets.'

    assert not (args.exclude is not None
                and args.include is not None), "Please don't use the flags --exclude and --include at the same time."

    if args.exclude is not None:
        exclude = []

        for item in args.exclude:
            if "-" in item:
                start, end = item.split('-')

                try:
                    for i in range(int(start), int(end) + 1):
                        exclude.append(i)
                except ValueError:
                    RuntimeError('ValueError while parsing IDs to exclude. Unknown value in: {}'.format(item))
            else:
                try:
                    exclude.append(int(item))
                except ValueError:
                    RuntimeError('ValueError while parsing IDs to exclude. Unknown value in: {}'.format(item))

        if not args.exclude_starts_at_one:
            exclude = [excluded_id + 1 for excluded_id in exclude]

        args.exclude = exclude
    else:
        args.exclude = []

    if args.include is not None:
        include = []

        for item in args.include:
            if "-" in item:
                start, end = item.split('-')

                try:
                    for i in range(int(start), int(end) + 1):
                        include.append(i)
                except ValueError:
                    RuntimeError('ValueError while parsing IDs to include. Unknown value in: {}'.format(item))
            else:
                try:
                    include.append(int(item))
                except ValueError:
                    RuntimeError('ValueError while parsing IDs to include. Unknown value in: {}'.format(item))

        if not args.include_starts_at_one:
            include = [included_id + 1 for included_id in include]

        args.include = include

    if args.remap_labels:
        assert args.mapping_type is not None, 'Please set a mapping type.'

        args.mapping = mapping_settings[args.mapping_type]

        if 'combine_by_id' in args.mapping and 'combine_by_substring' in args.mapping:
            assert args.mapping['combine_by_id'] is not args.mapping[
                'combine_by_substring'], 'Please check remap settings' \
                                         ' in \'label_mapping.py\' file.' \
                                         ' It\'s not possible to activate' \
                                         ' combine_by_substring and combine_by_id.'
        assert 'new_labels' in args.mapping, 'No new labels defined in \'label_mapping.py\' file.'

    if args.skip_images_without_label and args.target_format == 'coco':
        print('Skipping images without labels works only for COCO datsets.')

    if args.exclude_area is not None:
        assert args.exclude_area > 0, 'Area to exclude must be greater than 0.'

    return args


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser()

    # Path settings
    path_parser = parser.add_argument_group('Path settings')
    path_parser.add_argument('--image-path', help='Path to raw image directory.',
                             type=str, required=True)
    path_parser.add_argument('--label-path', help='Path to label directory.',
                             type=str, required=True)
    path_parser.add_argument('--output-path', help='Path to save converted dataset',
                             type=str, required=True)
    path_parser.add_argument('--label-map', help='Path to label map json file.',
                             type=str, default='./label_map.json')

    # Dataset settings
    dataset_parser = parser.add_argument_group('Dataset settings')
    dataset_parser.add_argument('--target-format', help='Format to save converted dataset in',
                                nargs='?', choices=['coco', 'csv', 'tfrecord', 'darknet'], required=True)
    dataset_parser.add_argument('--sets', help='List of subsets to create (e.g. "--sets train val").',
                                type=str, nargs='*', default=['train', 'val'])
    dataset_parser.add_argument('--set-sizes', help='Sizes of the subsets (e.g. "--sets 0.9 0.1").'
                                                    ' Can be "None" when file list are given. ',
                                type=float, nargs='*', default=[0.9, 0.1])
    dataset_parser.add_argument('--shuffle', help='Selects images randomly for each sample when set.'
                                                  ' Not possible when using file lists.',
                                action='store_const', const=True, default=False)
    dataset_parser.add_argument('--year', help='Sets the creation date of the data.',
                                type=int, default=2018)

    # Image filetypes
    image_parser = parser.add_argument_group('Image settings')
    image_parser.add_argument('--image-src-filetype', help='Defines the image source filetype (default: png)',
                              type=str, default="png")
    image_parser.add_argument('--image-dest-filetype', help='Defines the image destination filetype, '
                                                            'when show reqNone source type will be used. '
                                                            '(default: None)',
                              type=str, default=None)

    # Optional settings
    optional_parser = parser.add_argument_group('Optional settings')
    optional_parser.add_argument('--no-copy', help='Do not copy the images when set',
                                 action='store_const', const=True, default=False)
    optional_parser.add_argument('--skip-images-without-label',
                                 help='Do not copy the images without label when set. (COCO only)',
                                 action='store_const', const=True, default=False)

    # Optional path settings
    opt_path_parser = parser.add_argument_group('Optional path settings')
    opt_path_parser.add_argument('--file-list-path', help='Path to the file lists. '
                                                          'A file list is a .txt file containing the filenames'
                                                          ' for each set without the file extension. (default: None)',
                                 type=str, default=None)
    opt_path_parser.add_argument('--file-lists', help='List of the file list filenames. (default: None)',
                                 type=str, nargs='*', default=None)

    # Optional dataset settings
    opt_dataset_parser = parser.add_argument_group('Optional dataset settings')
    opt_dataset_parser.add_argument('--remap-labels',
                                    help='When set the script will lookup the \'label_mapping.py\' file'
                                         ' and remap the labels accordingly.',
                                    action='store_const', const=True, default=False)
    opt_dataset_parser.add_argument('--mapping-type',
                                    help='Set the mapping type. See \'label_mapping.py\' for options.',
                                    type=int,
                                    default=None)
    opt_dataset_parser.add_argument('--exclude', help='List of class IDs to exclude from label file.'
                                                      '(e.g. "--exclude 1 2 3" or "--exclude 1-3" or "--exclude 1 2-3")'
                                                      ' (default: None)',
                                    type=str, nargs='*', default=None)
    opt_dataset_parser.add_argument('--exclude-starts-at-one',
                                    help='When set the script counts the class IDs starting at 1, '
                                         'when not set counter starts at 0.',
                                    action='store_const', const=True, default=False)
    opt_dataset_parser.add_argument('--include', help='List of class IDs to include from label file.'
                                                      '(e.g. "--include 1 2 3" or "--include 1-3" or "--include 1 2-3")'
                                                      ' (default: None)',
                                    type=str, nargs='*', default=None)
    opt_dataset_parser.add_argument('--include-starts-at-one',
                                    help='When set the script counts the class IDs starting at 1, '
                                         'when not set counter starts at 0.',
                                    action='store_const', const=True, default=False)
    opt_dataset_parser.add_argument('--show-not-verified', help='Shows not verified label files.',
                                    action='store_const', const=True, default=False)
    opt_dataset_parser.add_argument('--exclude-area', type=int, default=None,
                                    help='Excludes all labels with a area less or equal than the given value. '
                                         '(default: None)')

    # Darknet settings
    darknet_parser = parser.add_argument_group('Darknet settings')
    darknet_parser.add_argument('--dataset-name', help='Name for the dataset. (Darknet only).',
                                type=str, default=None)
    darknet_parser.add_argument('--rel-output-path', help='Relative path to write in set file list (Darknet only).',
                                type=str, default=None)

    # Statistic settings
    stat_parser = parser.add_argument_group('Statistic settings')
    stat_parser.add_argument('--stats', help='Calculate image and label statistics when set.',
                             action='store_const', const=True, default=False)
    stat_parser.add_argument('--stats-img', help='Calculate image statistics when set.',
                             action='store_const', const=True, default=False)
    stat_parser.add_argument('--stats-label', help='Calculate label statistics when set.',
                             action='store_const', const=True, default=False)
    stat_parser.add_argument('--tablefmt', help="Various plain-text table formats (tablefmt) are supported.",
                             type=str, nargs='?', default='psql',
                             choices=['plain', 'simple', 'grid', 'fancy_grid', 'github', 'pipe', 'orgtbl', 'jira',
                                      'presto', 'psql', 'rst', 'mediawiki', 'moinmoin', 'youtrack', 'html', 'latex',
                                      'latex_raw', 'latex_booktabs', 'tsv', 'textile'])

    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.target_format == 'coco':
        converter = converters.COCOConverter(args)
    elif args.target_format == 'csv':
        converter = converters.CSVConverter(args)
    elif args.target_format == 'tfrecord':
        converter = converters.TFRecordConverter(args)
    elif args.target_format == 'darknet':
        converter = converters.DarknetConverter(args)
    else:
        sys.exit(-1)

    if args.sets or args.file_lists:
        converter.split(args.sets, args.set_sizes, args.shuffle)

    converter.convert()
    converter.print_class_distribution()

    if args.stats or args.stats_label:
        converter.calc_label_statistics()
    if args.stats or args.stats_img:
        converter.calc_img_statistics()


if __name__ == '__main__':
    main()
