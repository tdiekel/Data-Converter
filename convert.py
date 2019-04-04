import argparse
import os
import sys

from utils.COCOConverter import COCOConverter
from utils.CSVConverter import CSVConverter
from utils.TFRecordConverter import TFRecordConverter
from utils.DarknetConverter import DarknetConverter


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

        args.exclude = exclude
    else:
        args.exclude = []

    return args


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', help='Path to raw image directory.',
                        type=str, required=True)
    parser.add_argument('--file-list-path', help='Path to the file lists. '
                                                 'A file list is a .txt file containing the filenames for each set '
                                                 'without the file extension. (default: None)',
                        type=str, default=None)
    parser.add_argument('--file-lists', help='List of the file list filenames. (default: None)',
                        type=str, nargs='*', default=None)
    parser.add_argument('--image-src-filetype', help='Defines the image source filetype (default: png)',
                        type=str, default="png")
    parser.add_argument('--image-dest-filetype', help='Defines the image destination filetype,'
                                                      'when None source type will be used. (default: None)',
                        type=str, default=None)
    parser.add_argument('--label-path', help='Path to label directory.',
                        type=str, required=True)
    parser.add_argument('--target-format', help='Format to save converted dataset in',
                        nargs='?', choices=['coco', 'csv', 'tfrecord', 'darknet'], required=True)
    parser.add_argument('--output-path', help='Path to save converted dataset',
                        type=str, required=True)
    parser.add_argument('--rel-output-path', help='Relative path to write in set file list (Darknet only).',
                        type=str, default=None)
    parser.add_argument('--dataset-name', help='Name for the dataset. (Darknet only).',
                        type=str, default=None)
    parser.add_argument('--label-map', help='Path to label map json file.',
                        type=str, default='./label_map.json')
    parser.add_argument('--no-copy', help='Do not copy the images when set',
                        action='store_const', const=True, default=False)
    parser.add_argument('--sets', help='List of subsets to create (e.g. "--sets train val").',
                        type=str, nargs='*', default=['train', 'val'])
    parser.add_argument('--set-sizes', help='Sizes of the subsets (e.g. "--sets 0.9 0.1").'
                                            ' Can be "None" when file list are given. ',
                        type=float, nargs='*', default=[0.9, 0.1])
    parser.add_argument('--shuffle', help='Selects images randomly for each sample when set.'
                                          ' Not possible when using file lists.',
                        action='store_const', const=True, default=False)
    parser.add_argument('--exclude', help='List of class IDs to exclude from label file.'
                                          '(e.g. "--exclude 1 2 3" or "--exclude 1-3" or "--exclude 1 2-3")'
                                          ' (default: None)',
                        type=str, nargs='*', default=None)
    parser.add_argument('--stats', help='Calculate image statistics when set.',
                        action='store_const', const=True, default=False)

    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.target_format == 'coco':
        converter = COCOConverter(
            image_path=args.image_path,
            image_src_type=args.image_src_filetype,
            image_dest_type=args.image_dest_filetype,
            label_path=args.label_path,
            label_map=args.label_map,
            file_lists=args.file_lists,
            output_path=args.output_path,
            excluded_classes=args.exclude
        )
    elif args.target_format == 'csv':
        converter = CSVConverter(
            image_path=args.image_path,
            image_src_type=args.image_src_filetype,
            image_dest_type=args.image_dest_filetype,
            label_path=args.label_path,
            label_map=args.label_map,
            file_lists=args.file_lists,
            output_path=args.output_path,
            excluded_classes=args.exclude
        )
    elif args.target_format == 'tfrecord':
        converter = TFRecordConverter(
            image_path=args.image_path,
            image_src_type=args.image_src_filetype,
            image_dest_type=args.image_dest_filetype,
            label_path=args.label_path,
            label_map=args.label_map,
            file_lists=args.file_lists,
            output_path=args.output_path,
            excluded_classes=args.exclude
        )
    elif args.target_format == 'darknet':
        converter = DarknetConverter(
            image_path=args.image_path,
            image_src_type=args.image_src_filetype,
            image_dest_type=args.image_dest_filetype,
            label_path=args.label_path,
            label_map=args.label_map,
            file_lists=args.file_lists,
            output_path=args.output_path,
            rel_output_path=args.rel_output_path,
            dataset_name=args.dataset_name,
            excluded_classes=args.exclude
        )
    else:
        sys.exit(-1)

    if args.no_copy:
        converter.images_copied = True

    if args.sets or args.file_lists:
        converter.split(args.sets, args.set_sizes, args.shuffle)

    converter.convert()
    converter.print_class_distribution()

    if args.stats:
        converter.calc_statistics()


if __name__ == '__main__':
    main()
