import argparse
import os
import sys

from utils.COCOConverter import COCOConverter
from utils.CSVConverter import CSVConverter
from utils.TFRecordConverter import TFRecordConverter


def check_args(args):
    valid_formats = ['png', 'jpg']

    assert os.path.isdir(args.image_path), 'Image dir not found at: {}'.format(args.image_path)

    assert (args.image_src_filetype in valid_formats
            ), 'Image source filetype not valid, please choose on of {}.'.format(valid_formats)

    assert (args.image_dest_filetype in valid_formats
            ), 'Image destination filetype not valid, please choose on of {}.'.format(valid_formats)

    assert os.path.isdir(args.label_path), 'Label dir not found at: {}'.format(args.label_path)

    assert os.path.isfile(args.label_map), 'Label map file not found at: {}.'.format(args.label_map)

    assert (len(args.sets) == len(args.set_sizes)
            ), 'Number of set sizes does not fit the number of sets.'

    if args.target_format == 'tfrecord':
        assert args.csv_paths is not None, 'CSV files needed to convert to tfrecord file. ' \
                                           'Please add argument \"--csv-paths\" with paths to the CSV file of every set.'
        for csv_path in args.csv_paths:
            assert os.path.isfile(csv_path), 'CSV file not found at: {}.'.format(csv_path)

    return args


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', help='Path to raw image directory.',
                        type=str, required=True)
    parser.add_argument('--image-src-filetype', help='Defines the image source filetype (default: png)',
                        type=str, default="png")
    parser.add_argument('--image-dest-filetype', help='Defines the image destination filetype (default: jpg)',
                        type=str, default="jpg")
    parser.add_argument('--label-path', help='Path to label directory.',
                        type=str, required=True)
    parser.add_argument('--target-format', help='Format to save converted dataset in',
                        nargs='?', choices=['coco', 'csv', 'tfrecord', 'csv+tfrecord'], required=True)
    parser.add_argument('--csv-paths',
                        help='Paths to csv label files (e.g. "--csv-paths label/train.csv label/val.csv")',
                        type=str, nargs='*', default=None)
    parser.add_argument('--output-path', help='Path to save converted dataset',
                        type=str, required=True)
    parser.add_argument('--label-map', help='Path to label map json file.',
                        type=str, default='./label_map.json')
    parser.add_argument('--sets', help='List of subsets to create (e.g. "--sets train val")',
                        type=str, nargs='*', default=['train', 'val'])
    parser.add_argument('--set-sizes', help='Sizes of the subsets (e.g. "--sets 0.9 0.1").',
                        type=float, nargs='*', default=[0.9, 0.1])
    parser.add_argument('--shuffle', help='Selects images randomly for each sample. (default: False)',
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
            output_path=args.output_path
        )
    elif args.target_format == 'csv':
        converter = CSVConverter(
            image_path=args.image_path,
            image_src_type=args.image_src_filetype,
            image_dest_type=args.image_dest_filetype,
            label_path=args.label_path,
            label_map=args.label_map,
            output_path=args.output_path
        )
    elif args.target_format == 'tfrecord':
        converter = TFRecordConverter(
            image_path=args.image_path,
            image_src_type=args.image_src_filetype,
            image_dest_type=args.image_dest_filetype,
            label_path=args.label_path,
            label_map=args.label_map,
            output_path=args.output_path
        )
    elif args.target_format == 'csv+tfrecord':
        converter = TFRecordConverter(
            image_path=args.image_path,
            image_src_type=args.image_src_filetype,
            image_dest_type=args.image_dest_filetype,
            label_path=args.label_path,
            label_map=args.label_map,
            output_path=args.output_path,
            create_csv=True
        )
    else:
        RuntimeError('Got wrong target format.')

    if args.sets:
        converter.split(args.sets, args.set_sizes, args.shuffle)

    converter.convert()



if __name__ == '__main__':
    main()
