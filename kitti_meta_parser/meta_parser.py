#!/usr/bin/env python


import argparse
import logging

import os
import shutil

from pathlib import Path


def main():
    """
    main function
    """

    # PARSER
    # general arguments
    parser = argparse.ArgumentParser(
        description="TODO"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti",
        help="dataset which should be pared, valid options are kitti, nuimages and cityscapes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print debug level logs",
    )

    parser.add_argument(
        "--road_type",
        type=str,
        default="um,umm,uu",
        help="optional filter for road type; valid arguments is a comma separated list (without spaces) of um, umm and uu",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        help="source dir path",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="print the velocity meta data for the data points specified in the data point file",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="copy the identified files, needs --min and --max to be set",
    )
    parser.add_argument(
        "--min",
        default=-99999,
        help="min velocity",
    )
    parser.add_argument(
        "--max",
        default=99999,
        help="max velocity",
    )

    args = parser.parse_args()
    source_path = args.source_path
    road_type_list = args.road_type.split(',')
    dataset = args.dataset

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    # LOGGING
    # enable logging
    logging.basicConfig(
        # filename = log_file_path,
        # encoding = 'utf-8',
        level    = log_level,
        format   =  '%(asctime)s %(levelname)s %(message)s'
    )
    # logging.debug("debug")
    # logging.info("info")
    # logging.warning("warning")
    # logging.error("error")

    if dataset == 'Kitti':
        if not args.scan and not args.select:
            logging.error(f"at least one of the arguments --scan or --select is needed")
            exit(1)

        if args.select:
            if not args.min and not args.max:
                logging.error(f"--select needs --min and/or --max to be specified")
                exit(1)

        logging.info("##################################################")
        logging.info("# process started")

        meta_files_root = os.path.join(source_path, "oxts")
        logging.debug(f"meta_files_root: {str(meta_files_root)}")
        # exit(0)
        if not os.path.exists(meta_files_root):
            logging.error(f"{str(meta_files_root)} does not exist, exiting")
            exit(1)

        file_dict_list = []
        velocity_max = 0
        velocity_min = 100
        for meta_file_name_ego in Path(meta_files_root).rglob('*.txt'):
            file_dict = {}
            if any(f"{road_type}_" in str(meta_file_name_ego) for road_type in road_type_list):
                file_dict['meta_file_sub_path'] = str(meta_file_name_ego)
                with open(meta_file_name_ego, 'r') as meta_file:
                    meta_line = meta_file.readline()
                    logging.debug(meta_line)
                    meta_line_as_list = meta_line.split(' ')
                    logging.debug(meta_line_as_list)
                    velocity_in_m_per_s = float(meta_line_as_list[8])
                    velocity_in_m_per_s += abs(float(meta_line_as_list[9]))
                    velocity_in_m_per_s += abs(float(meta_line_as_list[10]))
                    logging.debug(velocity_in_m_per_s)
                    file_dict['velocity_in_km_per_h'] = int(velocity_in_m_per_s * 3.6)
                    logging.debug(file_dict['velocity_in_km_per_h'])
                    if args.min:
                        if file_dict['velocity_in_km_per_h'] < int(args.min):
                            continue
                    if args.max:
                        if file_dict['velocity_in_km_per_h'] > int(args.max):
                            continue
                    file_dict_list.append(file_dict)
                    if file_dict['velocity_in_km_per_h'] > velocity_max:
                        velocity_max = file_dict['velocity_in_km_per_h']
                    if file_dict['velocity_in_km_per_h'] < velocity_min:
                        velocity_min = file_dict['velocity_in_km_per_h']
        logging.debug(file_dict_list)
        # FIXME currently just ignoring velocity_min as I set it static to 0
        # histogram_length = velocity_max+1
        if args.scan:
            histogram_length = 100
            histogram = [0]*histogram_length
            filter_lenth = 5
            for file_dict in file_dict_list:
                histogram[file_dict['velocity_in_km_per_h']]+=1
            histogram_short = [sum(histogram[i:i+filter_lenth]) for i in range(0, len(histogram), filter_lenth)]

            print(f"total amount of images: {str(len(file_dict_list))}")
            for i in range(len(histogram_short)):
                print(f"{str(i*filter_lenth)}-{str(i*filter_lenth+filter_lenth-1)} {str(histogram_short[i])}")

        if args.select:
            target_folder = "image_2___filtered"
            os.makedirs(target_folder, exist_ok = True)
            for file_dict in file_dict_list:
                file_image_sub_path = file_dict['meta_file_sub_path'].replace('oxts', 'image_2')
                file_image_sub_path = file_image_sub_path.replace('.txt', '.png')
                file_dict['image_file_sub_path'] = file_image_sub_path
                logging.debug(f"checking meta file {str(file_dict['meta_file_sub_path'])} with image file {str(file_dict['image_file_sub_path'])} with velocity {str(file_dict['velocity_in_km_per_h'])}")
                if args.min:
                    if file_dict['velocity_in_km_per_h'] < int(args.min):
                        continue
                if args.max:
                    if file_dict['velocity_in_km_per_h'] > int(args.max):
                        continue
                logging.debug(f"file selected")

                source_image_path = file_dict['image_file_sub_path']
                target_image_path = file_dict['image_file_sub_path'].replace('image_2', target_folder)
                logging.info(f"copying {str(source_image_path)} to {str(target_image_path)}")
                shutil.copy(source_image_path, target_image_path)
    else:
        logging.error(f"unknown dataset: {str(dataset)}, exiting")
        exit(1)

if __name__ == "__main__":
    main()
