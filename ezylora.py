#! ./venv/bin/python3

"""
EzyLoRA script, based on Aitrepreneur tutorial:
https://www.youtube.com/watch?v=70H03cv57-o&ab_channel=Aitrepreneur
interfaced with bmaltais/kohya_ss gradio app

github.com/ceccott - Oct '23
"""

import os
import shutil
import logging
import argparse
from jinja2 import Environment, FileSystemLoader
from gradio_client import Client

# Global settings
TOT_ITER_NUM = 1500
ITER_MIN_NUM_PER_PIC = 100
PIC_NUM_THRESHOLD = 15
ALLOWED_IMG_EXT = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
LORA_SETTINGS_TPL_FILE = 'LoraSettings.json.tpl'
LORA_SETTINGS_XL_TPL_FILE = 'LoraSettingsXL.json.tpl'

# Koya_ss settings
DEFAULT_KOHYA_SS_ENDPOINT = 'http://0.0.0.0:7860/'
API_BLIP_CAPTION_FN_INDEX = 184
BLIP_CAPTION_MIN_LEN = 30
BLIP_CAPTION_MAX_LEN = 75

# Logger settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EzyLoRA')

# jinja2 settings
env = Environment(loader=FileSystemLoader("./"))


def is_image_file(file_path, allowed_img_ext=ALLOWED_IMG_EXT):
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in allowed_img_ext


def main():
    parser = argparse.ArgumentParser(description="Init, format, BLIP caption and training settings, for kohya_ss LoRA training")

    # Add the command-line arguments
    parser.add_argument("--lora_name", type=str, required=True,
                        help="name for LoRa model")
    parser.add_argument("--src_path", type=str, required=True,
                        help="training images folder, if supplied num_pics is automatically set")
    parser.add_argument("--dst_path", type=str, required=False,
                        help="destination path for the generated folder tree and files")
    parser.add_argument('--rename_pics', action='store_true',
                        help="rename pictures according to <LoRA name>_<number> pattern if supplied")
    parser.add_argument('--sdxl', action='store_true',
                        help="sdxl model training. Use suitable images with this option enabled")
    parser.add_argument("--endpoint", type=str, required=False, default=DEFAULT_KOHYA_SS_ENDPOINT,
                        help="kohya_ss endpoint")

    args = parser.parse_args()
    if args.dst_path is None:
        logger.warning('Destination path not set, defaulting to current folder')
        args.dst_path = os.path.abspath(os.path.dirname(__file__))

    if args.src_path is not None:
        if not os.path.isdir(args.src_path):
            raise FileNotFoundError('No folder exists at specified src_path')

        args.num_pics = len(list(filter(is_image_file, os.listdir(args.src_path))))
        logger.info(f'Found {args.num_pics} images in src_path')

    # General Folder tree
    logger.info('Building folder tree')
    lora_path = {}  # init folder tree dict
    lora_path['root'] = os.path.join(args.dst_path, args.lora_name)
    lora_path['image'] = os.path.join(lora_path['root'], 'image')
    lora_path['model'] = os.path.join(lora_path['root'], 'model')
    lora_path['log'] = os.path.join(lora_path['root'], 'log')

    for key, path in lora_path.items():
        logger.info(f'--> creating {key} folder')
        os.makedirs(path)

    # Image subfolder create & populate
    img_subfolder_prefix = ITER_MIN_NUM_PER_PIC if (args.num_pics >= PIC_NUM_THRESHOLD) else TOT_ITER_NUM // args.num_pics
    logger.info(f'--> creating {img_subfolder_prefix} image subfolder')
    lora_path['image_set'] = os.path.join(lora_path['image'], str(img_subfolder_prefix) + '_' + args.lora_name)
    os.makedirs(lora_path['image_set'])

    if args.src_path:
        image_files = list(filter(is_image_file, os.listdir(args.src_path)))
        for cnt, src_file in enumerate(image_files):
            src_file_ext = os.path.splitext(src_file)[1]
            shutil.copy(os.path.join(args.src_path, src_file),
                        os.path.join(lora_path['image_set'],
                                     args.lora_name + '_' + str(cnt) + src_file_ext if args.rename_pics else ''))

    if args.sdxl:
        template = env.get_template(LORA_SETTINGS_XL_TPL_FILE)
    else:
        template = env.get_template(LORA_SETTINGS_TPL_FILE)

    # Rendering of LORA settings file
    logger.info('Saving LoRA settings file to project root folder')
    rendered_lora_settings_file = template.render(
        lora_path=lora_path,
        model_name=args.lora_name
    )
    with open(os.path.join(lora_path['root'], 'LoraSettings.json'), 'w') as fw:
        fw.write(rendered_lora_settings_file)

    # Koya endpoint call for BLIP captioning
    logger.info("BLIP captioning image folder")
    client = Client(args.endpoint)
    client.predict(
        lora_path['image_set'],  # str  in 'Image folder to caption' Textbox component
        ".txt",  # str  in 'Caption file extension' Textbox component
        1,  # int | float  in 'Batch size' Number component
        1,  # int | float  in 'Number of beams' Number component
        0.9,  # int | float  in 'Top p' Number component
        BLIP_CAPTION_MAX_LEN,  # int | float  in 'Max length' Number component
        BLIP_CAPTION_MIN_LEN,  # int | float  in 'Min length' Number component
        True,  # bool  in 'Use beam search' Checkbox component
        args.lora_name,  # str  in 'Prefix to add to BLIP caption' Textbox component
        "",  # str  in 'Postfix to add to BLIP caption' Textbox component
        fn_index=API_BLIP_CAPTION_FN_INDEX
    )
    logger.info("BLIP captioning complete, exiting...")
    logger.info(f"load the settings file {lora_path['root']}/LoraSettings.json in kohya_ss webapp to start the training")


if __name__ == "__main__":
    main()
