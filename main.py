import argparse
import os
from src.preprocessing import import_data, bar_box_gui
from src.inpainting import inpaint_scalebars
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(mode, offline, tag):
    """[summary]

    :param mode: [description]
    :type mode: [type]
    :param offline: [description]
    :type offline: [type]
    :param tag: [description]
    :type tag: [type]
    :raises ValueError: [description]
    """
    print("Running in {} mode, tagged {}, offline {}".format(mode, tag, offline))

    # Initialise Config object
 

    if mode == 'import':
        import_data.import_data()
    elif mode == 'preprocess':
        bar_box_gui.preprocess_gui()
    elif mode == 'inpaint':
        inpaint_scalebars.inpaint_samples('train')
    elif mode == 'generate':
        inpaint_scalebars.inpaint_samples('generate')
    else:
        raise ValueError("Mode not recognised")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-t", "--tag")
    parser.add_argument('-o', '--offline', action='store_true',
                        help='disable wandb')
    args = parser.parse_args()
    if args.tag:
        tag = args.tag
    else:
        tag = 'test'

    main(args.mode, args.offline, tag)
# main('train', False, 'test')