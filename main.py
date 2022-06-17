import argparse
import os
from src.preprocessing import import_data, annotation_gui
from src.inpainting import run_inpaint
from src.slicegan import run_slicegan, animations
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(mode):
    """[summary]

    :param mode: [mode to run in]
    :type mode: [str]

    """

    # Initialise Config object
 

    if mode == 'import':
        import_data()
    elif mode == 'preprocess':
        annotation_gui.preprocess_gui()
    elif mode == 'inpaint':
        run_inpaint.inpaint_dataset('train')
    elif mode =='slicegan':
        run_slicegan.slicegan_dataset()
    elif mode == 'animate':
        animations.animate_dataset()

    else:
        raise ValueError("Mode not recognised")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    args = parser.parse_args()
    main(args.mode)
# main('train', False, 'test')