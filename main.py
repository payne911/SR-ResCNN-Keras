import utils
from constants import prepare_im
from constants import hr_img_path
import prepare_data


if __name__ == '__main__':

    if prepare_im:
        prepare_data.load_imgs(hr_img_path + "*.png")  # TODO: customize path (command line)
    else:
        utils.loadData()
