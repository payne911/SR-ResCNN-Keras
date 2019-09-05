from constants import load_model
from model import load_gen_model, setUpModel


if __name__ == '__main__':

    if load_model:
        load_gen_model()
    else:
        setUpModel()
