# TD Data Science Case
# By Daniel Yun

from train import *
from label import *

if __name__ == '__main__':
    opt = input("1 - train model\n"
                "2 - test model\n"
                "Enter choice: ")
    if opt == '1':
        load()
        model = create()
        test(model)
        save(model)
    elif opt == '2':
        label()
