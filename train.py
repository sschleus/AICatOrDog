import sys
from pathlib import Path
import argparse
from dogCatClassifier import DogCatClassifier
from model import Model

IMG_HEIGHT = 256
IMG_WIDTH = 256

def get_args():
    parser = argparse.ArgumentParser(description="CNN Trainer for the Cat or Dog app.")

    parser.add_argument( "-f", "--folder", type=str,help="Destination folder to save the model after training ends.",default="Custom",)
    args = parser.parse_args()

    if Path(f"model_{args.folder}/model_{args.folder}").is_dir():
        print(f"Folder model_{args.folder} already exists do you want to overwrite ?")
        y = input('Type "Yes" or "No": ')
        if y != "Yes":
            print("Aborting.")
            sys.exit()
    return args


def main():
    args = get_args()

    clf = DogCatClassifier()
    model = Model(IMG_HEIGHT, IMG_WIDTH)
    model = model.create_baseCNNModel()
    path = Path(f"model_{args.folder}/model_{args.folder}.keras")
    clf.fit(path, model)

if __name__ == "__main__":
    main()
