import sys
from pathlib import Path
import argparse
from dogCatClassifier import DogCatClassifier
from model import Model

IMG_HEIGHT = 256
IMG_WIDTH = 256

def get_args():
    parser = argparse.ArgumentParser(description="CNN Trainer for the cats or dogs app.")
    parser.add_argument("-f", "--folder", type=str, help="Destination folder to save the model after training ends.", default="Custom")
    args = parser.parse_args()

    # Check and create the directory if it doesn't exist
    save_dir = Path(f"model_{args.folder}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    if Path(f"model_{args.folder}/trained_model.keras").is_dir():
        print(f"Folder model_{args.folder} already exists. Do you want to overwrite?")
        y = input('Type "Yes" or "No": ')
        if y != "Yes":
            print("Aborting.")
            sys.exit()
    return args

def main():
    args = get_args()

    clf = DogCatClassifier()
    model = Model(IMG_HEIGHT, IMG_WIDTH)
    model = model.create_enhancedCNNModel()
    path = Path(f"model_{args.folder}")
    clf.fit(path, model)

if __name__ == "__main__":
    main()
