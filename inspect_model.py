from tensorflow.keras.applications import VGG16
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--include-top", type=int, default=1, help="to include top of CNN or not")
    args = vars(ap.parse_args())

    print("[INFO] Loading network...")
    model = VGG16(
        weights="imagenet", 
        include_top=args["include_top"] > 0)

    model.summary()

    print("[INFO] Showing layers...")
    for (i, layer) in enumerate(model.layers):
        print("[INFO] {}\t{}".format(i, layer.__class__.__name__))