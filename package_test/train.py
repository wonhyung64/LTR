#%%
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--epochs", default=1000, type=int)

    args = parser.parse_args()

    print(args.batch_size, args.epochs)
