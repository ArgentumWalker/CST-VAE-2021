from subprocess import run
import cv2
import os
import multiprocessing as mp
import shutil

DATA_PATH = "/home/argentumwalker/hdd/CST-VAE/data"
ARTIST_LIST = "./artists.txt"
PREPROCESS_WORKERS = 4
IMG_MIN_SIZE = 512
IMG_MAX_SIZE = 1024


def preprocess_worker(images_queue: mp.Queue):
    request = images_queue.get()
    while request is not None:
        d, i = request
        img = cv2.imread(f"{DATA_PATH}/_tmp/deviantart/{d}/{i}")
        if img.shape[0] > IMG_MAX_SIZE and img.shape[1] > IMG_MAX_SIZE:
            frac = IMG_MAX_SIZE / max(img.shape[0], img.shape[1])
            img = cv2.resize(img, dsize=None, fx=frac, fy=frac)
        if img.shape[0] > IMG_MIN_SIZE and img.shape[1] > IMG_MIN_SIZE:
            os.makedirs(f"{DATA_PATH}/deviantart/{d}", exist_ok=True)
            cv2.imwrite(f"{DATA_PATH}/deviantart/{d}/{i}", img)
        request = images_queue.get()


if __name__ == "__main__":
    run(f"gallery-dl -d {DATA_PATH}/_tmp -i {ARTIST_LIST}", shell=True, check=True)

    preprocess_queue = mp.Queue(maxsize=PREPROCESS_WORKERS * 4)
    ps = [mp.Process(target=preprocess_worker, args=(preprocess_queue,)) for _ in range(PREPROCESS_WORKERS)]
    for p in ps:
        p.start()

    for d in os.listdir(f"{DATA_PATH}/_tmp/deviantart"):
        for i in os.listdir(f"{DATA_PATH}/_tmp/deviantart/{d}"):
            preprocess_queue.put((d, i))
    for _ in range(PREPROCESS_WORKERS):
        preprocess_queue.put(None)

    for p in ps:
        p.join()
    shutil.rmtree(f"{DATA_PATH}/_tmp")

    print("COMPLETE")