import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Callable, Generator
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import tqdm
import config

# # # # # # # # # # # # # # # # # # # # # # # # #

# config
FACE_CONFIDENCE_THRESH = config.FACE_CONFIDENCE_THRESH
src_dirs = config.src_dirs
database_path = config.database_path

# # # # # # # # # # # # # # # # # # # # # # # # #


mtcnn = MTCNN(select_largest=False, device='cpu', margin=40, keep_all=True, post_process=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def db_get_paths(db_path: Path) -> List[Path]:
    paths = []
    with db_path.open("rb") as f:
        while True:
            try:
                im_path, _, _ = pickle.load(f)
            except EOFError:
                break
            paths.append(im_path)
    return paths

def init_db(path: Path) -> bool:
    if not path.exists:
        path.mkdir()
    if not path.joinpath("faces").exists:
        path.joinpath("faces").mkdir()
    return True

def get_all_imgs(dirs: List[Path]) -> List[Path]:
    img_paths = []
    for dir in src_dirs:
        for ext in config.VALID_IMG_EXT:
            img_paths += list(dir.rglob(f"*.{ext}"))
    return img_paths


def extract_faces(im_path: Path,
                  db_path: Path,
                  face_extractor: Callable,
                  emb_extractor: Callable) -> list:
    im = Image.open(str(im_path))
    faces_path = str(db_path.joinpath("faces", im_path.name))
    faces, probs = face_extractor(im, save_path=faces_path,
                           return_prob=True)
    if faces is None:
        embeds = []
        probs = []
    else:    
        embeds = [emb_extractor(face.unsqueeze(0)) for face in faces]
    with db_path.joinpath("embeds.pkl").open("ab") as f:
        pickle.dump((im_path, embeds, probs), f)

    return embeds, probs

def main():
    init_db(database_path)
    img_paths = get_all_imgs(src_dirs)
    
    print(f"Found {len(img_paths)} images")

    # ignore images that have already been processed
    if database_path.joinpath("embeds.pkl").exists():
        db_paths = db_get_paths(database_path.joinpath("embeds.pkl"))
        
        img_paths = [p for p in img_paths if p not in db_paths]
        
    print(f"Found {len(img_paths)} images to process")

    imgs_error = []

    for p in tqdm.tqdm(img_paths):
        try:
            embeds, probs = extract_faces(p, database_path, mtcnn, resnet)
            if len(embeds) == 0:
                print(f"=============> no faces found in {p}")
        except Exception as e:
            print(f"\n\nerror in {p}")
            print(e)
            print(f"\n continuing...")
            imgs_error.append(p)

    print(f"all images that were not processed:")
    for p in imgs_error:
        print(p)

            

        

if __name__ == "__main__":
    main()