import config
from pathlib import Path
import pickle
import numpy as np
import cv2
import itertools
import shutil
import tqdm

SKIP_LABELLED = True

def label_clusters(dataset_path: Path):
    # load dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # cycle clusters
    dataset['labels'] = dataset.get('labels', {})
    unique_clusters = np.unique(dataset['clusters'])

    unlabeled_clusters = [c for c in unique_clusters if c not in dataset['labels']]
    print(f"{len(unlabeled_clusters)} unlabeled clusters")

    i = 0
    while i < len(unique_clusters):
        c = unique_clusters[i]
        if c == -1:
            i += 1
            continue

        # skip cluster if already labelled
        if SKIP_LABELLED and c in dataset['labels']:
            i += 1
            continue

        # load cluster
        cluster_dir = config.database_path.joinpath("people", f"cluster_{c}")
        cluster = []
        for face_fn in cluster_dir.iterdir():
            face = cv2.imread(str(face_fn))
            cluster.append(face)

        # create grid of faces
        grid = [cluster[i:i+config.GRID_SIZE[0]] for i in range(0, len(cluster), config.GRID_SIZE[0])]
        
        grid_img_rows = []
        for i,row in enumerate(grid):
            if i==0:
                grid_img_rows.append(np.hstack(row))
            else:
                # create row the same size as the first row, stack row imgs and copy to row
                row_img = np.zeros_like(grid_img_rows[0])
                this_row = np.hstack(row)
                row_img[:this_row.shape[0], :this_row.shape[1]] = this_row
                grid_img_rows.append(row_img)


        grid_img = np.vstack(grid_img_rows)

        # show cluster
        cv2.imshow(f"cluster {c} - {len(cluster)} images", grid_img)
        # capture received key and destrow window
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == ord('q'):
            break

        i += 1

        # if key is "p", go to previous image
        if key == ord('p'):
            i -= 2

        # if key is anything other then l (for label), skip
        if key != ord('l'):
            continue
        

        # label cluster
        label = input(f"cluster {c} label: ")
        dataset['labels'][c] = dataset['labels'].get(c, [])
        dataset['labels'][c].append(label)

        with open(dataset_path, "wb") as f:
            pickle.dump(dataset, f)


def navigate_clusters(dataset_path: Path):
    # load dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    labels = dataset['labels']
    pass

def export_clusters(dataset_path: Path):
    # load dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    labels = dataset['labels']
    all_labels = sorted(list(set(itertools.chain.from_iterable(labels.values()))))
    print("Labels:")
    for i, label in enumerate(all_labels):
        print(i, label)
    
    # input and validate indeces
    # TODO better validation
    valid = False
    while not valid:
        idx = input("choose comma seperated label indeces:")
        idx = idx.strip().split(',')
        idx = [i for i in idx if i]
        idx = list(map(int, idx))
        valid = True
    
    # input and validate export dir
    output_dir = None
    while output_dir is None or (not output_dir.exists() and not output_dir.is_dir()):
        output_dir = Path(input('select export dir:'))

    # selected labels
    selected_labels = [all_labels[i] for i in idx]

    # clusters that have the labels
    clusters = [c for c, ls in labels.items() for label in ls if label in selected_labels]
    
    # im_fn in the clusters
    im_fns = [fn for c in clusters for fn in dataset['im_fn'][dataset['clusters']==c]]

    print(f"exporting {len(im_fns)} images to {output_dir}")

    # copy all images to the export folder
    for fn in tqdm.tqdm(im_fns):
        shutil.copy(fn, output_dir.joinpath(fn.name))
        

def main():
    valid_choices = ['l', 'n', 'e']
    choice = ''
    while choice != 'q':
        while choice not in valid_choices:
            print('l - label')
            print('n - navigate')
            print('e - export')
            print('q - quit')
            choice = input('Choice:')
        if choice == 'l':
            label_clusters(config.database_path.joinpath("dataset.pkl"))
        elif choice == 'n':
            navigate_clusters(config.database_path.joinpath("dataset.pkl"))
        elif choice == 'e':
            export_clusters(config.database_path.joinpath("dataset.pkl"))

if __name__ == "__main__":
    main()