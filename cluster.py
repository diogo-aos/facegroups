import config
from pathlib import Path
import pickle
from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import shutil

def load_db_gen(embeds_path: Path) -> list:
    with embeds_path.open("rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def get_face_filenames(im_path: Path, n_faces: int) -> List[str]:
    im_basename, im_ext = im_path.name.split('.')
    face_fns = []
    for i in range(n_faces):
        face_fn = f"{im_basename}{'' if i==0 else f'_{str(i+1)}'}.{im_ext}"
        face_fns.append(face_fn)
    return face_fns

def build_dataset(embeds_path: Path,
                  face_thresh=config.FACE_CONFIDENCE_THRESH) \
                    -> Dict[str, np.ndarray]:
    """
    This function will rebuilt the faces database such that in each row
    there is a face embedding and associated metadata.
    It receives a path to a previously built face pickle database with the app.py module.
    This pickle file is expected to be a stream of tuples, each tuple with the format:

    (source image path: Path,
     faces embeddings: pytorch.tensor,
     faces detection probabilities: np.ndarray)

    All embeddings for the different faces reside in the same multidimensional tensor.
    All detection probabilities reside in the same numpy array.

    By default, all faces with a confidence score below the configured one in the config.py module
    will be discarded.

    The function will return a dictionary, where the values are np.ndarrays and each row is data
    about a single face:
    {'embeds': np.ndarray,     matrix with all embeddings
     'face_fn': np.ndarray,    vector matrix
     'im_fn': np.ndarray,      vector matrix
    }

    """

    dataset = {'face_fn': [],
            'im_fn': [],
            'embeds': []}
    # for im, embeds, probs in db:
    for im, embeds, probs in load_db_gen(embeds_path):
        face_fns = get_face_filenames(im, len(embeds))
        for e, face_fn, p in zip(embeds, face_fns, probs):
            # skip face if below confidence threshold
            if p < face_thresh:
                continue
            dataset["embeds"].append(e[0].detach().numpy())
            dataset['face_fn'].append(face_fn)
            dataset['im_fn'].append(im)

    # turn metadata arrays to numpy arrays
    for k,v in dataset.items():
        if k == 'embeds':
            continue
        dataset[k] = np.array(v)
    dataset['embeds'] = np.stack(dataset['embeds'], axis=0)
    return dataset
        
def main():
    dataset = build_dataset(config.database_path.joinpath("embeds.pkl"))
    embeds = dataset['embeds']

    n_clusters = config.N_CLUSTERS
    biggest_cluster_size = float('inf')
    max_cluster_size = config.MAX_CLUSTER_SIZE
    #convergence_count = 5
    
    while biggest_cluster_size > max_cluster_size:
        print(f"computing with # clusters = {n_clusters}, biggest cluster size = {biggest_cluster_size}")
        model = AgglomerativeClustering(linkage="single",
                                        n_clusters=n_clusters,
                                        memory=str(config.database_path.joinpath('sklearn_cache')),
                                        compute_full_tree=True)
        clusters = model.fit_predict(embeds)

        counts = np.bincount(clusters)
        biggest_cluster_size = counts.max()
        if counts.max() > max_cluster_size:
            n_clusters += 50
            continue

        # copy faces from source to cluster folders
        people_dir = config.database_path.joinpath("people")
        if people_dir.exists():
            shutil.rmtree(people_dir)
        people_dir.mkdir()

        face_dir = config.database_path.joinpath("faces")

        # process clusters
        for c in range(n_clusters):
            # check which images belong to this cluster 
            # and discard cluster if too few images
            c_idx, = np.where(clusters == c)
            if c_idx.size < config.CLUSTER_MIN_IMG:
                clusters[c_idx] = -1
                continue

            # create cluster directory and copy face file to it
            cluster_dir = people_dir.joinpath(f"cluster_{c}")
            if not cluster_dir.exists():
                cluster_dir.mkdir()

            for face_fn in dataset["face_fn"][c_idx]:
                shutil.copyfile(face_dir.joinpath(face_fn),
                                cluster_dir.joinpath(face_fn))
    
    # update clusters
    dataset['clusters'] = clusters

    with open(config.database_path.joinpath('dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


    print(f"# final clusters = {n_clusters}")
    print(f"# undiscarded clusters = {np.unique(clusters).size}")
    print(f"# total faces: {embeds.shape[0]}")
    print(f"# faces of discarded clusters = {np.where(clusters == -1)[0].size}")
    print(f"% discarded faces = {np.where(clusters == -1)[0].size / embeds.shape[0]}")
    print(counts)

    

if __name__ == "__main__":
    main()




