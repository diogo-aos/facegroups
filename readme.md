# Usage

1. Configure the source directories and database directory (essentially the workspace directory for all intermediate work) in `config.py`.
2. Run `extract.py` to do face detection on all photos. All faces are stored in specific directory. A pickle file is created to keep track of faces and their source photo.
3. Run `label_clusters.py` to do clustering on all extracted faces. This runs agglomerative clustering on all faces. The number of clusters can be configured in `config.py`.
4. Run `label_clusters.py` to label clusters of faces. Clusters that don't a have minimum size (default is 2, `config.py`) are not considered.
    - Images from each cluster will be rendered on a OpenCV grid.
        - If you type 'l', you go back to the terminal to indicate the label for this cluster.
    - This script also allows to export all source images of faces belonging to clusters of specific labels.
        - The script will list all configured labels.
        - The user chooses which labels to export and the destination directory.


# Clustering

Defining the appropriate number of clusters is not straightforward.
Faces are clustered together according to the distance of their embeddings.
However, faces of 2 different people might be more similar then either people are to detection outliers (e.g. a flower).
This means a cluster would be created for the flower, and the 2 people would be clustered together, if too few clusters are allowed.
For this reason, the adopted strategy was to start with a small number of clusters and increase this number until the biggest cluster had less than given amount of faces (`config.py`).
This means another tunable parameter, but it served my objective.
This step has a lot of room for improvement.

# UI

It turns out the UI is one of the most important parts of this mini project.
Default settings for the clustering alone can only take the user so far.
Iterative clustering and labelling are key to quality people recognition.
Currently, this is accomplished with a hammered CLI for basic interaction and OpenCV GUI for cluster display.
If I were to continue this project, it would be nice to have a web based interface that managed labels and would allow for more complex operations, such as breaking apart clusters.


TODOs
[ ] allow user to query for list of photos, e.g. all images with this person, all images with these 2 people together, all images with these person but not this other person, etc. 

