# code
train.py: the main file
came.py: the proposed model
config.py: parameter setting
negativeSample.py: perform negative sampling
DataSet.py: data processsing and dataset generation


# data files (all_graph_path, song_graph_path and text_path) format for train.py:
## all_graph_path:
1. userid_1(or songid_1) songid_1(or userid_1) weight
2. userid_2(or songid_2) songid_2(or userid_2) weight
3. ...

## song_graph_path:
1. songid_1 songid2 weight
2. songid_3 songid4 weight
3. ...

## text_path:
1. content words for songid_1 (split by space)
2. content words for songid_2
3. ...


# environments:
python: 2.7.15
tensorflow: 1.8.0
OS: Ubuntu 18.04