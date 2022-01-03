import json

from freqAnalysis import tf, idf, tf_idf_analysis

def tf_idf_embed(paths, idf_counter, vec_of_paths):
    tf_counter = tf(paths)
    return tuple([tf_counter[path] * idf_counter[path] for path in vec_of_paths])

def tf_idf_uniform_embed(paths, tf_idf_counter, vec_of_paths):
    return tuple([(tf_idf_counter[path] if path in paths else 0) for path in vec_of_paths])

def tf_embed(paths, vec_of_paths):
    tf_counter = tf(paths)
    return tuple([tf_counter[path] for path in vec_of_paths])

def tf_decode(x, vec_of_paths):
    return [vec_of_paths[idx] for idx in range(len(vec_of_paths)) if x[idx] > 0]

def boolean_embed(paths, vec_of_paths):
    tf_counter = tf(paths)
    return tuple([tf_counter[path] > 0 for path in vec_of_paths])

def embed_data(filename, size_needed=None, use_names=None, key='eff', embed='tfidf'):

    """
    data should contain paths.
    :return: training data (embedding and efficiencies), the embedding function
    """
    data = json.load(open(filename, 'r'))

    bag_of_paths = set()

    if use_names is not None:
        names = use_names
    else:
        names = list(data.keys())

    if size_needed:
        names = names[:size_needed]

    for name in names:
        bag_of_paths.update(data[name]['paths'])

    vec_of_paths = sorted(list(bag_of_paths))
    print('len of bag of paths', len(vec_of_paths))

    # create dataset
    dataset_x = []
    dataset_y = []

    topos_with_same_paths = 0

    # pre-compute tfidf counts
    if embed == 'tfidf':
        idf_counter = idf([data[name]['paths'] for name in names], vec_of_paths)
    elif embed == 'tfidf-uniform':
        tf_idf_counter = tf_idf_analysis([data[name]['paths'] for name in names], vec_of_paths)

    for name in names:
        paths = data[name]['paths']

        if embed == 'tfidf':
            x = tf_idf_embed(paths, idf_counter, vec_of_paths)
        elif embed == 'tfidf-uniform':
            x = tf_idf_uniform_embed(paths, tf_idf_counter, vec_of_paths)
        elif embed == 'freq':
            x = tf_embed(paths, vec_of_paths)
        elif embed == 'boolean':
            x = boolean_embed(paths, vec_of_paths)
        else:
            raise Exception('unknown embed method ' + str(embed))

        if x in dataset_x:
            topos_with_same_paths += 1

        dataset_x.append(x)
        dataset_y.append(data[name][key])

    print('find topos with same paths', topos_with_same_paths)

    return dataset_x, dataset_y, vec_of_paths

