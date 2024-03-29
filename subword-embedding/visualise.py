import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# TODO: This should be a command line param or infered from the file
APOSTROPHE_TOKEN = 'A'


def visualise_embedding(embedding_dir, perplexity, learning_rate, image_path_name, label_mapping):
    """ Read an embedding from file and save the t-SNE visualisation

        Arguments:
            embedding_dir: The full path name to the embedding_dir as a string.
            perplexity: As described in https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
            learning_rate: As described in https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
            image_path_name: Name of the resulting t-SNE plot.
            label_mapping: The dictionary mapping to be used to insert the labels next to the data points.
    """
    tsne = TSNE(
        n_components=2,
        random_state=0,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=20000
    )

    with open(embedding_dir, 'r') as emb_file:
        embedding_list = emb_file.readlines()

    print('Number of subword units: {}'.format(len(embedding_list)))

    embedding_dict = {}
    vector_emb = []
    subword_labels = []
    # TODO: Make this a clean solution
    # Start at 2 to skip the random </s> character which is coming through (may need to be 3 for Georgian)
    for embedding in embedding_list[2:]:
        segmented_embedding = embedding.split()
        subword_labels.append(label_mapping[segmented_embedding[0]])
        embedding_vector = [float(dim) for dim in segmented_embedding[1:]]
        vector_emb.append(embedding_vector)
        embedding_dict[segmented_embedding[0]] = embedding_vector

    emb_2d = tsne.fit_transform(vector_emb)

    datapoint_indices = range(len(emb_2d))
    fig, ax = plt.subplots()
    for i, subword_label in zip(datapoint_indices, subword_labels):
        ax.scatter(emb_2d[i, 0], emb_2d[i, 1], c='c', label=subword_label)


    for i, subword_label in enumerate(subword_labels):
        ax.annotate(subword_label, (emb_2d[i, 0], emb_2d[i, 1]))

    plt.savefig(image_path_name)
    return embedding_dict


def label_maps_from_file(path_to_summary, label_mapping_code, separate_apostrophe_embedding, saved_dict=False):
    """ Generate a label mapping from the summary.txt file or a previously compiled
        mapping dictionary.

        Arguments:
            path_to_summary : A string variable of the path to the summary.txt file
            label_mapping_code : The mapping code is an integer defined as follows:
                                    0 = Map to self (no mapping)
                                    1 = Map to English phonetic spelling
                                    2 = Map to native language characters
            separate_apostrophe_embedding: boolean indicator of whether the apostrophe should be viewed
                                            a distinct subword unit or included within the pronunciation
            TODO: saved_dict : If the file is already a saved dictionary in the correct form
    """
    with open(path_to_summary, 'r') as mapping_file:
        mapping_list = mapping_file.readlines()

    apostrophe_options = find_apostrophe_options(mapping_list)

    label_map = {}
    for line in mapping_list:
        split_line = line.split()
        # Design the mapping such that the mapping code can be used to index the list
        # code: [code, phonetic English, native characters]
        code = split_line[1]
        code_options = [code, split_line[-1], split_line[0]]
        label_map[code] = code_options[label_mapping_code]
        if not separate_apostrophe_embedding:
            label_map[code + APOSTROPHE_TOKEN] = code_options[label_mapping_code] + apostrophe_options[label_mapping_code]
    # Add special characters sp and sil
    label_map['sp'] = 'sp'
    label_map['sil'] = 'sil'
    # Add special characters for the two hesitation characters
    label_map['G00'] = 'G00'
    label_map['G01'] = 'G01'
    return label_map


def find_apostrophe_options(mapping_list):
    for line in mapping_list:
        split_line = line.split()
        if split_line[0] == "'":
            en = split_line[0]
            native = split_line[-1]
            apostrophe_code = split_line[1]
            return [apostrophe_code, native, en]
