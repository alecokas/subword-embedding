import argparse
import json
import fasttext
import numpy as np
import os
from subprocess import call
import sys

sys.path.append('subword-embedding')
from subword_corpus import MLFDataset
from visualise import visualise_embedding, label_maps_from_file

sys.path.append('utils')
from utils import write_to_file


RES_DIR = 'results'

def parse_arguments(args_to_parse):
    """ Parse the command line arguments.

        Arguments:
            args_to_parse: CLI arguments to parse
    """
    description = "Generate a subword-based corpus from which to generate subword embeddings"
    parser = argparse.ArgumentParser(description=description)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument(
        '-i', '--mlf-file', type=str, required=True,
        help="Input MLF file from which to generate the corpus."
    )
    general.add_argument(
        '--subword-corpus', type=str,
        default='{}/subword-corpus.dat'.format(RES_DIR),
        help='The path name of the subword corpus produced.'
    )
    general.add_argument(
        '-u', '--unique-subwords', type=str,
        default='{}/unique-subword-list.json'.format(RES_DIR),
        help='Path name of the file where the the unique subword unit list will be saved.'
    )
    general.add_argument(
        '-c', '--subword-context', type=int,
        default=1, help='The subword context: monophone (1), biphone (2), triphone (3), etc'
    )
    general.add_argument(
        '--loc-info', dest='subword_loc_info', action='store_true'
    )
    general.add_argument(
        '--no-loc-info', dest='subword_loc_info', action='store_false'
    )
    general.set_defaults(subword_loc_info=False)

    # Visualisation options
    visuals = parser.add_argument_group('Visualisation options')
    visuals.add_argument(
        '--only-viz', dest='only_viz', action='store_true'
    )
    visuals.set_defaults(only_viz=False)
    # Map the phone / grapheme names
    visuals.add_argument(
        '--no-map', dest='map_label', action='store_const', const=0
    )
    visuals.add_argument(
        '--map-to-en', dest='map_label', action='store_const', const=1
    )
    visuals.add_argument(
        '--map-to-native', dest='map_label', action='store_const', const=2
    )
    visuals.set_defaults(
        map_label=0
    )
    visuals.add_argument(
        '-v', '--emb-visual', type=str,
        default='{}/embedding/visualisation.png'.format(RES_DIR),
        help='The path name to the visualisation image of the subword embeddings'
    )
    visuals.add_argument(
        '-s', '--summary-file', type=str,
        default='data/summary.txt',
        help='The path name to the summary file or saved map'
    )

    # Embedding options
    embedding = parser.add_argument_group('Embedding options')
    model_opts = ['word2vec', 'fastText']
    embedding.add_argument(
        '-m', '--model', type=str, default='word2vec', choices=model_opts,
        help='The model to use: word2vec or fastText'
    )
    embedding.add_argument(
        '-e', '--embedding', type=str,
        default='{}/embedding/'.format(RES_DIR),
        help='Path name of the embedding directory'
    )
    embedding.add_argument(
        '-w2v', '--word2vec-dir', type=str,
        default='/home/dawna/ar527/word2vec/word2vec',
        help='Path to local word2vec directory'
    )
    embedding.add_argument(
        '-l', '--vec-length', type=int, default=4,
        help='The length of the vector representing each embedded subword unit.'
    )
    embedding.add_argument(
        '--save-to-npy', dest='embed_to_npy', action='store_true'
    )
    embedding.set_defaults(embed_to_npy=False)
    embedding.add_argument(
        '--apostrophe-embedding', dest='apostrophe_embedding', action='store_true'
    )
    embedding.set_defaults(apostrophe_embedding=False)
    # t-SNE representation
    embedding.add_argument(
        '-p', '--perplexity', type=float,
        default=5,
        help='The perplexity for t-SNE'
    )
    embedding.add_argument(
        '-lr', '--learning-rate', type=int,
        default=200,
        help='The t-SNE learning rate.'
    )

    args = parser.parse_args(args_to_parse)
    return args

def word2vec_embed(corpus_path, vector_length, target_dir, word2vec_dir):
    call("subword-embedding/word2vec_embed.sh {} {} {} {}".format(
        vector_length, target_dir, corpus_path, word2vec_dir), shell=True
    )

def fasttext_embed(corpus_path, vector_length, unique_subwords_path, target_dir):
    model = fasttext.train_unsupervised(corpus_path, dim=vector_length)
    print('In-vocab subwords: {}'.format(model.words))

    with open(unique_subwords_path) as unique_subwords_file:
        unique_subwords = json.load(unique_subwords_file)

    output_list = []
    # Add header
    output_list.append('{} {}'.format(len(model.words), vector_length))
    unique_subwords = ['</s>'] + unique_subwords

    for subword in unique_subwords:
        embedding = ' '.join([str(element) for element in model[subword].tolist()])
        line = '{} {}'.format(subword, embedding)
        output_list.append(line)

    # Check that nothing has been left out
    for subword in model.words:
        if not subword in unique_subwords:
            # Need to add subword
            print('Adding {}'.format(subword))
            embedding = ' '.join([str(element) for element in model[subword].tolist()])
            line = '{} {}'.format(subword, embedding)
            output_list.append(line)

    with open(os.path.join(target_dir, 'embedding.txt'), 'w') as embedding_file:
        embedding_file.write('\n'.join(output_list))

def save_embedding_to_npy(embedding_dict,
                          npy_embedding_file='{}/embedding/embedding.npy'.format(RES_DIR)):
    for key, val in embedding_dict.items():
        embedding_dict[key] = np.array(val)
    np.save(npy_embedding_file, embedding_dict)

def main(args):
    """ Primary point of entry for generating sub-word (phone or grapheme) level embeddings
    """
    if not args.only_viz:
        if not os.path.exists(RES_DIR):
            os.makedirs(RES_DIR)

        print('Generating corpus...')
        subword_dataset = MLFDataset(
            path_to_mlf=args.mlf_file,
            subword_context_width=args.subword_context,
            incl_posn_info=args.subword_loc_info,
            separate_apostrophe_embedding=args.apostrophe_embedding
        )
        subword_dataset.save_unique_subwords(target_file=args.unique_subwords)
        write_to_file(content=subword_dataset.corpus(), target=args.subword_corpus)

        print('Using {} to generate embeddings...'.format(args.model))
        if args.model == 'word2vec':
            word2vec_embed(
                corpus_path=args.subword_corpus,
                vector_length=args.vec_length,
                target_dir=args.embedding,
                word2vec_dir=args.word2vec_dir
            )
        elif args.model == 'fastText':
            if not os.path.exists(args.embedding):
                os.makedirs(args.embedding)
            fasttext_embed(
                corpus_path=args.subword_corpus,
                vector_length=args.vec_length,
                unique_subwords_path=args.unique_subwords,
                target_dir=args.embedding,
            )

    print('Creating visualisation using t-SNE...')

    label_map = label_maps_from_file(
        path_to_summary=args.summary_file,
        label_mapping_code=args.map_label,
        separate_apostrophe_embedding=args.apostrophe_embedding,
        saved_dict=False
    )

    embedding_dict = visualise_embedding(
        embedding_dir=os.path.join(args.embedding, 'embedding.txt'),
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        image_path_name=args.emb_visual,
        label_mapping=label_map
    )

    if args.embed_to_npy:
        save_embedding_to_npy(embedding_dict)

    print('Complete')

if __name__=='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
