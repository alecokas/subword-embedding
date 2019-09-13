import re
import sys

sys.path.append('utils')
from utils import to_float, remove_comment_elements


TIME_SCALE_FACTOR = 1e9
DECIMAL_PLACES = 6
POSN_INFO_LEN = 2
APOSTROPHE_TOKEN = 'A'

class Arc(object):
    """ Container for the arc information """
    def __init__(self, arc_string, subword_context_width, include_position_information=True):
        """ Initialise the Arc object

            Arguments:
                arc_string: A string line from the unzipped HTK MLF file
                subword_context_width: The subword context width as an integer (the number of grams to consider)
                include_position_information: Boolean for whether the position information should be included (^I, ^M, ^F) 
        """
        self.start_time_s, self.end_time_s, subword_info, self.neg_log_score = self.extract_arc(arc_string)
        self.token = self.strip_subword(subword_info, subword_context_width, include_position_information)

    def __str__(self):
        return '{} ---- {} / {} ----> {}'.format(self.start_time_s, self.token, self.neg_log_score, self.end_time_s)

    def extract_arc(self, arc_string):
        """ Extract arc information and format it such that it is ready to be saved as object attributes
        """
        start_time, end_time, subword_info, neg_log_score = to_float(arc_string.split())
        start_time_s = round(start_time / TIME_SCALE_FACTOR, DECIMAL_PLACES)
        end_time_s = round(end_time / TIME_SCALE_FACTOR, DECIMAL_PLACES)
        return start_time_s, end_time_s, subword_info, neg_log_score

    def strip_subword(self, subword_info, subword_context_width, incl_posn_info):
        """ Strip subwords of context and optionally the location indicator

            Arguments:
                subword_info: String with the full subword context information and location indicators.
                subword_context_width: The subword context width as an integer (the number of grams to consider)
                incl_posn_info: A boolean indicator for whether or not to include the subword position information (^I, ^M, ^F)
        """
        if subword_context_width > 3:
            raise Exception('The subword context width cannot be greater than 3.')

        itemised_subword_info = re.split(r'\+|\-', subword_info)
        if len(itemised_subword_info) == 1:
            return itemised_subword_info[0] if incl_posn_info else self.remove_location_indicator(itemised_subword_info[0])
        elif len(itemised_subword_info) == 3:
            if subword_context_width > 1:
                # Assume that if the context is 2 (bigram), we want the include the preceding subword unit
                stop = subword_context_width
                return ''.join(itemised_subword_info[:stop]) if incl_posn_info else self.remove_location_indicator(itemised_subword_info[:stop])
            else:
                return itemised_subword_info[1] if incl_posn_info else self.remove_location_indicator(itemised_subword_info[1])
        else:
            raise Exception('The subword unit length should be 1 or 3, but found {}'.format(len(itemised_subword_info)))

    def remove_location_indicator(self, subword_with_location):
        """ Strip location indicators from a string or strings within a list and return the result as a string

            Arguments:
                subword_with_location: Either a string or list containing the raw subword unit with location indicators.
        """
        if isinstance(subword_with_location, list):
            clean_subword_list = []
            for subword in subword_with_location:
                subword_split = subword.split('^')
                if len(subword_split) == 1:
                    clean_subword_list.append(subword_split[0])
                else:
                    clean_subword, apostrophe = self.__clean_subword_split(subword_split)
                    clean_subword_list.append(clean_subword)

                    if apostrophe:
                        clean_subword_list.append(apostrophe)
            return ' '.join(clean_subword_list)
        else:
            subword_split = subword_with_location.split('^')
            if len(subword_split) == 1:
                return subword_split[0]
            else:
                clean_subword, apostrophe = self.__clean_subword_split(subword_split)

                if apostrophe is not None:
                    return ' '.join([clean_subword, apostrophe])
                else:
                    return clean_subword

    def __clean_subword_split(self, raw_subword_split):
        pronunciation = raw_subword_split[1][POSN_INFO_LEN - 1:]
        if pronunciation.endswith(APOSTROPHE_TOKEN):
            pronunciation = pronunciation.replace(APOSTROPHE_TOKEN, '')
            apostrophe = APOSTROPHE_TOKEN
        else:
            apostrophe = None

        raw_subword = raw_subword_split[0] + pronunciation
        return raw_subword, apostrophe


class SentenceLabels(object):
    """ A representation of a sentence as a one-best sequence. """
    def __init__(self, string_mlf, subword_context_width, include_posn_info):
        """ Initialise the SentenceLabels object

            Arguments:
                string_mlf: The raw form of a reference sentence (MLF) as a string
                subword_context_width: The subword unit context width as an integer
        """
        if string_mlf:
            one_best_list = remove_comment_elements(string_mlf)
            self.label_name = one_best_list[0]
            self.arc_list = self.extract_arcs(one_best_list[1:], subword_context_width, include_posn_info)
            self.start_arc = self.arc_starts_at_zero()
        else:
            self.label_name = None
            self.arc_list = None

    def __str__(self):
        viz_arcs = ""
        for arc in self.arc_list:
            viz_arcs += (str(arc) + '\n')
        return "{}\n{}".format(self.label_name, viz_arcs)

    def extract_arcs(self, raw_arcs_list, subword_context_width, include_posn_info):
        """ Converts a string list of arcs to a list of Arc objects

            Arguments:
                raw_arcs_list: A list of string arcs

            Returns:
                arc_list: A list of arcs as Arc objects
        """
        arc_list = []
        for raw_arc in raw_arcs_list:
            arc_list.append(Arc(raw_arc, subword_context_width, include_posn_info))
        return arc_list

    def is_none(self):
        """ Returns a boolean indicating if the SentenceLabels object is empty
        """
        if self.label_name is None and self.arc_list is None:
            return True
        else:
            return False

    def arc_starts_at_zero(self):
        """ Returns the first arc in the SentenceLabels object
            (i.e. the arc which has a start time of zero seconds)
        """
        for arc in self.arc_list:
            if arc.start_time_s == 0.0:
                return arc

    def get_unique_tokens(self):
        """ Extract all unique sub-word level units in the dataset

            Returns:
                A set of unique sub-word level units
        """
        unique_tokens = set()
        for arc in self.arc_list:
            if not arc.token in unique_tokens:
                unique_tokens.add(arc.token)
        return unique_tokens

    def sentence(self):
        """ Combine all tokens on each arc (in order) to generate the original sentence. """
        return ' '.join([arc.token for arc in self.arc_list])


class MLFDataset(object):
    """ A class for containing the sub-word marked one-best reference sequences described in the MLF file """
    def __init__(self, path_to_mlf, subword_context_width, incl_posn_info):
        """ Initialises the MLFDataset object

            Arguments:
                path_to_mlf: The path to the MLF file as a string
                subword_context_width: whether to use monophones, biphones, triphones, etc
        """
        self.ref_list = self.read_mlf_list(path_to_mlf, subword_context_width, incl_posn_info)
        self.subwords = self.unique_subwords()

    def read_mlf_list(self, path_to_mlf, subword_context_width, incl_posn_info):
        """ Saves the MLF (one-best sequence) in the target file as a list of SentenceLabels objects

            Arguments:
                path_to_mlf: The path to the MLF file as a string
                subword_context_width: The subword context width as an integer
        """
        with open(path_to_mlf, 'r') as mlf_file:
            contents_list = mlf_file.read().split('\n.\n')

            one_best_list = []
            for string_sentence in contents_list:
                mlf_labels = SentenceLabels(string_sentence, subword_context_width, incl_posn_info)
                if not mlf_labels.is_none():
                    one_best_list.append(mlf_labels)

            return one_best_list

    def unique_subwords(self):
        """ Compiles a set containing all sub-word units in the one-best training set.
        """
        subwords_in_dataset = set()
        for one_best_ref in self.ref_list:
            unique_token_set = one_best_ref.get_unique_tokens()
            for subword in unique_token_set:
                if not subword in subwords_in_dataset:
                    subwords_in_dataset.add(subword)
        return subwords_in_dataset

    def save_unique_subwords(self, target_file):
        with open(target_file, 'w') as subword_file:
            subword_file.write(str(self.subwords))

    def corpus(self):
        """ Generate a text corpus from the 1-best reference sequences from ASR recording.
        """
        corpus = []
        for lat in self.ref_list:
            corpus.append(lat.sentence())
        return '\n'.join(corpus)
