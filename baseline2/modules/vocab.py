import csv
import re

class Vocabulary(object):
    """
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, *args, **kwargs):
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.blank_id = None

    def label_to_string(self, labels):
        raise NotImplementedError


class KoreanSpeechVocabulary(Vocabulary):
    def __init__(self, vocab_path, output_unit: str = 'character', sp_model_path=None):
        super(KoreanSpeechVocabulary, self).__init__()

        self.vocab_dict, self.id_dict = self.load_vocab(vocab_path, encoding='utf-8')
        self.sos_id = int(self.vocab_dict['<sos>'])
        self.eos_id = int(self.vocab_dict['<eos>'])
        self.pad_id = int(self.vocab_dict['<pad>'])
        self.blank_id = int(self.vocab_dict['<blank>'])
        self.labels = self.vocab_dict.keys()

        self.vocab_path = vocab_path
        self.output_unit = output_unit

    def __len__(self):

        return len(self.vocab_dict)

    def label_to_string(self, labels, target=True):
        """
        Converts label to string (number => Hangeul)

        Args:
            labels (numpy.ndarray): number label

        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """

        if len(labels.shape) == 1:
            sentence = str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
                if not target:
                    sentence = self.revise(sentence)
            return sentence

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def load_vocab(self, label_path, encoding='utf-8'):
        """
        Provides char2id, id2char

        Args:
            label_path (str): csv file with character labels
            encoding (str): encoding method

        Returns: unit2id, id2unit
            - **unit2id** (dict): unit2id[unit] = id
            - **id2unit** (dict): id2unit[id] = unit
        """
        unit2id = dict()
        id2unit = dict()

        try:
            with open(label_path, 'r', encoding=encoding) as f:
                labels = csv.reader(f, delimiter=',')
                next(labels)

                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]

                unit2id['<blank>'] = len(unit2id)
                id2unit[len(unit2id)] = '<blank>'

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(label_path))
        
    def revise(sentence: str):
        whitelist = {
            '간간이': 'PLACEHOLDER1',
            '스스로': 'PLACEHOLDER2',
            '겹겹이': 'PLACEHOLDER3',
            # ... add more if needed
        } # or maybe we can create a .json for these words...

        reverse_whitelist = {v: k for k, v in whitelist.items()}

        # Temporarily replace valid words with placeholders
        for word, placeholder in whitelist.items():
            sentence = sentence.replace(word, placeholder)

        # Handle repeated characters, syllables, or punctuation marks
        pattern = r'(.)\1+'
        sentence = re.sub(pattern, r'\1', sentence)

        # Handle specific cases of repeated word components
        word_pattern = r'(\w\w+)\1+'
        sentence = re.sub(word_pattern, r'\1', sentence)

        # Replace placeholders back with valid words
        for placeholder, word in reverse_whitelist.items():
            sentence = sentence.replace(placeholder, word)

        return sentence
    # def revise(self, sentence: str, redup_path: str):
    #     assert type(sentence) == str, "Input is not a string"
    #     words = sentence.split()
    #     result = []
    #     whitelist = {
    #         '간간이': 'PLACEHOLDER1',
    #         '스스로': 'PLACEHOLDER2',
    #         '겹겹이': 'PLACEHOLDER3'
    #     }
                
    #     for word in words:
    #         revised_w = ''    
    #         for t in word:
    #             if not revised_w:
    #                 revised_w += t
    #             elif revised_w[-1]!= t:
    #                 revised_w += t
    #         if revised_w == '스로':
    #             revised_w = '스스로'
    #         result.append(revised_w)
    #     return ' '.join(result)