# purpose: translate CELEX into IPA and split into validation, test, and training sets

from random import shuffle
import argparse
import os

# parameters

parser = argparse.ArgumentParser(description='CELEX to IPA and split into sets')

# model parameters
parser.add_argument('--input_file', type=str, default = 'epl.cd',
                    help='location of the data corpus file (this will end in .cd)')
parser.add_argument('--output_dir', type=str, default = 'celex_ipa',
                    help='where to write the processed splits')
parser.add_argument('--val_n', type=int, default = 10000,
                    help='number of words in validation set')
parser.add_argument('--test_n', type=int, default = 10000,
                    help='number of words in test set')
parser.add_argument('--train_n', type=int, default = 10000,
                    help='number of words in training set')
args = parser.parse_args()

cd_file = args.input_file          # location of corpus file
output_dir = args.output_dir    # where to write processed data
val_n = args.val_n.               # number of words in validation set
test_n = args.test_n              # number of words in test set
train_n = args.train_n             # number of words in training set

word_list = []
word_dict = {}
# dictionary to translate things into IPA
disc2ipa = {
    '1': 'e ɪ', # remove space to make this a proper diphthong
    'z': 'z',
    '{': 'æ',
    'b': 'b',
    '@': 'ə',
    's': 's',
    '2': 'a ɪ', # remove space to make this a proper diphthong
    'k': 'k',
    'I': 'ɪ',
    '#': 'ɑː',
    'f': 'f',
    't': 't',
    'n': 'n',
    'd': 'd',
    'N': 'ŋ',
    'm': 'm',
    'S': 'ʃ',
    'w': 'w',
    'R': 'ɹ', # modify to another symbol to make syllabic
    'E': 'ɛ',
    'r': 'ɹ',
    'i': 'iː',
    'v': 'v',
    'H': 'n', # could be syllabic
    'Q': 'ɑ',
    'P': 'l', # coule be syllabic
    'V': 'ʌ',
    'h': 'h',
    '$': 'ɔː',
    'l': 'l',
    '5': 'o ʊ',
    '_': 'd͡ʒ',
    '9': 'ʊ ə',
    '6': 'a ʊ',
    'u': 'uː',
    'g': 'g',
    'D': 'ð',
    '3': 'ɜː',
    'Z': 'ʒ',
    'p': 'p',
    'T': 'θ',
    '7': 'ɪ ə',
    'j': 'j',
    'J': 't͡ʃ',
    'U': 'ʊ',
    'q': 'ɑː', # nasalized, originally
    '4': 'ɔ ɪ',
    '8': 'ɛ ə',
    'F': 'm',
    '~': 'ɒː', # nasalized originally
    '0': 'æː', # nasalized originally
    'c': 'æ',
    'x': 'x',
    'C': 'n',
}

# set of vowels
vowelset = {'ɛ', 'ɔ', 'ɜː', 'ʊ', 'ʌ', 'e', 'a', 'ɑː', 'ɑ', 'æː', 'ə', 'æ', 'ɪ', 'uː', 'ɒː', 'ɛ', 'iː', 'o', 'ɔː'}
two_words = 0

with open(cd_file, 'r') as csvfile:

    for row in csvfile:
        # get orthographic transcription
        ortho = row.split('\\')[1]

        # skip over multiple-word phrases
        if " " in ortho:
            two_words += 1
        else:
            word = row.split('\\')[5]

            syllables = word.split("-")

            for sy, syllable in enumerate(syllables):
                # define stress level
                if "'" in syllable:
                    stress_level = 1
                elif '"' in syllable:
                    stress_level = 2
                else:
                    stress_level = 0

                syl_list = list(syllable)
                syl_string = " ".join([disc2ipa[x] for x in syl_list if x in disc2ipa])
                syl_ipa_list = syl_string.split(" ")

                if stress_level > 0:
                    stress_placed = False
                    for s, symbol in enumerate(syl_ipa_list):
                        if not stress_placed and symbol in vowelset:
                            syl_ipa_list[s] = symbol[0] + str(stress_level) + symbol[1:]
                            stress_placed = True


                syl_string = " ".join(syl_ipa_list)

                syllables[sy] = syl_string


            joined_word = " ".join(syllables)

            if ortho in word_dict:
                word_list.append(joined_word)
            else:
                word_dict[ortho] = [joined_word]

            word_list.append(joined_word)

print(len(word_list)," number of words excluding multiple-word phrases")
print(len(word_list) + two_words," number of words, including multi-word phrases")

# write the words
if (val_n + test_n + train_n) > len(word_list):
    print("Will not work, not enough words in the corpus to create testing/training/validation splits of the specified size")
    raise SystemExit

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

shuffle(word_list)

splits = {
    'valid': word_list[:val_n],
    'test': word_list[val_n:val_n + test_n],
    'train': word_list[-train_n:]
}

for split_name, split_words in splits.items():
    out_file = os.path.join(output_dir, split_name + '.txt')
    with open(out_file, 'w') as out:
        for word in split_words:
            out.write("%s\n" % "".join(word))
