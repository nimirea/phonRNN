import csv
import argparse

parser = argparse.ArgumentParser(description='CELEX to IPA and split into sets')

# model parameters
parser.add_argument('--input_file', type=str, default = "Daland_etal_2011__AverageScores.csv",
                    help='location of the data corpus')
parser.add_argument('--output_file', type=str, default = "daland_test.txt",
                    help='where to write the processed splits')
args = parser.parse_args()


daland_results_path = args.input_file
outfile = args.output_file
output_list = []

translation = {
    'iy': 'i',
    'ih': 'ɪ',
    'aa': 'ɑ',
    'eh': 'ɛ',
    'sh': 'ʃ',
    'th': 'θ',
    'r': 'ɹ',
}
total_chars = []

with open(daland_results_path) as infile:
    with open(outfile, "w") as out:
        reader = csv.DictReader(infile)
        for row in reader:
            word = row['phono_cmu']
            orig_word = word
            word = word.lower() # make lowercase
            word = word[:-4] # strip coda
            word = word.translate(translation)
            word = word.replace("0", "")

            charlist = word.split(" ")
            new_charlist = []
            for ch in charlist:
                if ch not in total_chars:
                    total_chars.append(ch)

                if ch[:2] in translation:
                    recombined_char = translation[ch[:2]] + ch[2:]
                    if ch[:2] == 'iy' or ch[:2] == 'aa':
                        # stressed /i/ is always long in the source data
                        recombined_char += 'ː'
                    new_charlist.append(recombined_char)
                else:
                    new_charlist.append(ch)
            word= " ".join(new_charlist)

            out.write(word + "\n")
