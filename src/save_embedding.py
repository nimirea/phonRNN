import torch
import torch.nn as nn
import numpy
import os
import model
import csv
import argparse
import data

# Model parameters.
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--data', type=str, default='./data/switchboard_splits',
                    help='location of the data corpus')
# parser.add_argument('--batch_dir', type=str, default='',
#                     help='model checkpoints to use for batch operation (if unset, will only run script on one file)')
parser.add_argument('--summary_file', type=str, default='summary.csv',
                    help='file name of the summary file, within the batch_dir')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--stress', action='store_true',
                    help='keep track of word stress')
args = parser.parse_args()

model_after = torch.load(args.checkpoint)
if args.cuda:
    model_after.cuda()

model_after.eval() # switch model to eval mode

# extract the embedding
emb_after = model_after.preset_embedding.weight.data
emb_after = emb_after.cpu()
np_after = emb_after.numpy()

# initialize a new model with identical parameters that represents this model "before" training

# set the random seed
basename = os.path.split(args.checkpoint)[1]
basename = os.path.splitext(basename)[0]
seed = int(basename.split("-")[1])
torch.manual_seed(seed)

# get the requisite parameters from the summary file
basedir = os.path.split(args.checkpoint)[0]
summary_filename = os.path.join(basedir, args.summary_file)
with open(summary_filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['save'] == args.checkpoint:
            infodict = row

# convert params to the right types
info = {}
for key, value in infodict.items():
    if value.isdigit():
        info[key] = int(value)
    else:
        try:
            info[key] = float(value)
        except ValueError:
            info[key] = value
    if value == "True" or value == "False":
        info[key] = (value == "True")

print(info)

corpus = data.Corpus(info['data'], stress=info['stress'])

# flatten the data and move it to GPU
for d in [corpus.train, corpus.valid, corpus.test]:
    d['char_tensor'] = d['char_tensor'].view(1, -1).t().contiguous()
    if args.cuda:
        d['char_tensor'] = d['char_tensor'].cuda()

###############################################################################
# Build the model
###############################################################################

nchars = len(corpus.dictionary)

if info['phonol_emb']:
    preset_embedding = corpus.dictionary.feature_embedding()
else:
    preset_embedding = False

model_before = model.RNNModel(info['model'], nchars, info['emsize'], info['nhid'], info['nlayers'], info['dropout'], info['tied'], preset_embedding, info['fixed_emb'])
if args.cuda:
    model_before.cuda()

model_before.eval() # switch model to eval mode

# extract the embedding
emb_before = model_before.preset_embedding.weight.data
emb_before = emb_before.cpu()
np_before = emb_before.numpy()

# save the tensors
numpy.savetxt(os.path.join(basedir, basename + '_emb-after.txt'), np_after, delimiter=",")
numpy.savetxt(os.path.join(basedir, basename + '_emb-before.txt'), np_before, delimiter=",")
