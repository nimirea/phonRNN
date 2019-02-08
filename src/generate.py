import argparse
import operator

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch Phone-Level Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/switchboard_splits',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--stress', action='store_true',
                    help='keep track of word stress')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--argmax', action='store_true',
                    help='take argmax at each timestep instead of sampling')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data, stress=args.stress)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.ones(1, 1).long(), volatile=True) # initialize with start symbol
if args.cuda:
    input.data = input.data.cuda()

with open(args.outf, 'w') as outf:
    for i in range(args.words):
        output, hidden = model(input, hidden)

        phone_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        if args.argmax:
            phone_idx = torch.topk(phone_weights, 1)[1][0]
        else:
            phone_idx = torch.multinomial(phone_weights, 1)[0]

        # use as input for next thing
        input.data.fill_(phone_idx)

        phone = corpus.dictionary.idx2phone[phone_idx]

        outf.write(phone)
        # reset state between words, start conditioning from start symbol between words
        if phone == "\n":
            hidden = model.init_hidden(1)
            input.data.fill_(1) # initialize with start symbol
        else:
            outf.write(" ")

        if i % args.log_interval == 0:
            print('| Generated {}/{} phones, including newline'.format(i, args.words))
