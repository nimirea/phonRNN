# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import csv
import data
import features
import numpy

def main(preset_args = False):

    import model

    parser = argparse.ArgumentParser(description='PyTorch Phone-Level LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/processed/CELEX2/lemmas',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--phonol_emb', action='store_true',
                        help='use phonological embedding as a starting point')
    parser.add_argument('--fixed_emb', action='store_true',
                        help='change embedding weights')
    parser.add_argument('--emsize', type=int, default=34,
                        help='size of phone embeddings')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--anneal_factor', type=float, default=0.25,
                        help='amount by which to anneal learning rate if no improvement on annealing criterion set (1 = no annealing, 0.5 = learning rate halved)')
    parser.add_argument('--anneal_train', action='store_true',
                        help='anneal using the training loss instead of the validation loss')
    parser.add_argument('--patience', type=int, default=16,
                        help='number of training epochs to wait for validation loss to improve before annealing')
    parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the phone embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--stress', action='store_true',
                        help='keep track of word stress')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save_dir', type=str,  default='results',
                        help='directory in which to save various characteristics of the final model')
    parser.add_argument('--summary', type=str, default='summary.csv',
                        help='path to save summary CSV')
    parser.add_argument('--condition', type=int, default=0,
                        help='Condition referenced in summary CSV')
    parser.add_argument('--run', type=int, default=0,
                        help='Run within condition')
    parser.add_argument('--feat_tree', type=str, default='Futrell',
                        help='Feature tree to use (\'Futrell\' or \'orig\')')
    parser.add_argument('--alphabet', type=str, default='IPA',
                        help='Format that the data is in (IPA or ARPA)')
    parser.add_argument('--set_unset_nodes', action='store_true',
                        help='Run a test using set/unset nodes')
    parser.add_argument('--random_reset', action='store_true',
                        help='Create the model such that it resets to a random state after each word')
    args, unknown = parser.parse_known_args()
    output_info = vars(args) # this is the variable to use for outputting checkpoints

    # maybe you've passed a dictionary into this function!
    if preset_args:
        vars(args).update(preset_args)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # decide whether to anneal learning rate
    if args.anneal_factor >= 1.0:
        annealing = False
    else:
        annealing = True

    # make save directory if it doesn't exist already
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data, stress=args.stress)

    # flatten the data and move it to GPU
    for d in [corpus.train, corpus.valid, corpus.test]:
        d['char_tensor'] = d['char_tensor'].view(1, -1).t().contiguous()
        if args.cuda:
            d['char_tensor'] = d['char_tensor'].cuda()

    ###############################################################################
    # Build the model
    ###############################################################################

    nchars = len(corpus.dictionary)

    if args.phonol_emb:
        preset_embedding = corpus.dictionary.feature_embedding(ftree=args.feat_tree, encoding=args.alphabet, set_unset = args.set_unset_nodes)
    else:
        preset_embedding = corpus.dictionary.random_embedding(ftree=args.feat_tree, encoding=args.alphabet, set_unset = args.set_unset_nodes)

    # set batch size
    bsz = 1

    # generate pre-word activations here
    if args.model == "LSTM":
        reset_activations = [Variable(torch.FloatTensor(args.nlayers, bsz, args.nhid)),
                            Variable(torch.FloatTensor(args.nlayers, bsz, args.nhid))]
    else:
        # this is still a tuple, for DRY purposes!
        reset_activations = [Variable(torch.FloatTensor(args.nlayers, bsz, args.nhid)),]
    for v, variable in enumerate(reset_activations):
        # zero out pre-word activation if we're not using random initializations
        if not(args.random_reset):
            variable.zero_()
        # move variable to GPU if necessary
        if args.cuda:
            reset_activations[v] = variable.cuda()
    # save list as tuple
    reset_activations = tuple(reset_activations)

    # save this tuple as a file "random-reset.data"
    reset_activations_path = os.path.join(args.save_dir, 'random-reset.data')
    torch.save(reset_activations, reset_activations_path)


    model = model.RNNModel(args.model,
        nchars,
        args.emsize,
        args.nhid,
        args.nlayers,
        hidden_activations_path = reset_activations_path,
        dropout = args.dropout,
        tie_weights = args.tied,
        preset_embedding = preset_embedding,
        fixed_emb = args.fixed_emb)
    if args.cuda:
        model.cuda()

    # save initial encoder layer
    emb_before = model.encoder.weight.data.cpu()
    numpy.savetxt(os.path.join(args.save_dir, 'emb-before.txt'), emb_before.numpy(), delimiter = ",")

    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # Training code
    ###############################################################################

    def repackage_hidden(h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(repackage_hidden(v) for v in h)


    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(source, word_start, word_len, evaluation=False):
        seq_len = word_len - 1
        data = Variable(source[word_start:word_start+seq_len], volatile=evaluation)
        target = Variable(source[word_start+1:word_start+1+seq_len].transpose(1,0).contiguous().view(-1))
        return data, target

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0
        nchars = len(corpus.dictionary)
        hidden = model.init_hidden()
        current_word_start = 0
        for word_index, word_len in enumerate(data_source['lengths']):
            data, targets = get_batch(data_source['char_tensor'], current_word_start, word_len, evaluation=True)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, nchars)
            total_loss += criterion(output_flat, targets).data
            hidden = repackage_hidden(hidden)
            current_word_start += word_len
        return total_loss[0] / len(data_source['lengths'])

    def train(optimizer):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        start_time = time.time()
        nchars = len(corpus.dictionary)
        hidden = model.init_hidden()
        loss_hist = []
        current_lr = 0 # keep track of what the learning rate is right now

        current_word_start = 0
        for word_index, word_len in enumerate(corpus.train['lengths']):
            data, targets = get_batch(corpus.train['char_tensor'], current_word_start, word_len)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = model.init_hidden() # reset model state
            hidden = repackage_hidden(hidden) # detach from history
            model.zero_grad()
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, nchars), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            # update weight using optimizer
            optimizer.step()

            total_loss += loss.data
            loss_hist.append(loss.data[0])

            if word_index % args.log_interval == 0 and word_index > 0:
                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time

                for g in optimizer.param_groups:
                    current_lr = g['lr']

                print('| epoch {:3d} | {:5d}/{:5d} words | lr {:02.2f} | ms/word {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, word_index, len(corpus.train['lengths']), current_lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

                total_loss = 0
                start_time = time.time()

            # set everything up for the next word
            current_word_start += word_len

        return sum(loss_hist) / len(loss_hist), current_lr

    # Loop over epochs.
    lr = args.lr
    best_val_loss = None
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) # initialize optimizer
    if annealing:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.anneal_factor, patience=args.patience)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss, current_lr = train(optimizer)
            val_loss = evaluate(corpus.valid)

            output_info['epoch'] = epoch
            output_info['train_loss'] = train_loss
            output_info['val_loss'] = val_loss
            output_info['val_ppl'] = math.exp(val_loss) if val_loss < 700 else "na"
            output_info['epoch_time'] = time.time() - epoch_start_time
            output_info['current_lr'] = current_lr
            output_info['run_cond'] = str(args.condition) + "-" + str(args.run) # write run-cond, which will make merge with final-words.csv easier

            print('-' * 89)
            print(type(val_loss))
            if val_loss > 700:
                print('| end of epoch {:3d} | time: {:5.2f}s | loss too high!'.format(epoch, output_info['epoch_time']))
            else:
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, output_info['epoch_time'],
                                               val_loss, output_info['val_ppl']))
            print('-' * 89)

            # add checkpoint to summary file
            with open(args.summary, 'a+') as csvfile:
                read_data = csv.reader(csvfile)
                writer = csv.writer(csvfile)

                # write header row
                if os.path.getsize(args.summary) == 0:
                    writer.writerow(output_info.keys())

                writer.writerow(output_info.values())

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(os.path.join(args.save_dir, 'best-model.pt'), 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss

                # save trained encoder layer
                emb_after = model.encoder.weight.data.cpu()
                numpy.savetxt(os.path.join(args.save_dir, 'emb-after.txt'), emb_after.numpy(), delimiter = ",")

            if annealing:
                if args.anneal_train:
                    scheduler.step(train_loss)
                else:
                    scheduler.step(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(os.path.join(args.save_dir, 'best-model.pt'), 'rb') as f:
        model = torch.load(f)

    # Generate text from best model
    generate_opts = {
        'data': args.data,
        'checkpoint': os.path.join(args.save_dir, 'best-model.pt'),
        'outf': os.path.join(args.save_dir, 'sample.txt'),
        'cuda': args.cuda,
        'stress': args.stress
    }
    generate(generate_opts)

    # don't evaluate on test set here, let's do that at the end of everything

    print('| End of training |')
    print('=' * 89)

def generate(preset_args = False):
    # Model parameters.
    parser = argparse.ArgumentParser(description='Options for generating')
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
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='reporting interval')
    parser.add_argument('--argmax', action='store_true',
                        help='take argmax at each timestep instead of sampling')
    parser.add_argument('--stress', action='store_true',
                        help='keep track of word stress')
    args, unknown = parser.parse_known_args()
    output_info = vars(args) # this is the variable to use for outputting checkpoints

    # maybe you've passed a dictionary into this function!
    if preset_args:
        vars(args).update(preset_args)

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
    nchars = len(corpus.dictionary)
    hidden = model.init_hidden()
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
            input.data.fill_(phone_idx)
            phone = corpus.dictionary.idx2phone[phone_idx]

            outf.write(phone)
            # reset state between words
            if phone == "\n":
                hidden = model.init_hidden()
                input.data.fill_(1) # initialize with start symbol
            else:
                outf.write(" ")

            if i % args.log_interval == 0:
                print('| Generated {}/{} phones, including newline'.format(i, args.words))

if __name__ == '__main__':
    main()
