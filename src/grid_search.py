import os
from itertools import product
import main
import argparse

parser = argparse.ArgumentParser(description='Phone-Level LSTM Language Model')
parser.add_argument('--data', type=str, default='data/processed/CELEX2/lemmas',
                    help='location of the data corpus')
parser.add_argument('--alphabet', type=str, default='IPA',
                        help='Format that the data is in (IPA or ARPA)')
parser.add_argument('--feat_tree', type=str, default='Futrell',
                    help='Tree to use for features (default: Futrell)')
parser.add_argument('--condition_runs', type=int, default=50,
                    help='Runs per condition')
parser.add_argument('--output_dir', type=str, default='results',
                    help='path to save results, including summary CSV and model checkpoint')
parser.add_argument('--summary_filename', type=str, default='summary.csv',
                    help='path to save summary CSV, within results directory')
parser.add_argument('--cuda', action='store_true',
                    help='Whether to use CUDA')
parser.add_argument('--run_start', type=int, default=0,
                    help='Where to start seeding the model')
gs_args = parser.parse_args()

condition_runs = gs_args.condition_runs
results_dir = gs_args.output_dir

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

grid_args = {
    'data': gs_args.data,
    'alphabet': gs_args.alphabet,
    'set_unset_nodes': True,
    'model': 'LSTM',
    'epochs': 25,
    'phonol_emb': [True, False],
    'feat_tree': gs_args.feat_tree,
    'fixed_emb': False,
    'nhid': 512,
    'nlayers': 2,
    'lr': 1.0,
    'clip': 5,
    'cuda': gs_args.cuda,
    'anneal_factor': 0.25,
    'anneal_train': False,
    'patience': 0,
    'stress': True,
    'summary': os.path.join(results_dir, gs_args.summary_filename),
    'random_reset': False
}

# change number of features dynamically
if gs_args.feat_tree == "Futrell":
    grid_args['emsize'] = 34
elif gs_args.feat_tree == "orig":
    grid_args['emsize'] = 27

# change number of features based on whether representation includes set/unset nodes
if grid_args['set_unset_nodes']:
    grid_args['emsize'] *= 2

def conditions(grid_args):
    """ Function that generates a list of dictionaries, with each dictionary representing the arguments of a condition. """

    arg_keys = list(grid_args.keys())
    # listify things that aren't lists
    arg_values = tuple([x if isinstance(x,list) else [x] for x in grid_args.values()])
    arg_combos = list(product(*arg_values))

    conditions = [dict(zip(arg_keys, x)) for x in arg_combos]

    return conditions

# generate and run conditions
for c, condition in enumerate(conditions(grid_args)):
    for run in range(gs_args.run_start, condition_runs):
        condition['seed'] = run # use a randomly-generated seed instead?
        condition['run'] = run
        condition['condition'] = c
        condition['save_dir'] = os.path.join(results_dir, str(c) + '-' + str(run))

        print()

        # could potentially parallelize here?
        main.main(condition)
