import os
import torch
import argparse

parser = argparse.ArgumentParser('get data from checkpoint')
parser.add_argument('--dataset',type=str, required=False, default='MPIIGaze')

args = parser.parse_args()
dataset = args.dataset
CHECKPOINTS_PATH = os.path.join('./checkpoint/', dataset)
SAVE_PATH = os.path.join('./checkpoint/', dataset)

def load_checkpoint(filename='checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

best_value = []
if dataset=='MPIIGaze':
    for pn in range(15):
        checkpoint_file = '{}_{:0>2}thcheckpoint.pth.tar'.format('modifed_itracker_model',pn)
        state = load_checkpoint(checkpoint_file)
        best_value.append(state['best_prec1'])
elif dataset =='EyeDiap':
    for gn in range(5):
        checkpoint_file = '{}_{:0>2}thcheckpoint.pth.tar'.format('best', gn)
        state = load_checkpoint(checkpoint_file)
        best_value.append(state['best_prec1'])

txt_file = os.path.join(SAVE_PATH, 'data.txt')
with open(txt_file, 'w') as f:
    for value in best_value:
        f.write(str(value.item())+'\n')

mean = 0.0
for value in best_value:
    value = float(value)
    mean += value
mean /= len(best_value)

print 'mean predict on {} is {}'.format(dataset, mean)
