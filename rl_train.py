from config import config as cfg
from rl_trading_framework import *
from itertools import combinations 
import multiprocessing
from torch.multiprocessing import Process


def main():

    print('devices being used:', cfg['n_of_gpu'], cfg['device'])

    price_data_all = get_data_torch(cfg['tickers'], cfg['train_start'], cfg['train_end'], cfg['start_time'], 
                                    cfg['end_time'], cfg['multiplier'], cfg['data_freq'], cfg['join_type'])

    t, n = price_data_all.shape
    all_pairs = list(combinations(range(n), 2))

    if cfg['full_sample_train']:
        pair_indices_all = generate_pairs(all_pairs, cfg['n_of_gpu'], full_sample=True)
    else:
        pair_indices_all = generate_pairs(all_pairs, cfg['n_of_gpu'], k=cfg['batch_size_train'])

    print('')
    print('Total time steps:', t)
    print('Total number of individual assets:', n)
    print('Total number of possible pairs:', len(all_pairs))
    print('Training batch size:', len(pair_indices_all))

    world_size = cfg['n_of_gpu']
    processes = []

    for rank in range(world_size):
        pair_indices = split_pair_indices(pair_indices_all, rank, world_size)
        p = Process(target=train, args=(rank, world_size, price_data_all, pair_indices))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()

