import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from aizynthfinder.aizynthfinder import AiZynthFinder
import pickle
import json
from tqdm import tqdm
import argparse
import multiprocessing

def predict(chunk_id, data_id):
    filename = 'dataset/config.yml'
    finder = AiZynthFinder(configfile=filename)
    finder.stock.select("n1")
    finder.expansion_policy.select("uspto_clean")
    # finder.filter_policy.select("uspto")
    # test_molecules = json.load(open('chembl_32_chemreps_medium.json', 'r'))
    # test_routes = pickle.load(open('dataset/routes_possible_test_hard.pkl', 'rb'))
    # test_molecules = [route[0].split('>>')[0] for route in test_routes]
    test_molecules = open('dataset/n1-targets.txt').read().split('\n')
    chunk_size = 11
    chunk_start = chunk_id * chunk_size
    chunk_end = (chunk_id + 1) * chunk_size
    if chunk_end > len(test_molecules):
        chunk_end = len(test_molecules)
    test_molecules = test_molecules[chunk_start:chunk_end]
    for idx, test_mol in enumerate(tqdm(test_molecules)):
        if chunk_start+idx in existed_number:
           continue
        '''if idx % 2 != data_id:
            continue'''
        finder.target_smiles = test_mol
        finder.tree_search()
        finder.build_routes()
        stat = finder.extract_statistics()
        routes = [route['reaction_tree'].to_dict() for route in finder.routes]
        with open(f'/teamdrive/projects/n1routes/mcts_v2/routes_{chunk_start+idx}.json', 'w') as f:
        # with open(f'routes_{chunk_start+idx}.json', 'w') as f:
            json.dump(routes, f)
        with open(f'/teamdrive/projects/n1routes/mcts_v2/statistics_{chunk_start+idx}.json', 'w') as f:
        # with open(f'statistics_{chunk_start+idx}.json', 'w') as f:
            json.dump(stat, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--data_id', type=int, default=0)
    args = parser.parse_args()
    existed = os.listdir('/teamdrive/projects/n1routes/mcts_v2/')
    existed_number = set([int(name.split('_')[1].split('.')[0]) for name in existed if name.split('_')[0] == 'routes'])
    print(len(existed_number))
    
    predict(args.chunk_id, 0)
    '''process_list = []
    for i in range(2):
        process_list.append(multiprocessing.Process(target=predict, args=(args.chunk_id, i)))
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()'''
        