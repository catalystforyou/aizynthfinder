import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from aizynthfinder.aizynthfinder import AiZynthFinder
import pickle
import json
import logging
from rxnmapper import RXNMapper
logging.basicConfig(level=logging.DEBUG)
from tqdm import tqdm
import numpy as np
from rdkit import Chem
import argparse

rxnmapper = RXNMapper()

def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

def mapping_smiles(smiles):
    try:
        return rxnmapper.get_attention_guided_atom_maps([smiles])[0]['mapped_rxn']
    except:
        return smiles

def predict(idx):
    np.random.seed(137)
    finder = AiZynthFinder()
    # simple_rxns = json.load(open('../data/simple_reactions.json'))
    # test_rxns = np.random.choice(simple_rxns, 100, replace=False)
    test_rxns = json.load(open('mechanism_results/50K_results_{}.json'.format(idx)))
    for subidx, (k, v) in enumerate(tqdm(test_rxns.items())):
        for subsubidx, v_ in enumerate(v):
            # r, p = test_mol.split('>')[0], test_mol.split('>')[-1]
            # clean_rxn = canonical_smiles(r) + '>>' + canonical_smiles(p)
            # print(clean_rxn)
            rxn = v_ + '>>' + k
            mapped_rxn = mapping_smiles(rxn)
            try:
                finder.target_smiles = mapped_rxn
                print(mapped_rxn)
                finder.tree_search()
                finder.build_routes()
                stat = finder.extract_statistics()
                routes = [route['reaction_tree'].to_dict() for route in finder.routes]
                with open(f'mechanism_results/routes_{idx}_{subidx}_{subsubidx}.json', 'w') as f:
                    json.dump(routes, f)
                with open(f'mechanism_results/statistics_{idx}_{subidx}_{subsubidx}.json', 'w') as f:
                    json.dump(stat, f)
            except:
                continue


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()

    predict(args.idx)