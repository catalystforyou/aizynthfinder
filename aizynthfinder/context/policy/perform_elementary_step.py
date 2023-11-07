import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from bondnet.prediction.predictor import evaluate
from bondnet.prediction.load_model import load_model, load_dataset, get_model_info
from bondnet.prediction.io import PredictionOneReactant
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.utils import to_path
import time
from timeout_decorator import timeout, TimeoutError

bond_energy_info = open('/home/v-junrenli/mechanism/aizynthfinder/aizynthfinder/context/policy/bond_energy.txt', 'r').read().split('\n')
bond_energy_dict = {}
for line in bond_energy_info:
    bond_type, energy = line.split()[0], line.split()[1]
    if '-' not in bond_type:
        continue
    bond_energy_dict[tuple(sorted(bond_type.split('-')))] = float(energy) / 96

model_path = to_path('/home/v-junrenli/mechanism/bondnet/bondnet/prediction/pretrained/mechanism/20231009/')
model = load_model(model_path)
print('Using model: ', model_path)
model = model.to('cuda:0')


def calc_similarity(sml1, sml2):
    mol1 = Chem.MolFromSmiles(sml1)
    mol2 = Chem.MolFromSmiles(sml2)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2, useFeatures=True)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2, useFeatures=True)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_prediction(model, unit_converter, molecules, labels, extra_features):

    dataset = load_dataset(model_path, molecules, labels, extra_features)
    data_loader = DataLoaderReactionNetwork(dataset, batch_size=100, shuffle=False)

    feature_names = ["atom", "bond", "global"]

    # evaluate
    predictions = evaluate(model, feature_names, data_loader)

    # in case some entry fail
    if len(predictions) != len(dataset.failed):
        pred = []
        idx = 0
        for failed in dataset.failed:
            if failed:
                pred.append(None)
            else:
                pred.append(predictions[idx] * unit_converter)
                idx += 1
        predictions = pred
    else:
        predictions = np.asarray(predictions) * unit_converter

    return predictions

@timeout(15)
def predict_single_molecule(
    model_name,
    molecule,
    charge=0,
    ring_bond=True,
    one_per_iso_bond_group=False,
    write_result=False,
    figure_name="prediction.png",
    format=None,
    output='dict',
    ):
    """
    Make predictions for a single molecule.

    Breaking a bond may result in products with different combination of products
    charge, we report the smallest charge w.r.t. the product charge assignation.

    Args:
        model_name (str): The pre-trained model to use for making predictions. A model
            should be of the format `dataset/date`, e.g. `bdncm/20200808`,
            `pubchem/20200521`. It is possible to provide only the `dataset` part,
            and in this case, the latest model will be used.
        molecule (str): SMILES or InChI string or a path to a file storing these string.
        charge (int): charge of the molecule.
        ring_bond (bool): whether to make predictions for ring bond.
        one_per_iso_bond_group (bool): If `True`, keep one reaction for each
            isomorphic bond group (fragments obtained by breaking different bond
            are isomorphic to each other). If `False`, keep all.
        write_result (bool): whether to write the returned sdf to stdout.
        figure_name (str): the name of the figure to be created showing the bond energy.
        format (str): format of the molecule, if not provided, will guess based on the
            file extension.

    Returns:
        str: sdf string representing the molecules and energies.
    """

    model_info = get_model_info(model_path)
    allowed_charge = model_info["allowed_charge"]
    unit_converter = model_info["unit_conversion"]

    assert (
        charge in allowed_charge
    ), f"expect charge to be one of {allowed_charge}, but got {charge}"

    format = 'smiles'

    predictor = PredictionOneReactant(
        molecule, charge, format, allowed_charge, ring_bond, one_per_iso_bond_group
    )

    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(
        model, unit_converter, molecules, labels, extra_features
    )

    if output == 'dict':
        return predictor.get_bond_dict(predictions)

def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

def add_H(smiles, mapnum=set()):
    smls = smiles.split('.')
    result_mols = []
    already_protonated = False
    for sml in smls:
        other_smls = [s for s in smls if s != sml]
        raw_mol = Chem.MolFromSmiles(sml)
        if raw_mol is None:
            continue
        for atom in raw_mol.GetAtoms():
            if atom.GetFormalCharge() == 1 and atom.GetTotalNumHs() > 0:
                already_protonated = True
                break
        if not len(mapnum):
            mapnum = set([a.GetAtomMapNum() for a in raw_mol.GetAtoms()])
        add_site = [atom.GetIdx() for atom in raw_mol.GetAtoms() if atom.GetFormalCharge() == -1 and atom.GetAtomicNum() not in {9, 17, 35, 53} and atom.GetAtomMapNum() in mapnum]
        if len(add_site) == 0:
            add_site = [atom.GetIdx() for atom in raw_mol.GetAtoms() if atom.GetFormalCharge() == 0 and atom.GetExplicitValence() + atom.GetImplicitValence() < 4 and atom.GetAtomicNum() not in {9, 17, 35, 53} and atom.GetAtomMapNum() in mapnum]
        added_mol = Chem.MolFromSmiles(sml + '.[H]')
        for a_site in add_site:
            curr_mol = Chem.RWMol(added_mol)
            curr_mol.AddBond(a_site, len(raw_mol.GetAtoms()), Chem.BondType.SINGLE)
            curr_mol.GetAtomWithIdx(a_site).SetFormalCharge(curr_mol.GetAtomWithIdx(a_site).GetFormalCharge() + 1)
            try: 
                Chem.MolToSmiles(Chem.RemoveHs(curr_mol))
            except ValueError:
                continue
            try:
                Chem.SanitizeMol(curr_mol)
            except:
                continue
            result_mols.append(('.'.join([Chem.MolToSmiles(Chem.RemoveHs(curr_mol))] + other_smls), already_protonated))
    # print(list(set(result_mols)))
    return list(set(result_mols))

def remove_H(smiles, mapnum=set()):
    smls = smiles.split('.')
    result_mols = []
    existed_result = set()
    for idx, sml in enumerate(smls):
        other_smls = [s for s in smls if s != sml]
        raw_mol = Chem.MolFromSmiles(sml)
        if raw_mol is None:
            continue
        if not len(mapnum):
            mapnum = set([a.GetAtomMapNum() for a in raw_mol.GetAtoms()])
        remove_site = [atom.GetIdx() for atom in raw_mol.GetAtoms() if atom.GetFormalCharge() == 1 and atom.GetTotalNumHs() > 0 and not atom.GetIsAromatic() and atom.GetAtomMapNum() in mapnum]
        if len(remove_site) == 0:
            remove_site = [atom.GetIdx() for atom in raw_mol.GetAtoms() if atom.GetFormalCharge() == 0 and atom.GetTotalNumHs() > 0 and atom.GetExplicitValence() + atom.GetImplicitValence() <= 4 and atom.GetAtomMapNum() in mapnum]
        for r_site in remove_site:
            curr_mol = Chem.RWMol(Chem.AddHs(raw_mol))
            remove_H_index = [nei.GetIdx() for nei in curr_mol.GetAtomWithIdx(r_site).GetNeighbors() if nei.GetAtomicNum() == 1][0]
            curr_mol.RemoveBond(r_site, remove_H_index)
            if curr_mol.GetAtomWithIdx(r_site).GetFormalCharge() > 0:
                positive_site = 1
            elif curr_mol.GetAtomWithIdx(r_site).GetFormalCharge() == 0:
                if sum([a.GetFormalCharge() for a in curr_mol.GetAtoms()]) < 0:
                    positive_site = -1
                else:
                    positive_site = 0
            else:
                positive_site = -1
            curr_mol.GetAtomWithIdx(r_site).SetFormalCharge(curr_mol.GetAtomWithIdx(r_site).GetFormalCharge() - 1)
            curr_mol.RemoveAtom(remove_H_index)
            break_site = (smls[idx], r_site, remove_H_index, positive_site)
            if curr_mol.GetAtomWithIdx(r_site).GetHybridization() == Chem.HybridizationType.SP3 and curr_mol.GetAtomWithIdx(r_site).GetAtomicNum() == 6:
                for neighbor in curr_mol.GetAtomWithIdx(r_site).GetNeighbors():
                    if neighbor.GetHybridization() == Chem.HybridizationType.SP2 and curr_mol.GetBondBetweenAtoms(r_site, neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                        for n_neighbor in curr_mol.GetAtomWithIdx(neighbor.GetIdx()).GetNeighbors():
                            if n_neighbor.GetAtomicNum() == 8 and curr_mol.GetBondBetweenAtoms(neighbor.GetIdx(), n_neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                                break_site = (smls[idx], neighbor.GetIdx(), n_neighbor.GetIdx(), positive_site)
                                break
            try:
                Chem.SanitizeMol(curr_mol)
            except:
                continue
            if '.'.join([Chem.MolToSmiles(curr_mol)] + other_smls) not in existed_result:
                existed_result.add(break_site)
                result_mols.append(('.'.join([Chem.MolToSmiles(curr_mol)] + other_smls), break_site))
    # print(result_mols)
    return result_mols

def nucleophilic_attack_anion(smiles, mapnum=set(), intermol=False, doublebond_priority=False):
    result_mols = []
    existed_results = set()
    attack_sites = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    atom_in_mol = [len(Chem.MolFromSmiles(m).GetAtoms()) for m in smiles.split('.')]
    if not len(mapnum):
        mapnum = set([a.GetAtomMapNum() for a in mol.GetAtoms()])
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() == -1 and atom.GetAtomicNum() not in {9, 17, 35, 53} and atom.GetAtomMapNum() in mapnum:
            attack_sites.append(atom.GetIdx())
    sink_sites = [idx for idx in range(mol.GetNumAtoms()) if idx not in attack_sites and mol.GetAtomWithIdx(idx).GetAtomMapNum() in mapnum]
    for a_site in attack_sites:
        for s_site in sink_sites:
            for neighbor in mol.GetAtomWithIdx(s_site).GetNeighbors():
                if neighbor.GetIdx() != a_site and neighbor.GetAtomMapNum() in mapnum:
                    curr_mol = Chem.RWMol(mol)
                    # print(curr_mol.GetBondBetweenAtoms(s_site, neighbor.GetIdx()).GetBondType())
                    if curr_mol.GetAtomWithIdx(s_site).GetHybridization() == Chem.HybridizationType.SP2 and curr_mol.GetBondBetweenAtoms(s_site, neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                        # print(curr_mol.GetAtomWithIdx(s_site).GetHybridization(), curr_mol.GetBondBetweenAtoms(s_site, neighbor.GetIdx()).GetBondType(), curr_mol.GetAtomWithIdx(s_site).GetSymbol(), curr_mol.GetAtomWithIdx(neighbor.GetIdx()).GetSymbol())
                        continue
                    if curr_mol.GetBondBetweenAtoms(a_site, s_site) is None:
                        curr_mol.AddBond(a_site, s_site, Chem.BondType.SINGLE)
                    else:
                        curr_mol.GetBondBetweenAtoms(a_site, s_site).SetBondType(Chem.BondType.DOUBLE)
                    curr_mol.GetAtomWithIdx(a_site).SetFormalCharge(0)
                    if curr_mol.GetBondBetweenAtoms(s_site, neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                        # print('double bond', curr_mol.GetAtomWithIdx(s_site).GetSymbol(), curr_mol.GetAtomWithIdx(neighbor.GetIdx()).GetSymbol())
                        curr_mol.GetBondBetweenAtoms(s_site, neighbor.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                    elif curr_mol.GetAtomWithIdx(s_site).GetHybridization() == Chem.HybridizationType.SP3:
                        curr_mol.RemoveBond(s_site, neighbor.GetIdx())
                    else:
                        continue
                    break_site = (s_site, neighbor.GetIdx())
                    new_bond_type = (curr_mol.GetAtomWithIdx(a_site).GetSymbol(), curr_mol.GetAtomWithIdx(s_site).GetSymbol())
                    cumulative_minus = 0
                    for mol_idx, num in enumerate(atom_in_mol):
                        if s_site >= num + cumulative_minus:
                            cumulative_minus += num
                        else:
                            break_site = (smiles.split('.')[mol_idx], s_site - cumulative_minus, neighbor.GetIdx() - cumulative_minus, new_bond_type)
                            break
                    curr_mol.GetAtomWithIdx(neighbor.GetIdx()).SetFormalCharge(curr_mol.GetAtomWithIdx(neighbor.GetIdx()).GetFormalCharge() - 1)
                    try:
                        Chem.SanitizeMol(curr_mol)
                    except:
                        # print(Chem.MolToSmiles(curr_mol))
                        continue
                    if Chem.MolToSmiles(curr_mol) not in existed_results:
                        existed_results.add(Chem.MolToSmiles(curr_mol))
                        result_mols.append((Chem.MolToSmiles(curr_mol), break_site))
    # print(result_mols)
    '''if intermol:
        reactant_set = set(smiles.split('.'))
        result_mols = [res for res in result_mols if len(set(res[0].split('.')) & reactant_set) == 0]
    if doublebond_priority and len(result_mols) > 0:
        max_doublebond = max([res[0].count('=') for res in result_mols])
        result_mols = [res for res in result_mols if res[0].count('=') == max_doublebond]'''
    # print(result_mols)
    return result_mols

def nucleophilic_attack_neutral(smiles, mapnum=set(), intermol=False):
    result_mols = []
    attack_sites = []
    existed_results = set()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    atom_in_mol = [len(Chem.MolFromSmiles(m).GetAtoms()) for m in smiles.split('.')]
    if not len(mapnum):
        mapnum = set([a.GetAtomMapNum() for a in mol.GetAtoms()])
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() == 0 and atom.GetExplicitValence() + atom.GetImplicitValence() < 4 and atom.GetTotalNumHs() > 0 and atom.GetAtomicNum() not in {9, 17, 35, 53} and atom.GetAtomMapNum() in mapnum:
            attack_sites.append(atom.GetIdx())
    sink_sites = [idx for idx in range(mol.GetNumAtoms()) if idx not in attack_sites and mol.GetAtomWithIdx(idx).GetAtomMapNum() in mapnum]
    for a_site in attack_sites:
        for s_site in sink_sites:
            for neighbor in mol.GetAtomWithIdx(s_site).GetNeighbors():
                if neighbor.GetIdx() != a_site and neighbor.GetAtomMapNum() in mapnum:
                    curr_mol = Chem.RWMol(mol)
                    if curr_mol.GetAtomWithIdx(s_site).GetHybridization() == Chem.HybridizationType.SP2 and curr_mol.GetBondBetweenAtoms(s_site, neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                        continue
                    if curr_mol.GetBondBetweenAtoms(a_site, s_site) is None:
                        curr_mol.AddBond(a_site, s_site, Chem.BondType.SINGLE)
                    else:
                        curr_mol.GetBondBetweenAtoms(a_site, s_site).SetBondType(Chem.BondType.DOUBLE)
                    curr_mol.GetAtomWithIdx(a_site).SetFormalCharge(1)
                    if curr_mol.GetBondBetweenAtoms(s_site, neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                        curr_mol.GetBondBetweenAtoms(s_site, neighbor.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                    elif curr_mol.GetAtomWithIdx(s_site).GetHybridization() == Chem.HybridizationType.SP3:
                        curr_mol.RemoveBond(s_site, neighbor.GetIdx())
                    else:
                        continue
                    break_site = (s_site, neighbor.GetIdx())
                    new_bond_type = (curr_mol.GetAtomWithIdx(a_site).GetSymbol(), curr_mol.GetAtomWithIdx(s_site).GetSymbol())
                    cumulative_minus = 0
                    for mol_idx, num in enumerate(atom_in_mol):
                        if s_site >= num + cumulative_minus:
                            cumulative_minus += num
                        else:
                            break_site = (smiles.split('.')[mol_idx], s_site - cumulative_minus, neighbor.GetIdx() - cumulative_minus, new_bond_type)
                            break
                    curr_mol.GetAtomWithIdx(neighbor.GetIdx()).SetFormalCharge(curr_mol.GetAtomWithIdx(neighbor.GetIdx()).GetFormalCharge() - 1)
                    try:
                        Chem.SanitizeMol(curr_mol)
                    except:
                        continue
                    if Chem.MolToSmiles(curr_mol) not in existed_results:
                        existed_results.add(Chem.MolToSmiles(curr_mol))
                        result_mols.append((Chem.MolToSmiles(curr_mol), break_site))
    '''if intermol:
        reactant_set = set(smiles.split('.'))
        result_mols = [res for res in result_mols if len(set(res[0].split('.')) & reactant_set) == 0]'''
    return result_mols

def elimination(smiles, mapnum=set(), intermol=False):
    smls = smiles.split('.')
    result_mols = []
    existed_results = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    if not len(mapnum):
        mapnum = set([a.GetAtomMapNum() for a in mol.GetAtoms()])
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in {6, 7} and atom.GetTotalNumHs() > 0 and not atom.GetIsAromatic() and atom.GetAtomMapNum() in mapnum:
            for neighbor in mol.GetAtomWithIdx(atom.GetIdx()).GetNeighbors():
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE and atom.GetAtomMapNum() in mapnum:
                    for nei in mol.GetAtomWithIdx(neighbor.GetIdx()).GetNeighbors():
                        if nei.GetIdx() != atom.GetIdx() and mol.GetBondBetweenAtoms(neighbor.GetIdx(), nei.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                            curr_mol = Chem.RWMol(Chem.AddHs(mol))
                            curr_mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
                            curr_mol.RemoveBond(neighbor.GetIdx(), nei.GetIdx())
                            curr_mol.GetAtomWithIdx(nei.GetIdx()).SetFormalCharge(curr_mol.GetAtomWithIdx(nei.GetIdx()).GetFormalCharge() - 1)
                            for nei_H in curr_mol.GetAtomWithIdx(atom.GetIdx()).GetNeighbors():
                                if nei_H.GetAtomicNum() == 1:
                                    curr_mol.RemoveAtom(nei_H.GetIdx())
                                    break
                            break_site = ((neighbor.GetIdx(), nei.GetIdx()), (atom.GetIdx(), nei_H.GetIdx()))
                            new_bond_type = (curr_mol.GetAtomWithIdx(atom.GetIdx()).GetSymbol(), curr_mol.GetAtomWithIdx(neighbor.GetIdx()).GetSymbol())
                            if Chem.MolToSmiles(curr_mol) not in existed_results:
                                existed_results.append(Chem.MolToSmiles(curr_mol))
                                result_mols.append((Chem.MolToSmiles(curr_mol), break_site, new_bond_type))
    if intermol:
        reactant_set = set(smiles.split('.'))
        result_mols = [res for res in result_mols if len(set(res[0].split('.')) & reactant_set) == 0]
    return result_mols


def perform_step(smiles, mapnum=set()):
    return add_H(smiles, mapnum) + remove_H(smiles, mapnum) + nucleophilic_attack_anion(smiles, mapnum, intermol=True) + nucleophilic_attack_neutral(smiles, mapnum, intermol=True)# + elimination(smiles, mapnum, intermol=True)


def perform_step_with_prob(smiles, energy_data={}, mapnum={}):
    # print('performing step on', smiles)
    # print(calc_similarity(smiles, 'CC(O)(C(=O)NCC(F)(F)C(F)(F)F)C(=O)N[C@@H]1C(=O)N(CCOCc2ccccc2)c2ccccc2-c2ccccc21'))
    # print(len(energy_data))
    results = perform_step(smiles, mapnum)
    # print(mapnum)
    # print(results)
    t0 = time.time()
    # mol = Chem.AddHs(Chem.MolFromSmiles(smiles.split('.')[1]))
    # for idx, atom in enumerate(mol.GetAtoms()):
    #     atom.SetAtomMapNum(idx + 1)
    # print(Chem.MolToSmiles(mol))
    orig_smls = smiles.split('.')
    orig_mols = [Chem.MolFromSmiles(s) for s in orig_smls]
    orig_chrg = [Chem.GetFormalCharge(m) for m in orig_mols]
    energy_dict = {}
    # print('finished preparation', time.time() - t0)
    for s, m, c in zip(orig_smls, orig_mols, orig_chrg):
        c_clipped = int(max(min(c, 1), -1))
        if s in energy_data.keys():
            continue
        if len(Chem.AddHs(m).GetAtoms()) > 1:
            # print('bondnet calculation on', s, time.time() - t0)
            try:
                # print(s)
                curr_energy_dict = predict_single_molecule('mechanism', s, c_clipped, True, False, output='dict')
                # print(s, curr_energy_dict)
                energy_data[s] = curr_energy_dict
            except:
                energy_data[s] = {}
        else:
            energy_data[s] = {}
    for result in results:
        if isinstance(result[1], str) or isinstance(result[1], tuple):
            result_sml = result[0]
            # try:
            if isinstance(result[1][-1], int) and result[1][-1] == 1:
                energy = 0.45
            else:
                mol_energy = energy_data[result[1][0]]
                if len(mol_energy) == 0:
                    # print('Failed on', result[1][0])
                    energy = 9999
                else:
                    if isinstance(result[1][-1], int):
                        if result[1][-1] == 0:
                            bond_energy = mol_energy[tuple(sorted(result[1][1:3]))]
                            energy = bond_energy * 0.3 if bond_energy is not None else 9999
                        else:
                            bond_energy = mol_energy[tuple(sorted(result[1][1:3]))]
                            energy = bond_energy * 0.6 if bond_energy is not None else 9999
                    else:
                        # print(result)
                        bond_energy = mol_energy[tuple(sorted(result[1][1:3]))]
                        energy = (bond_energy * 0.6 - bond_energy_dict[tuple(sorted(result[1][-1]))] * 0.3) if bond_energy is not None else 9999
            energy_dict[result_sml] = energy
            # print(energy)
    # print('finished bondnet', time.time() - t0)
    # print(smiles, energy_dict)
    for result in results:
        if isinstance(result[1], bool):
            energy_dict[result[0]] = 0.45 if not result[1] else 1.2
    # print(smiles, energy_dict)
    prob_dict = {}
    for k, v in energy_dict.items():
        prob_dict[k] = np.exp(-v * 96000 / 8.314 / 298)
    cumulative_prob = sum(prob_dict.values())
    for k, v in prob_dict.items():
        prob_dict[k] = v / cumulative_prob
    # print(smiles, prob_dict)
    # print(energy_data)
    # print('finished', time.time() - t0)
    return prob_dict, energy_dict, energy_data
            
if __name__ == '__main__':
    sml = 'CCCC([O-])CC(C)=O'
    prob_dict, energy_dict = perform_step_with_prob(sml)
    # print sorted results
    # print(Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles('.'.join(list(prob_dict.keys()))))))

    # for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
    #     print(k, v)

    # print('\n')
    
    # for k, v in sorted(energy_dict.items(), key=lambda x: x[1]):
    #     print(k, v)