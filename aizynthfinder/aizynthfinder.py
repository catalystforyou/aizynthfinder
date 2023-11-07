""" Module containing a class that is the main interface the retrosynthesis tool.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

from tqdm import tqdm

from aizynthfinder.analysis import (
    RouteCollection,
    RouteSelectionArguments,
    TreeAnalysis,
)
from aizynthfinder.chem import FixedRetroReaction, Molecule, TreeMolecule
from aizynthfinder.context.config import Configuration
from aizynthfinder.context.policy.expansion_strategies import ElementaryStep
from aizynthfinder.context.stock import ProductStock
from aizynthfinder.reactiontree import ReactionTreeFromExpansion
from aizynthfinder.search.andor_trees import AndOrSearchTreeBase
from aizynthfinder.search.mcts import MctsSearchTree
from aizynthfinder.utils.exceptions import MoleculeException
from aizynthfinder.utils.loading import load_dynamic_class
from rdkit import Chem

# This must be imported first to setup logging for rdkit, tensorflow etc
from aizynthfinder.utils.logging import logger

if TYPE_CHECKING:
    from aizynthfinder.chem import RetroReaction
    from aizynthfinder.utils.type_utils import (
        Callable,
        Dict,
        List,
        Optional,
        StrDict,
        Tuple,
        Union,
    )

def get_reacting_core(rs, p, buffer=1):
    '''
    use molAtomMapNumber of molecules
    buffer: neighbor to be cosidered as reacting center
    return: atomidx of reacting core
    '''
    def _get_buffer(m, cores, buffer):
        neighbors = set(cores)

        for i in range(buffer):
            neighbors_temp = list(neighbors)
            for c in neighbors_temp:
                neighbors.update([n.GetIdx()
                                 for n in m.GetAtomWithIdx(c).GetNeighbors()])

        neighbors = [m.GetAtomWithIdx(x).GetAtomMapNum() for x in neighbors]

        return neighbors

    def _verify_changes(r_mols, p_mol, core_rs, core_p, discard_rs):
        core_rs = core_rs + discard_rs
        r_mols, p_mol = Chem.AddHs(r_mols), Chem.AddHs(p_mol)
        remove_rs_idx, remove_p_idx = [], []
        for atom in r_mols.GetAtoms():
            if atom.GetIdx() in core_rs or atom.GetAtomMapNum() == 0:
                remove_rs_idx.append(atom.GetIdx())
        for atom in p_mol.GetAtoms():
            if atom.GetIdx() in core_p or atom.GetAtomMapNum() == 0:
                remove_p_idx.append(atom.GetIdx())
        r_mols, p_mol = Chem.RWMol(r_mols), Chem.RWMol(p_mol)
        for idx in sorted(remove_rs_idx, reverse=True):
            r_mols.RemoveAtom(idx)
        for idx in sorted(remove_p_idx, reverse=True):
            p_mol.RemoveAtom(idx)
        return Chem.MolToSmiles(r_mols) == Chem.MolToSmiles(p_mol)

    r_mols = Chem.MolFromSmiles(rs)
    p_mol = Chem.MolFromSmiles(p)

    rs_dict = {a.GetAtomMapNum(): a for a in r_mols.GetAtoms()}
    rs_bond_dict = {'{}-{}'.format(*sorted([b.GetBeginAtom().GetAtomMapNum(),
                                            b.GetEndAtom().GetAtomMapNum()])): b
                    for b in r_mols.GetBonds()}

    p_dict = {a.GetAtomMapNum(): a for a in p_mol.GetAtoms()}
    p_bond_dict = {'{}-{}'.format(*sorted([b.GetBeginAtom().GetAtomMapNum(),
                                          b.GetEndAtom().GetAtomMapNum()])): b
                   for b in p_mol.GetBonds()}

    rs_reactants = []
    for r_smiles in rs.split('.'):
        for a in Chem.MolFromSmiles(r_smiles).GetAtoms():
            if a.GetAtomMapNum() in p_dict:
                rs_reactants.append(r_smiles)
                break
    rs_reactants = '.'.join(rs_reactants)

    core_mapnum = set()
    core_bond = set()
    for a_map in p_dict:

        a_neighbor_in_p = set([a.GetAtomMapNum()
                              for a in p_dict[a_map].GetNeighbors()])
        a_neighbor_in_rs = set([a.GetAtomMapNum()
                               for a in rs_dict[a_map].GetNeighbors()])
        if a_neighbor_in_p != a_neighbor_in_rs:
            core_mapnum.add(a_map)
        else:
            for a_neighbor in a_neighbor_in_p:
                b_in_p = p_mol.GetBondBetweenAtoms(
                    p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx())
                b_in_r = r_mols.GetBondBetweenAtoms(
                    rs_dict[a_neighbor].GetIdx(), rs_dict[a_map].GetIdx())
                if b_in_p.GetBondType() != b_in_r.GetBondType():
                    core_bond.add(b_in_r.GetIdx())
                    core_mapnum.add(a_map)

    for k, v in rs_bond_dict.items():
        if (k not in p_bond_dict.keys() and k != '0-0') or (k.split('_')[0] in core_mapnum and k.split('_')[1] in core_mapnum):
            # the marked bond changes here only contain those between heavy atoms
            core_bond.add(v.GetIdx())

    core_rs = _get_buffer(r_mols, [rs_dict[a].GetIdx()
                          for a in core_mapnum], buffer)
    core_p = _get_buffer(p_mol, [p_dict[a].GetIdx()
                         for a in core_mapnum], buffer)

    fatom_index_rs = \
        {a.GetAtomMapNum(): a.GetIdx() for a in r_mols.GetAtoms()}
    fatom_index_p = \
        {a.GetAtomMapNum(): a.GetIdx() for a in p_mol.GetAtoms()}

    # core_rs = [fatom_index_rs[x] for x in core_rs]
    # core_p = [fatom_index_p[x] for x in core_p]

    discard_rs = []

    for atom in r_mols.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            discard_rs.append(atom.GetIdx())

    if _verify_changes(r_mols, p_mol, core_rs, core_p, discard_rs):
        return core_rs# , discard_rs, list(core_bond), core_p
    else:
        return core_rs#[], [], [], []
        # return core_rs, discard_rs, list(core_bond), core_p

class AiZynthFinder:
    """
    Public API to the aizynthfinder tool

    If instantiated with the path to a yaml file or dictionary of settings
    the stocks and policy networks are loaded directly.
    Otherwise, the user is responsible for loading them prior to
    executing the tree search.

    :ivar config: the configuration of the search
    :ivar expansion_policy: the expansion policy model
    :ivar filter_policy: the filter policy model
    :ivar stock: the stock
    :ivar scorers: the loaded scores
    :ivar tree: the search tree
    :ivar analysis: the tree analysis
    :ivar routes: the top-ranked routes
    :ivar search_stats: statistics of the latest search

    :param configfile: the path to yaml file with configuration (has priority over configdict), defaults to None
    :param configdict: the config as a dictionary source, defaults to None
    """

    def __init__(self, configfile: str = None, configdict: StrDict = None) -> None:
        self._logger = logger()

        if configfile:
            self.config = Configuration.from_file(configfile)
        elif configdict:
            self.config = Configuration.from_dict(configdict)
        else:
            self.config = Configuration()

        self.expansion_policy = ElementaryStep()
        self.filter_policy = self.config.filter_policy
        self.scorers = self.config.scorers
        self.tree: Optional[Union[MctsSearchTree, AndOrSearchTreeBase]] = None
        self._target_mol: Optional[Molecule] = None
        self.search_stats: StrDict = dict()
        self.routes = RouteCollection([])
        self.analysis: Optional[TreeAnalysis] = None

    @property
    def target_smiles(self) -> str:
        """The SMILES representation of the molecule to predict routes on."""
        if not self._target_mol:
            return ""
        return self._target_mol.smiles

    @target_smiles.setter
    def target_smiles(self, smiles: str) -> None:
        reactant, product = smiles.split('>')[0], smiles.split('>')[-1]
        mapnum = get_reacting_core(reactant, product)
        self.mapnum = set(mapnum) if mapnum != set([0]) else set()
        self.target_mol = Molecule(smiles=reactant)
        # self.stock = ProductStock(product=product)
        self.config.stock = ProductStock(product=product)

    @property
    def target_mol(self) -> Optional[Molecule]:
        """The molecule to predict routes on"""
        return self._target_mol

    @target_mol.setter
    def target_mol(self, mol: Molecule) -> None:
        self.tree = None
        self._target_mol = mol

    @property
    def mapnum(self) -> set:
        """The mapnum of the reacting core"""
        return self._mapnum

    @mapnum.setter
    def mapnum(self, mapnum: set) -> None:
        self._mapnum = mapnum

    def build_routes(
        self, selection: RouteSelectionArguments = None, scorer: str = "state score"
    ) -> None:
        """
        Build reaction routes

        This is necessary to call after the tree search has completed in order
        to extract results from the tree search.

        :param selection: the selection criteria for the routes
        :param scorer: a reference to the object used to score the nodes
        :raises ValueError: if the search tree not initialized
        """
        if not self.tree:
            raise ValueError("Search tree not initialized")

        self.analysis = TreeAnalysis(self.tree, scorer=self.scorers[scorer])
        config_selection = RouteSelectionArguments(
            nmin=self.config.post_processing.min_routes,
            nmax=self.config.post_processing.max_routes,
            return_all=self.config.post_processing.all_routes,
        )
        self.routes = RouteCollection.from_analysis(
            self.analysis, selection or config_selection
        )

    def extract_statistics(self) -> StrDict:
        """Extracts tree statistics as a dictionary"""
        if not self.analysis:
            return {}
        stats = {
            "target": self.target_smiles,
            "search_time": self.search_stats["time"],
            "first_solution_time": self.search_stats.get("first_solution_time", 0),
            "first_solution_iteration": self.search_stats.get(
                "first_solution_iteration", 0
            ),
        }
        stats.update(self.analysis.tree_statistics())
        return stats

    def prepare_tree(self) -> None:
        """
        Setup the tree for searching

        :raises ValueError: if the target molecule was not set
        """
        if not self.target_mol:
            raise ValueError("No target molecule set")

        try:
            self.target_mol.sanitize()
        except MoleculeException:
            raise ValueError("Target molecule unsanitizable")

        '''self.stock.reset_exclusion_list()
        if self.config.exclude_target_from_stock and self.target_mol in self.stock:
            self.stock.exclude(self.target_mol)
            self._logger.debug("Excluding the target compound from the stock")'''

        self._setup_search_tree()
        self.analysis = None
        self.routes = RouteCollection([])

    def stock_info(self) -> StrDict:
        """
        Return the stock availability for all leaf nodes in all collected reaction trees

        The key of the return dictionary will be the SMILES string of the leaves,
        and the value will be the stock availability

        :return: the collected stock information.
        """
        if not self.analysis:
            return {}
        _stock_info = {}
        for tree in self.routes.reaction_trees:
            for leaf in tree.leafs():
                if leaf.smiles not in _stock_info:
                    _stock_info[leaf.smiles] = self.stock.availability_list(leaf)
        return _stock_info

    def tree_search(self, show_progress: bool = False) -> float:
        """
        Perform the actual tree search

        :param show_progress: if True, shows a progress bar
        :return: the time past in seconds
        """
        if not self.tree:
            self.prepare_tree()
        # This is for type checking, prepare_tree is creating it.
        assert self.tree is not None
        self.search_stats = {"returned_first": False, "iterations": 0}

        time0 = time.time()
        i = 1
        self._logger.debug("Starting search")
        time_past = time.time() - time0

        if show_progress:
            pbar = tqdm(total=self.config.iteration_limit, leave=False)

        while time_past < self.config.time_limit and i <= self.config.iteration_limit:
            if show_progress:
                pbar.update(1)
            self.search_stats["iterations"] += 1

            try:
                is_solved = self.tree.one_iteration()
                print(is_solved)
            except StopIteration:
                break

            if is_solved and "first_solution_time" not in self.search_stats:
                self.search_stats["first_solution_time"] = time.time() - time0
                self.search_stats["first_solution_iteration"] = i

            if self.config.return_first and is_solved:
                self._logger.debug("Found first solved route")
                self.search_stats["returned_first"] = True
                break
            i = i + 1
            time_past = time.time() - time0

        if show_progress:
            pbar.close()
        time_past = time.time() - time0
        self._logger.debug("Search completed")
        self.search_stats["time"] = time_past
        return time_past

    def _setup_search_tree(self) -> None:
        self._logger.debug("Defining tree root: %s" % self.target_smiles)
        if self.config.search_algorithm.lower() == "mcts":
            self.tree = MctsSearchTree(
                root_smiles=self.target_smiles, config=self.config, mapnum=self.mapnum
            )
        else:
            cls = load_dynamic_class(self.config.search_algorithm)
            self.tree = cls(root_smiles=self.target_smiles, config=self.config)


class AiZynthExpander:
    """
    Public API to the AiZynthFinder expansion and filter policies

    If instantiated with the path to a yaml file or dictionary of settings
    the stocks and policy networks are loaded directly.
    Otherwise, the user is responsible for loading them prior to
    executing the tree search.

    :ivar config: the configuration of the search
    :ivar expansion_policy: the expansion policy model
    :ivar filter_policy: the filter policy model

    :param configfile: the path to yaml file with configuration (has priority over configdict), defaults to None
    :param configdict: the config as a dictionary source, defaults to None
    """

    def __init__(self, configfile: str = None, configdict: StrDict = None) -> None:
        self._logger = logger()

        if configfile:
            self.config = Configuration.from_file(configfile)
        elif configdict:
            self.config = Configuration.from_dict(configdict)
        else:
            self.config = Configuration()

        self.expansion_policy = self.config.expansion_policy
        self.filter_policy = self.config.filter_policy
        self.stats: StrDict = {}

    def do_expansion(
        self,
        smiles: str,
        return_n: int = 5,
        filter_func: Callable[[RetroReaction], bool] = None,
    ) -> List[Tuple[FixedRetroReaction, ...]]:
        """
        Do the expansion of the given molecule returning a list of
        reaction tuples. Each tuple in the list contains reactions
        producing the same reactants. Hence, nested structure of the
        return value is way to group reactions.

        If filter policy is setup, the probability of the reactions are
        added as metadata to the reaction.

        The additional filter functions makes it possible to do customized
        filtering. The callable should take as only argument a `RetroReaction`
        object and return True if the reaction can be kept or False if it should
        be removed.

        :param smiles: the SMILES string of the target molecule
        :param return_n: the length of the return list
        :param filter_func: an additional filter function
        :return: the grouped reactions
        """
        self.stats = {"non-applicable": 0}

        mol = TreeMolecule(parent=None, smiles=smiles)
        actions, _ = self.expansion_policy.get_actions([mol])
        results: Dict[Tuple[str, ...], List[FixedRetroReaction]] = defaultdict(list)
        for action in actions:
            reactants = action.reactants
            if not reactants:
                self.stats["non-applicable"] += 1
                continue
            if filter_func and not filter_func(action):
                continue
            for name in self.filter_policy.selection or []:
                if hasattr(self.filter_policy[name], "feasibility"):
                    _, feasibility_prob = self.filter_policy[name].feasibility(action)
                    action.metadata["feasibility"] = float(feasibility_prob)
                    break
            action.metadata["expansion_rank"] = len(results) + 1
            unique_key = tuple(sorted(mol.inchi_key for mol in reactants[0]))
            if unique_key not in results and len(results) >= return_n:
                continue
            rxn = next(ReactionTreeFromExpansion(action).tree.reactions())  # type: ignore
            results[unique_key].append(rxn)
        return [tuple(reactions) for reactions in results.values()]
