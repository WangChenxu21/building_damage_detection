#!/usr/bin/env python
# coding:utf-8

import json
import codecs

import numpy as np
import torch.nn as nn

from models.hierarchical_encoder.graphcnn import HierarchyGCN
from models.hierarchical_encoder.tree_lstm import WeightedHierarchicalTreeLSTMEndtoEnd
from models.hierarchical_encoder.tree import Tree


MODEL_MODULE = {
    'TreeLSTM': WeightedHierarchicalTreeLSTMEndtoEnd,
    'GCN': HierarchyGCN
}


def get_hierarchy_relations(hierar_taxonomy, label_map, root=None, fortree=False):
    """
    get parent-children relationships from given hierar_taxonomy
    parent_label \t child_label_0 \t child_label_1 \n
    :param hierar_taxonomy: Str, file path of hierarchy taxonomy
    :param label_map: Dict, label to id
    :param root: Str, root tag
    :param fortree: Boolean, True : return label_tree -> List
    :return: label_tree -> List[Tree], hierar_relation -> Dict{parent_id: List[child_id]}
    """
    label_tree = dict()
    label_tree[0] = root
    hierar_relations = {}
    with codecs.open(hierar_taxonomy, "r", "utf8") as f:
        for line in f:
            line_split = line.rstrip().split('\t')
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label not in label_map:
                if fortree and parent_label == 'Root':
                    parent_label_id = -1
                else:
                    continue
            else:
                parent_label_id = label_map[parent_label]
            children_label_ids = [label_map[child_label] \
                                  for child_label in children_label if child_label in label_map]
            hierar_relations[parent_label_id] = children_label_ids
            if fortree:
                assert (parent_label_id + 1) in label_tree
                parent_tree = label_tree[parent_label_id + 1]

                for child in children_label_ids:
                    assert (child + 1) not in label_tree
                    child_tree = Tree(child)
                    parent_tree.add_child(child_tree)
                    label_tree[child + 1] = child_tree
    if fortree:
        return hierar_relations, label_tree
    else:
        return hierar_relations


class StructureEncoder(nn.Module):
    def __init__(self,
                 hierarchy_path,
                 hierarchy_prob_json,
                 label_map,
                 graph_model_type,
                 in_dim,
                 dropout):
        """
        Structure Encoder module
        :param config: helper.configure, Configure Object
        :param label_map: data_modules.vocab.v2i['label']
        :param device: torch.device, config.train.device_setting.device
        :param graph_model_type: Str, model_type, ['TreeLSTM', 'GCN']
        """
        super(StructureEncoder, self).__init__()

        self.label_map = label_map
        self.root = Tree(-1)

        self.hierarchical_label_dict, self.label_trees = get_hierarchy_relations(hierarchy_path,
                                                                                 self.label_map,
                                                                                 root=self.root,
                                                                                 fortree=True)
        f = open(hierarchy_prob_json, 'r')
        hierarchy_prob_str = f.readlines()
        f.close()
        self.hierarchy_prob = json.loads(hierarchy_prob_str[0])
        self.node_prob_from_parent = np.zeros((len(self.label_map), len(self.label_map)))
        self.node_prob_from_child = np.zeros((len(self.label_map), len(self.label_map)))

        for p in self.hierarchy_prob.keys():
            if p == 'Root':
                continue
            for c in self.hierarchy_prob[p].keys():
                # self.hierarchy_id_prob[self.label_map[p]][self.label_map[c]] = self.hierarchy_prob[p][c]
                self.node_prob_from_child[int(self.label_map[p])][int(self.label_map[c])] = 1.0
                self.node_prob_from_parent[int(self.label_map[c])][int(self.label_map[p])] = self.hierarchy_prob[p][c]
        #  node_prob_from_parent: row means parent, col refers to children

        self.model = MODEL_MODULE[graph_model_type](num_nodes=len(self.label_map),
                                                    in_matrix=self.node_prob_from_child,
                                                    out_matrix=self.node_prob_from_parent,
                                                    in_dim=in_dim,
                                                    dropout=dropout,
                                                    device='cuda',
                                                    root=self.root,
                                                    hierarchical_label_dict=self.hierarchical_label_dict,
                                                    label_trees=self.label_trees)

    def forward(self, inputs):
        return self.model(inputs)
