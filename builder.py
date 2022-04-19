import numpy as np

from models import *
from models.hierarchical_encoder.tree import Tree


def build_loc_model(model_name):
    if model_name == 'seresnext':
        model = SeResNext_Loc()
    elif model_name == 'dpn':
        model = Dpn_Loc()
    elif model_name == 'resnet':
        model = ResNet_Loc()
    elif model_name == 'senet':
        model = SeNet_Loc()
    else:
        raise Exception("not support this model.")

    return model


def build_cls_model(model_name, **kwargs):
    if model_name == 'seresnext':
        model = SeResNext_Double(**kwargs)
    elif model_name == 'dpn':
        model = Dpn_Double(**kwargs)
    elif model_name == 'resnet':
        model = ResNet_Double(**kwargs)
    elif model_name == 'senet':
        model = SeNet_Double(**kwargs)
    else:
        raise Exception("not support this model.")

    return model


def build_cls_hierarchy_model(model_name,
                              hierarchy_path,
                              hierarchy_prob_json,
                              graph_model_type,
                              direction):
    label_map = {'no_building': 0,
                 'is_building': 1,
                 'no_damage': 2,
                 'is_damage': 3,
                 'minor_damage': 4,
                 'major_damage': 5,
                 'destroyed_damage': 6,}

    if model_name == 'seresnext':
        model = SeResNext_Double_Hierarchy(hierarchy_path, hierarchy_prob_json,
                label_map, graph_model_type, 1, direction)
    elif model_name == 'dpn':
        model = Dpn_Double_Hierarchy(hierarchy_path, hierarchy_prob_json,
                label_map, graph_model_type, 1, direction)
    elif model_name == 'resnet':
        model = ResNet_Double_Hierarchy(hierarchy_path, hierarchy_prob_json,
                label_map, graph_model_type, 1, direction)
    elif model_name == 'senet':
        model = SeNet_Double_Hierarchy(hierarchy_path, hierarchy_prob_json,
                label_map, graph_model_type, 1, direction)
    else:
        raise Exception("not support this model.")

    return model

