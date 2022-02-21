from models import *


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


def build_cls_model(model_name):
    if model_name == 'seresnext':
        model = SeResNext_Double()
    elif model_name == 'dpn':
        model = Dpn_Double()
    elif model_name == 'resnet':
        model = ResNet_Double()
    elif model_name == 'senet':
        model = SeNet_Double()
    else:
        raise Excepion("not support this model.")

    return model
