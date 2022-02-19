from logging import raiseExceptions
from models import *

def build_loc_model(model_name):
    if model_name == 'seresnext50':
        model = SeResNext50_Unet_Loc()
    elif model_name == 'dpn92':
        model = Dpn92_Unet_Loc()
    elif model_name == 'res34':
        model = Res34_Unet_Loc()
    elif model_name == 'senet154':
        model = SeNet154_Unet_Loc()
    else:
        raiseExceptions("not support this model.")

    return model