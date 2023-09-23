from torch.nn.utils.spectral_norm import spectral_norm
import torch.nn as nn

def build_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == 'bnorm':
        return nn.BatchNorm1d(num_feats)
    elif norm_type == 'snorm':
        spectral_norm(param)
        return None
    elif norm_type is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm_type)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1DResBlock') != -1:
        # print('Initializing weights of convresblock to 0.0, 0.02')
        for k, p in m.named_parameters():
            if 'weight' in k and 'conv' in k:
                p.data.normal_(0.0, 0.02)
    elif classname.find('Conv1d') != -1:
        # print('Initialzing weight to 0.0, 0.02 for module: ', m)
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            # print('bias to 0 for module: ', m)
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        # print('Initializing FC weight to xavier uniform')
        nn.init.xavier_uniform_(m.weight.data)