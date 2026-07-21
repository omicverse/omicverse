from torch import nn
import torch

class EmbLayer(nn.Module):
    def __init__(self,
                 hidden_dim=None,
                 aggregate='sum',
                 bias=True,
                 activation=None,
                 ntypes=None,
                 dropout=0.0,
                 conv_dict=None,
                 ):
        super(EmbLayer, self).__init__()
        from dgl.nn.pytorch import HeteroGraphConv

        self.conv = HeteroGraphConv(conv_dict, aggregate=aggregate)
        if ntypes is not None:
            self.use_layernorm = True
            self.norm_layers = nn.ModuleDict({
                ntype: nn.LayerNorm(hidden_dim, elementwise_affine=True)
                for ntype in ntypes
            })
        else:
            self.use_layernorm = False

        self.bias = bias
        if bias:
            self.h_bias = nn.ParameterDict()
            for ntype in ntypes:
                self.h_bias[ntype] = nn.Parameter(torch.Tensor(hidden_dim))
                nn.init.zeros_(self.h_bias[ntype])

        self.activation = activation

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs: dict,
                norm=True, bias=True, activate=True):

        hs = self.conv(g, inputs)

        def _apply(ntype, h):
            if self.use_layernorm and norm:
                h = self.norm_layers[ntype[-4:]](h)
            if self.bias and bias:
                h = h + self.h_bias[ntype[-4:]]
            if self.activation and activate:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}
