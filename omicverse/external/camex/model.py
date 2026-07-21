import torch
import dgl
import torch.nn as nn
import dgl.function as fn
import numpy as np

from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch import GraphConv, GATConv
from torch.autograd import Function
from torch.distributions import Normal


class GAE(nn.Module):
    def __init__(self, encoder, decoder, classifier, domain_classifier, cluster_classifier=None, mlp=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.domain_classifier = domain_classifier
        self.cluster_classifier = cluster_classifier
        self.mlp = mlp

    def forward(self, graph, feature):
        """
        just for feature importance selection
        :param graph:
        :param feature:
        :return:
        """
        h_hidden = self.encoder(graph, feature)
        h_class = self.classifier(graph, h_hidden)
        return h_class

    def reset_parameters(self):
        for name, submodule in self.named_modules():
            # if type(submodule) in [nn.Linear, GraphConv, GATConv]:
            if isinstance(submodule, (nn.Linear, GraphConv, GATConv)):
                submodule.reset_parameters()


# class RGNNEncoder(nn.Module):
#     def __init__(self, relation, dim_in, dim_hidden, layer_num, res=True, share=True, graph=None):
#         super().__init__()
#
#         self.relation = set(relation)
#         self.dim_in = dim_in
#         self.dim_hidden = dim_hidden
#         self.layer_num = layer_num
#         self.head_num = 8
#
#         self.relation_selfloop = []
#         self.relation_emb = []
#         for item in relation:
#             src, rel, dst = item
#             if rel in {'selfloop'}:
#                 self.relation_selfloop.append(item)
#             if rel in {'express', 'selfloop'}:
#                 self.relation_emb.append(item)
#
#         # emb_layer
#         self.emb_layer = {item[1]: GraphConv(dim_in, dim_hidden, weight=True, bias=True, activation=nn.LeakyReLU(0.02)) for item in self.relation_emb}
#         # self.emb_layer = {item[1]: GraphConv(dim_in, dim_hidden, weight=True, bias=True, activation=nn.LeakyReLU(0.02), norm='right') for item in self.relation_emb}
#         self.emb = HeteroGraphConv({rel: self.emb_layer[rel] for src, rel, dst in self.relation_emb}, aggregate='mean')
#
#         self.emb_norm = nn.ModuleDict({'cell': nn.LayerNorm(dim_in), 'gene': nn.LayerNorm(dim_in)})
#         self.emb_bias = nn.ParameterDict({'cell': nn.Parameter(torch.Tensor(dim_hidden)), 'gene': nn.Parameter(torch.Tensor(dim_hidden))})
#         self.emb_ac = nn.LeakyReLU()
#         self.emb_dp = nn.Dropout(0.3)
#         # self._build_embed_layer()
#
#         # share
#         self.conv_layer = {item[1]: GraphConv(dim_hidden, dim_hidden, weight=True, bias=True, activation=nn.LeakyReLU(0.02)) for item in self.relation}
#         self.conv = HeteroGraphConv({rel: self.conv_layer[rel] for src, rel, dst in self.relation}, aggregate='mean')
#
#         self.convs_layer = [{item[1]: GraphConv(dim_hidden, dim_hidden, weight=True, bias=True, activation=nn.LeakyReLU(0.02)) for item in self.relation}
#                             for i in range(layer_num + 1)]
#         self.convs = nn.ModuleList([HeteroGraphConv({rel: self.convs_layer[i][rel] for
#                                                     src, rel, dst in relation}, aggregate='mean') for i in range(layer_num + 1)])
#
#         self.conv_norm = nn.ModuleDict({'cell': nn.LayerNorm(dim_hidden), 'gene': nn.LayerNorm(dim_hidden)})
#         self.conv_dp = nn.Dropout(0.3)
#
#         assert res in [True, False]
#         self.res = res
#         assert share in [True, False]
#         self.share = share
#
#     def forward(self, graph, feature):
#
#         h = {k: self.emb_norm[k[-4:]](v) for k, v in feature.items()}
#         sub_graph_emb = dgl.edge_type_subgraph(graph, self.relation_emb)
#         emb = self.emb(sub_graph_emb, h)
#
#         # h = {k: self.emb_bias[k[-4:]] + v for k, v in h.items()}
#         # h = {k: self.emb_ac(v) for k, v in h.items()}
#         # emb = {k: self.emb_dp(v) for k, v in h.items()}
#
#         # conv
#         for i in range(1, self.layer_num + 1):
#             h = {k: self.conv_norm[k[-4:]](v) for k, v in emb.items()}  # TODO cell gene
#             if self.share:
#                 h = self.conv(graph, h)
#             else:
#                 h = self.convs[i](graph, h)
#             h = {k: self.conv_dp(v.float()) for k, v in h.items()}
#             if self.res:
#                 h = {k: v + emb[k] for k, v in h.items()}  # TODO identity
#
#         h_gnn = nn.Identity()(h)
#         return h_gnn
#
#     def _build_embed_layer(self, **kwds):
#         embed_params = dict(
#             hidden_dim=self.dim_hidden,
#             aggregate='sum',
#             bias=True,  #
#             activation=nn.LeakyReLU(),  # None #nn.LeakyReLU(0.2)
#             ntypes=['cell', 'gene'],  # None, #
#             dropout=0.2,
#             conv_dict=self.emb_layer
#         )
#         if len(kwds) > 0:
#             embed_params.update(**kwds)
#         self.emb = EmbLayer(**embed_params)


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


class RGNNEncoder(nn.Module):
    def __init__(self, relation, dim_in, dim_hidden, layer_num, res=True, share=True):
        super().__init__()

        self.relation = set(relation)
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.layer_num = layer_num
        self.head_num = 8

        self.relation_selfloop = []
        self.relation_emb = []
        for item in relation:
            src, rel, dst = item
            if rel in {'selfloop'}:
                self.relation_selfloop.append(item)
            if rel in {'express', 'selfloop'}:
                self.relation_emb.append(item)

        # emb_layer
        # self.emb_layer = GraphConv(dim_in, dim_hidden, weight=True, bias=True, activation=nn.LeakyReLU(0.02))
        # self.emb = HeteroGraphConv({rel: self.emb_layer for src, rel, dst in self.relation_emb}, aggregate='mean')
        self.emb_layer = {item[1]: GraphConv(dim_in, dim_hidden, weight=True, bias=True, activation=nn.LeakyReLU(0.02)) for item in self.relation_emb}
        self.emb = HeteroGraphConv({rel: self.emb_layer[rel] for src, rel, dst in self.relation_emb}, aggregate='mean')

        self.emb_dp = nn.Dropout(0.3)
        self.emb_norm = nn.LayerNorm(dim_in)
        # self._build_embed_layer()

        # share
        # self.conv_layer = GraphConv(dim_hidden, dim_hidden, weight=True, bias=True, activation=nn.LeakyReLU(0.02))
        # self.conv = HeteroGraphConv({rel: self.conv_layer for src, rel, dst in self.relation}, aggregate='mean')
        self.conv_layer = {item[1]: GraphConv(dim_hidden, dim_hidden, weight=True, bias=True, activation=nn.LeakyReLU(0.02)) for item in self.relation}
        self.conv = HeteroGraphConv({rel: self.conv_layer[rel] for src, rel, dst in self.relation}, aggregate='mean')

        self.convs_layer = [{item[1]: GraphConv(dim_hidden, dim_hidden, weight=True, bias=True, activation=nn.LeakyReLU(0.02)) for item in self.relation}
                            for i in range(layer_num + 1)]
        self.convs = nn.ModuleList([HeteroGraphConv({rel: self.convs_layer[i][rel] for
                                                    src, rel, dst in relation}, aggregate='mean') for i in range(layer_num + 1)])

        # self.conv_norm = nn.ModuleDict({'cell': nn.LayerNorm(dim_hidden), 'gene': nn.LayerNorm(dim_hidden)})
        self.conv_norm = nn.LayerNorm(dim_hidden)
        self.conv_dp = nn.Dropout(0.3)

        assert res in [True, False]
        self.res = res
        assert share in [True, False]
        self.share = share

    def forward(self, graph, feature):

        feature = {k: self.emb_norm(v) for k, v in feature.items()}
        sub_graph_emb = dgl.edge_type_subgraph(graph, self.relation_emb)
        emb = self.emb(sub_graph_emb, feature)
        h = {k: self.emb_dp(v) for k, v in emb.items()}

        # conv
        for i in range(1, self.layer_num + 1):
            # h = {k: self.conv_norm[k[-4:]](v) for k, v in emb.items()}  # TODO cell gene
            h = {k: self.conv_norm(v) for k, v in h.items()}  # TODO cell gene
            if self.share:
                h = self.conv(graph, h)
            else:
                h = self.convs[i](graph, h)
            h = {k: self.conv_dp(v.float()) for k, v in h.items()}
            if self.res:
                h = {k: v + emb[k] for k, v in h.items()}

        h_gnn = nn.Identity()(h)
        return h_gnn

    def _build_embed_layer(self, activation=None, **kwds):
        embed_params = dict(
            in_dim_dict=self.in_dim_dict,
            out_dim_dict=self.h_dims[0],
            canonical_etypes=[('cell', 'express', 'gene'),
                              ('cell', 'self_loop_cell', 'cell')],
            norm='right',
            use_weight=True,
            bias=True,  #
            activation=activation,  # None #nn.LeakyReLU(0.2)
            batchnorm_ntypes=self.batchnorm_ntypes,
            layernorm_ntypes=self.layernorm_ntypes,  # None, #
            dropout_feat=0.0,
            dropout=0.2,
            aggregate='sum',
        )
        if len(kwds) > 0:
            embed_params.update(**kwds)
        self.emb = EmbLayer(**embed_params)


class RGATEncoder(nn.Module):
    def __init__(self, relation, dim_in, dim_hidden, layer_num, res=True, share=True):
        super().__init__()
        #
        #
        self.relation = set(relation)  #
        self.head_num = 4
        self.layer_num = layer_num

        self.emb = HeteroGraphConv(
            {rel: GATConv(dim_in, int(dim_hidden / self.head_num), self.head_num, activation=nn.LeakyReLU(0.1),
                          feat_drop=0.1)
             for src, rel, dst in relation}, aggregate='mean')  # 128 / 8 attn_drop

        self.bn0 = nn.LayerNorm(dim_in)
        self.dp0 = nn.Dropout(0.2)
        self.conv = HeteroGraphConv(
            {rel: GATConv(dim_hidden, int(dim_hidden / self.head_num), self.head_num, activation=nn.LeakyReLU(0.1),
                          feat_drop=0.1) for src, rel, dst in relation}, aggregate='mean')  # 128 / 8 attn_drop
        self.convs = nn.ModuleList([HeteroGraphConv(
            {rel: GATConv(dim_hidden, int(dim_hidden / self.head_num), self.head_num, activation=nn.LeakyReLU(0.1),
                          feat_drop=0.1) for src, rel, dst in relation}, aggregate='mean') for i in range(layer_num + 1)])      # TODO

        self.bn1 = nn.LayerNorm(dim_hidden)
        self.dp1 = nn.Dropout(0.2)

        self.head = 8
        self.gat = HeteroGraphConv({rel: GATConv(dim_hidden, 16, self.head) for src, rel, dst in relation},
                                   aggregate='mean')

        assert res in [True, False]
        self.res = res
        assert share in [True, False]
        self.share = share

    def forward(self, graph, feature):
        # emb + 1~2gnn +gat
        # emb
        emb = {k: self.bn0(v) for k, v in feature.items()}
        emb = self.emb(graph, emb)  # TODO identity
        emb = {k: v.view(v.size(0), -1) for k, v in emb.items()}  # reshape
        h = {k: self.dp0(v) for k, v in emb.items()}
        temp = [emb]
        # gnn，+1。
        for i in range(1, self.layer_num + 1):
            h = {k: self.bn1(v) for k, v in h.items()}

            if self.share:
                h = self.conv(graph, h)
            else:
                h = self.convs[i](graph, h)
            h = {k: v.view(v.size(0), -1) for k, v in h.items()}  # reshape
            if self.res:
                h = {k: v + emb[k] for k, v in h.items()}  # TODO identity
            h = {k: self.dp0(v.float()) for k, v in h.items()}

        h_hidden = nn.Identity()(h)  # TODO identity
        # h_gat = self.gat(graph, h_hidden)
        # # h_gat = {k: torch.sum(v, dim=-2).squeeze() / v.size(-1) for k, v in h_gat.items()}
        # h_gat = {k: v.view(v.size(0), -1) for k, v in h_gat.items()}    # batch, hidden

        return h_hidden


class DotDecoder(nn.Module):
    def __init__(self):
        super(DotDecoder, self).__init__()
        self.device = None

    def construct_negative_edge(self, edges, node_num, k):
        # node_num: dst node num
        src, dst = edges  # src->dst
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, node_num, (len(src) * k,))
        return neg_src.to(self.device), neg_dst.to(self.device)  # TODO device

    def construct_negative_graph(self, graph, k=1):
        #
        edge_type_list = list(graph.canonical_etypes)
        data_dict = {
            edge_type: self.construct_negative_edge(graph.edges(etype=edge_type), graph.num_nodes(edge_type[-1]), k=k)
            for edge_type in edge_type_list}
        num_nodes_dict = {node_type: graph.num_nodes(node_type) for node_type in graph.ntypes}
        return dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)

    def dot_product(self, graph, feature):
        out_dict = {}
        with graph.local_scope():
            for stype, etype, dtype in graph.canonical_etypes:
                rel_graph = graph[stype, etype, dtype]  # TODO check
                rel_graph.nodes[stype].data['hs'] = feature[stype]
                rel_graph.nodes[dtype].data['hd'] = feature[dtype]
                rel_graph.apply_edges(fn.u_dot_v('hs', 'hd', 'score'))
                out_dict[(stype, etype, dtype)] = rel_graph.edata['score'].squeeze()

        return out_dict

    def forward(self, graph, feature):
        graph_positive = graph
        graph_negative = self.construct_negative_graph(graph)

        out_positive = self.dot_product(graph_positive, feature)
        out_negative = self.dot_product(graph_negative, feature)
        return out_positive, out_negative  # (dict, dict)


class GATClassifier(nn.Module):
    def __init__(self, relation, dim_hidden, dim_out):
        super(GATClassifier, self).__init__()
        self.head_num = 8
        self.relation = set(relation)
        self.dim_hidden = dim_hidden

        self.relation_class = []
        for item in relation:
            src, rel, dst = item
            if rel in {'homo', 'expressedby', 'selfloop'}:
                self.relation_class.append(item)

        # self.classifier_layer = {item[1]: GATConv(dim_hidden, dim_out, self.head_num, activation=nn.LeakyReLU(0.1), feat_drop=0.1) for item in self.relation_class}
        # self.classifier = HeteroGraphConv({rel: self.classifier_layer[rel] for src, rel, dst in self.relation_class}, aggregate='mean')
        self.classifier_layer = {item[1]: GATConv(dim_hidden, dim_out, self.head_num, activation=nn.LeakyReLU(0.1), feat_drop=0.1) for item in self.relation}
        self.classifier = HeteroGraphConv({rel: self.classifier_layer[rel] for src, rel, dst in self.relation}, aggregate='mean')

    def forward(self, graph, feature):
        # graph = dgl.edge_type_subgraph(graph, self.relation_class)
        prediction = self.classifier(graph, feature)
        prediction = {k: torch.sum(v, dim=-2).squeeze() / v.size(-1) for k, v in prediction.items()}
        return prediction

from dgl.utils import expand_as_pair
from dgl.nn.pytorch.softmax import edge_softmax
class GraphAttentionLayer(nn.Module):
    """
    Modified version of `dgl.nn.GATConv`
    * message passing with attentions.
    * directed and asymmetric message passing, allowing different dimensions
        of source and destination node-features.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 n_heads=8,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 attn_type='mul',  # or 'add' as the original paper
                 heads_fuse=None,  # 'flat' or 'mean'
                 ):
        super(GraphAttentionLayer, self).__init__()
        self._n_heads = n_heads
        self._in_src_dim, self._in_dst_dim = expand_as_pair(in_dim)
        self._out_dim = out_dim

        ### weights for linear feature transform
        if isinstance(in_dim, tuple):
            ### asymmetric case
            self.fc_src = nn.Linear(
                self._in_src_dim, out_dim * n_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_dim, out_dim * n_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_dim, out_dim * n_heads, bias=False)
        ### weights for attention computation
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        if residual:
            if self._in_dst_dim != out_dim:
                self.res_fc = nn.Linear(
                    self._in_dst_dim, n_heads * out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        self.leaky_relu = nn.LeakyReLU(negative_slope)  # for thresholding attentions
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

        self.activation = activation  # output
        self.attn_type = attn_type
        self._set_attn_fn()
        self.heads_fuse = heads_fuse
        self._set_fuse_fn()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:  # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, g, feat, return_attn=False):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor or a pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        g = g.local_var()
        ### feature linear transform
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._n_heads, self._out_dim)
            feat_dst = self.fc_dst(h_dst).view(-1, self._n_heads, self._out_dim)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._n_heads, self._out_dim)
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        if self.heads_fuse == 'mul':
            er /= np.sqrt(self._out_dim)
        g.srcdata.update({'ft': feat_src, 'el': el})
        g.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        g.apply_edges(self.attn_fn)

        e = self.leaky_relu(g.edata.pop('e'))
        # compute softmax (normalized weights)
        g.edata['a'] = self.attn_drop(edge_softmax(g, e))
        # message passing
        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        rst = g.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_dim)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)

        # handling multi-heads
        rst = self.fuse_heads(rst)
        if return_attn:
            return rst, g.edata['a']
        return rst

    def _set_attn_fn(self, ):
        if self.attn_type == 'mul':
            self.attn_fn = fn.u_mul_v('el', 'er', 'e')
        elif self.attn_type == 'add':
            # use the same attention as the GAT paper
            self.attn_fn = fn.u_add_v('el', 'er', 'e')
        else:
            raise ValueError('`attn_type` shoul be either "add" (paper) or "mul"')

    def _set_fuse_fn(self, ):
        # function handling multi-heads
        if self.heads_fuse is None:
            self.fuse_heads = lambda x: x
        elif self.heads_fuse == 'flat':
            self.fuse_heads = lambda x: x.flatten(1)  # then the dim_out is of H * D_out
        elif self.heads_fuse == 'mean':
            self.fuse_heads = lambda x: x.mean(1)  # then the dim_out is of D_out
        elif self.heads_fuse == 'max':
            self.fuse_heads = lambda x: torch.max(x, 1)[0]  # then the dim_out is of D_out


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class MLPClassifier(nn.Module):
    """

    """

    def __init__(self, dim_hidden, dim_out):
        super(MLPClassifier, self).__init__()
        #

        self.fc0 = nn.Linear(dim_hidden, dim_hidden)
        self.ac0 = nn.LeakyReLU()
        self.dp0 = nn.Dropout()
        self.fc1 = nn.Linear(dim_hidden, dim_hidden)
        self.ac1 = nn.ReLU()
        self.dp1 = nn.Dropout()
        self.out = nn.Linear(dim_hidden, dim_out)
        self.sf = nn.Softmax(dim=-1)

    # def forward(self, _, feature, alpha):
    #     # alpha
    #     alpha = alpha
    #     # reverse
    #     h = {name: ReverseLayerF.apply(data, alpha) for name, data in feature.items()}
    #     # class
    #     h = {name: self.fc0(feature) for name, feature in h.items()}
    #     h = {name: self.ac_func(feature) for name, feature in h.items()}  #
    #     h = {name: self.bn0(feature) for name, feature in h.items()}  #
    #     h = {name: self.fc1(feature) for name, feature in h.items()}
    #     # softmax
    #     prediction = {name: self.sf(data) for name, data in h.items()}
    #     return prediction

    def forward(self, feature, alpha):
        # alpha
        h = {name: self.fc0(data) for name, data in feature.items()}
        h = {name: self.ac0(data) for name, data in h.items()}
        h = {name: self.dp0(data) for name, data in h.items()}
        h = {name: self.fc1(data) for name, data in h.items()}
        h = {name: self.ac1(data) for name, data in h.items()}
        h = {name: self.dp1(data) for name, data in h.items()}
        h = {name: self.out(data) for name, data in h.items()}
        # softmax
        prediction = {name: self.sf(data) for name, data in h.items()}
        return prediction


class ClusterClassifier(nn.Module):
    """

    """
    def __init__(self, dim_hidden, dim_out):
        super(ClusterClassifier, self).__init__()
        self.fc0 = nn.Linear(dim_hidden, dim_hidden)
        self.ac0 = nn.LeakyReLU()
        self.dp0 = nn.Dropout()
        self.fc1 = nn.Linear(dim_hidden, dim_hidden)
        self.ac1 = nn.ReLU()
        self.dp1 = nn.Dropout()
        self.out = nn.Linear(dim_hidden, dim_out)
        self.sf = nn.Softmax(dim=-1)

    def forward(self, feature, alpha):
        h = self.fc0(feature)
        h = self.ac0(h)
        h = self.dp0(h)
        h = self.fc1(h)
        h = self.ac1(h)
        h = self.dp1(h)
        h = self.out(h)
        # softmax
        prediction = self.sf(h)
        return prediction

def target_distribution(q):
    # Pij
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


# class ClusterClassifier(nn.Module):
#     """
#
#     """
#     def __init__(self, dim_hidden, cluster_num_dict):
#         super(ClusterClassifier, self).__init__()
#         #
#         from torch.nn.parameter import Parameter
#         self.params_dict = nn.ParameterDict({name: Parameter(torch.Tensor(n_cluster, dim_hidden)) for name, n_cluster in cluster_num_dict.items()})
#         # self.bn_dict = nn.ModuleDict({name: nn.BatchNorm1d(dim_hidden) for name, n_cluster in cluster_num_dict.items()})
#
#     def forward(self, feature):
#         # q = {name: self.bn_dict[name](data) for name, data in feature.items() if name.endswith('cell')}
#         q = feature
#         q = {name: 1. / (1. + torch.sum(torch.pow(data.unsqueeze(1) - self.params_dict[name], 2), 2) / 1)
#              for name, data in q.items() if name.endswith('cell')}
#         q = {name: data.pow((1. + 1.) / 2.) for name, data in q.items()}
#         q = {name: (data.t() / torch.sum(data, 1)).t() for name, data in q.items()}
#         p = {name: target_distribution(data) for name, data in q.items()}
#         return q, p


class MLPDecoder(nn.Module):
    """

    """
    def __init__(self, dim_hidden, dim_out):
        super(MLPDecoder, self).__init__()
        self.fc0 = nn.Linear(dim_hidden, dim_hidden)
        self.ac0 = nn.LeakyReLU()
        self.dp0 = nn.Dropout()
        self.fc1 = nn.Linear(dim_hidden, dim_hidden)
        self.ac1 = nn.LeakyReLU()
        self.dp1 = nn.Dropout()
        self.out = nn.Linear(dim_hidden, dim_out)

    def forward(self, feature):
        h = self.fc0(feature)
        h = self.ac0(h)
        h = self.dp0(h)
        h = self.fc1(h)
        h = self.ac1(h)
        h = self.dp1(h)
        h = self.out(h)
        return h
