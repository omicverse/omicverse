import numpy as np
from pathlib import Path
import os
from ..bonsai.bonsai_helpers import set_recursion_limits


class Cluster_Tree:
    # This is just a slim version of the scdata_tree, to make getting 
    # make it as fast and lean as possible.
    root = None
    nNodes = None
    n_cell_nodes = None
    vert_ind_to_node = None

    # Layout information
    coords = None
    nodeIds = None

    def __init__(self):
        self.root = Cluster_TreeNode(vert_ind=-1, isRoot=True)
        self.nNodes = 1
        self.n_cell_nodes = 0

    def copy(self, minimal_copy=False):
        new_tree = Cluster_Tree()
        if not minimal_copy:
            new_tree.coords = self.coords.copy()
        new_tree.nNodes = self.nNodes
        new_tree.root = self.root.copy(minimal_copy=minimal_copy)
        for child in new_tree.root.childNodes:
            child.parentNode = new_tree.root
        return new_tree

    def to_newick(self, use_ids=True, results_path=None):
        nwk_str = self.root.to_newick_node(use_ids=use_ids)
        if results_path is not None:
            with open(results_path, 'w') as f:
                f.write(nwk_str)
        return nwk_str

    # Used
    def storeTreeInFolder(self, treeFolder, nwk=True):
        Path(treeFolder).mkdir(parents=True, exist_ok=True)
        # TODO: Replace this with newer file, as in bonsai.bonsai_treeHelpers
        edgeList, distList, vertInfo = self.getEdgeVertInfo(verbose=False)

        with open(os.path.join(treeFolder, 'edgeInfo.txt'), "w") as file:
            for ind, edge in enumerate(edgeList):
                file.write('%d\t%d\t%.8e\n' % (edge[0], edge[1], distList[ind]))
        with open(os.path.join(treeFolder, 'vertInfo.txt'), "w") as file:
            file.write("vertInd\tnodeInd\tvertName\n")
            for vert in vertInfo:
                file.write('%d\t%d\t%s\n' % (vert, vertInfo[vert][0], vertInfo[vert][1]))

        if nwk:
            self.to_newick(use_ids=True, results_path=os.path.join(treeFolder, 'tree_test.nwk'))

    def from_newick_string(self, nwk_str, node_id_to_vert_ind=None):
        self.from_newick(nwk_str=nwk_str, node_id_to_vert_ind=node_id_to_vert_ind)

        # Renumber vert_inds on tree such that they are in line with a depth-first search
        set_recursion_limits(int(self.nNodes))
        vertIndToNode, self.nNodes = self.root.renumber_verts(vertIndToNode={}, vert_count=0)
        self.vert_ind_to_node = vertIndToNode
        self.root.storeParent()

    def from_newick_file(self, nwk_file, node_id_to_vert_ind=None):

        with open(nwk_file, "r") as f:
            nwk_str = f.readline()

        self.from_newick(nwk_str=nwk_str, node_id_to_vert_ind=node_id_to_vert_ind)

        # Renumber vert_inds on tree such that they are in line with a depth-first search
        vertIndToNode, self.nNodes = self.root.renumber_verts(vertIndToNode={}, vert_count=0)
        self.vert_ind_to_node = vertIndToNode
        self.root.storeParent()

    def from_newick(self, nwk_str, node_id_to_vert_ind=None):
        # nodes = []
        # Here we will keep a dictionary from level down the tree, to lists of nodes that don't have an assigned parent
        level_to_nodes = {0: []}
        level = 0
        curr_str = ''
        nodeIdY_tParentF = True
        curr_node = None
        new_node = False
        self.nNodes = 0
        internal_node_ids_made_up = 0
        for char in nwk_str:
            if char == '(':
                # We go down the tree one level
                level += 1
                level_to_nodes[level] = []
                # It also means we start a new node
                new_node = True
            elif char == ')':
                # We go back up to the parent level
                level -= 1
                # It also means we start a new node
                new_node = True
            elif char == ':':
                # This indicates a time is coming
                if curr_node is None:
                    # This indicates no node-id is stored in the nwk-file for this node. Make one up
                    curr_node = Cluster_TreeNode(isLeaf=True)
                    self.nNodes += 1
                    curr_node.level = level
                    level_to_nodes[level].append(curr_node)
                    # nodeIdY_tParentF = True
                    curr_str = 'internal_{}'.format(internal_node_ids_made_up)
                    internal_node_ids_made_up += 1
                    new_node = False

                curr_node.nodeId = curr_str
                if (node_id_to_vert_ind is not None) and (curr_node.nodeId in node_id_to_vert_ind):
                    curr_node.vert_ind = node_id_to_vert_ind[curr_node.nodeId]
                nodeIdY_tParentF = False
                curr_str = ''
            elif char == ',':
                # This indicates the time is complete, new child coming
                new_node = True
            elif char == ';':
                if curr_node is None:
                    # This indicates no node-id is stored in the nwk-file for this node. Make one up
                    curr_node = Cluster_TreeNode(isLeaf=True)
                    self.nNodes += 1
                    curr_node.level = level
                    level_to_nodes[level].append(curr_node)
                    # nodeIdY_tParentF = True
                    curr_str = 'internal_{}'.format(internal_node_ids_made_up)
                    internal_node_ids_made_up += 1
                    nodeIdY_tParentF = True
                curr_node.isRoot = True
                new_node = True
            else:
                if curr_node is None:
                    curr_node = Cluster_TreeNode(isLeaf=True)
                    self.nNodes += 1
                    curr_node.level = level
                    level_to_nodes[level].append(curr_node)
                    nodeIdY_tParentF = True
                curr_str += char
                new_node = False

            if new_node:
                if curr_node is not None:
                    if nodeIdY_tParentF:
                        curr_node.nodeId = curr_str
                        if (node_id_to_vert_ind is not None) and (curr_node.nodeId in node_id_to_vert_ind):
                            curr_node.vert_ind = node_id_to_vert_ind[curr_node.nodeId]
                    else:
                        curr_node.tParent = float(curr_str)
                    curr_str = ''
                    # level_to_nodes[level].append(curr_node)
                    # Get all children and add them to this node
                    if (curr_node.level + 1) in level_to_nodes:
                        curr_node.childNodes = level_to_nodes[curr_node.level + 1]
                        if len(level_to_nodes[curr_node.level + 1]) > 0:
                            curr_node.isLeaf = False
                            level_to_nodes[curr_node.level + 1] = []
                    else:
                        curr_node.childNodes = []
                curr_node = None
        self.root = level_to_nodes[0][0]

    def set_coords_at_node(self):
        node_list = self.get_vert_ind_to_node_DF()
        for vert_ind, node in node_list.items():
            node.coords = self.coords[vert_ind, :]

    def get_vert_ind_to_node_DF(self, update=False):
        if (self.vert_ind_to_node is None) or update:
            vert_ind_to_node = {}
            self.vert_ind_to_node = self.root.get_vert_ind_to_node_node(vert_ind_to_node)
        return self.vert_ind_to_node

    def reset_root(self, new_root_ind):
        vert_ind_to_node = self.get_vert_ind_to_node_DF()
        if list(vert_ind_to_node.values())[1].parentNode is None:
            self.root.storeParent()
        self.root = vert_ind_to_node[new_root_ind]
        # self.root.isRoot = True
        # self.root.tParent = None
        self.root.reset_root_node(parent_ind=None, old_tParent=self.root.tParent)

    def set_midpoint_root(self):
        # Find longest path below each internal node (discard branches to leafs)
        # Store maximum for each node in some dictionary
        self.root.get_longest_path_ds()

        # Go to node with longest path. Find pair of nodes on path with root in-between
        vert_ind_to_node = self.get_vert_ind_to_node_DF()
        max_dist_node_tuple = (None, 0)
        for _, node in vert_ind_to_node.items():
            if node.max_summed_dist_ds > max_dist_node_tuple[1]:
                max_dist_node_tuple = (node, node.max_summed_dist_ds)
        max_dist_node, max_summed_dist = max_dist_node_tuple

        # Add new node in-between these pairs at right position
        the_parent, child_ind = max_dist_node.find_midpoint_pair(max_summed_dist)

        # Change child-nodes structure to get this new root
        the_child = the_parent.childNodes[child_ind]
        delta = max_summed_dist / 2 - the_child.max_dist_ds
        # Create new-node that is going to be in-between the_parent and the_child
        midpoint_root_node = Cluster_TreeNode(vert_ind=-1, childNodes=[the_child], parentNode=the_parent, isLeaf=False,
                                              isRoot=False, tParent=the_child.tParent - delta, nodeId="midpoint_root")
        # Delete the_child from the_parent
        the_parent.childNodes = [parent_child for curr_ind, parent_child in enumerate(the_parent.childNodes) if curr_ind != child_ind] + [midpoint_root_node]
        the_child.parentNode = midpoint_root_node
        the_child.tParent = delta

        # Renumber vert_inds on tree such that they are in line with a depth-first search
        vertIndToNode, self.nNodes = self.root.renumber_verts(vertIndToNode={}, vert_count=0)
        self.vert_ind_to_node = vertIndToNode
        self.root.storeParent()

        self.reset_root(new_root_ind=midpoint_root_node.vert_ind)

        vertIndToNode, self.nNodes = self.root.renumber_verts(vertIndToNode={}, vert_count=0)
        self.vert_ind_to_node = vertIndToNode
        self.root.storeParent()

    def getEdgeVertInfo(self, verbose=False):
        edgeList, distList, nodeIndToVertId, _, nodeIndToNode = self.compile_tree_from_scData_tree()
        vertInfo = {}
        nodeIndToVertInd = {}
        vertIndCounter = 0

        for edge in edgeList:
            for ind, nodeInd in enumerate(edge):
                if nodeInd not in nodeIndToVertInd:
                    nodeIndToVertInd[nodeInd] = vertIndCounter
                    vertInfo[vertIndCounter] = (nodeInd, nodeIndToVertId[nodeInd])
                    vertIndCounter += 1
                vertInd = nodeIndToVertInd[nodeInd]
                edge[ind] = vertInd
        return edgeList, distList, vertInfo

    # Used
    def compile_tree_from_scData_tree(self):
        edge_list = []
        dist_list = []
        if self.root.nodeId is None:
            self.root.nodeId = 'root'
        rootId = self.root.nodeId
        orig_vert_names = {self.root.vert_ind: rootId}
        intCounter = 0
        starryYN = len(self.root.childNodes) > 3
        nodeIndToNode = {}
        edge_list, dist_list, orig_vert_names, intCounter, nodeIndToNode = getEdgeDistVertNamesFromNode(self.root,
                                                                                                        edge_list,
                                                                                                        dist_list,
                                                                                                        orig_vert_names,
                                                                                                        intCounter,
                                                                                                        nodeIndToNode)
        return edge_list, dist_list, orig_vert_names, starryYN, nodeIndToNode

    def assign_cluster_id_to_all_downstream_nodes(self, start_node, cluster_index):
        """
        get all downstream nodes in depth first (DF) order, store them and for each of the downstream nodes the cluster index
        """

        # so far, every nodes knows its downstream nodes
        # maybe want to change this, due to potential memory issues?? maybe not..
        _, ds_nodes_idxs = start_node.getDsLeafs_DForder()

        for ds_node_idx in ds_nodes_idxs: # TODO check that start node is here included, but I think this is the case
            ds_node = self.vert_ind_to_node[ds_node_idx]
            if ds_node.cluster_idx == 0:
                ds_node.cluster_idx = cluster_index

    def print_all_nodes_info(self):
        for n_idx in range(len(self.vert_ind_to_node)):
            n = self.vert_ind_to_node[n_idx]
            print("vert_idx: {} - nodeId: {} - edge_length: {} - parent node idx : {} - isLeaf {} -cluster_idx: {}".format(n_idx, n.nodeId, n.tParent,
                                                                                                      n.parentNode.vert_ind if n.parentNode else None,
                                                                                                      n.isLeaf,
                                                                                                      n.cluster_idx))

    def print_all_leafs_info(self):
        for n_idx in range(len(self.vert_ind_to_node)):

            n = self.vert_ind_to_node[n_idx]
            if n.isLeaf:
                print("n_idx: {} - edge_length: {} - parent node: {} - isLeaf {} -cluster_idx: {}".format(n_idx, n.tParent,
                                                                                                      n.parentNode.vert_ind if n.parentNode else None,
                                                                                                      n.isLeaf,
                                                                                                      n.cluster_idx))

    # def get_downstream_leafs(self, root_node):
    def cut(self, subtree_root, cell_ids=None):

        # from subtree_root, traverse whole subtree and return leafs
        # and set all traversed nodes to deleted
        _, ds_nodes_idxs = subtree_root.getDsLeafs_DForder()

        downstream_leafs = []
        for ds_node_idx in ds_nodes_idxs:  # TODO check that start node is here included, but I think this is the case
            ds_node = self.vert_ind_to_node[ds_node_idx]

            if cell_ids is None:
                if ds_node.isLeaf and not ds_node.is_deleted:
                    # downstream_leafs.append(ds_node)
                    downstream_leafs.append(ds_node.nodeId)
            else:
                if (ds_node.nodeId in cell_ids) and not ds_node.is_deleted:
                    # downstream_leafs.append(ds_node)
                    downstream_leafs.append(ds_node.nodeId)
            ds_node.is_deleted = True

        return downstream_leafs

    def get_min_pdists_info(self):
        self.n_leafs, self.root.ds_dists, self.n_cell_nodes = self.root.get_ds_and_parent_info_plus_dists()
        self.root.us_dists = 0
        self.root.store_us_dists(total_leafs=self.n_leafs)
        return self.n_leafs

    def get_ent_changes(self):
        if self.vert_ind_to_node is None:
            self.vert_ind_to_node, self.nNodes = self.root.renumber_verts(vertIndToNode={}, vert_count=0)

        clade_ent_root = self.root.clade_ent_contrib_ds
        annot_ent_root = self.root.annot_ent_contrib_ds
        for vert_ind, node in self.vert_ind_to_node.items():
            if node.include_as_cand:
                node.clade_ent_change = clade_ent_root - node.clade_ent_contrib_ds - node.clade_ent_contrib_us
                node.annot_ent_change = annot_ent_root - node.annot_ent_contrib_ds - node.annot_ent_contrib_us


class Cluster_TreeNode:
    # TODO remove all the stuff that is not used...
    vert_ind = None  # identifier index for the node
    nodeId = None
    tParent = None  # diffusion time to parent along tree
    childNodes = None  # list with pointers to child TreeNode objects
    parentNode = None
    isLeaf = None
    isRoot = None
    ds_leafs = None

    # Variables necessary for the annotation-based clustering
    own_annot_counts = None
    ds_annot_counts = None
    own_n_annots = None
    ds_n_annots = None
    clade_ent_contrib_ds = None
    annot_ent_contrib_ds = None
    clade_ent_contrib_us = None
    annot_ent_contrib_us = None
    clade_ent_change = None
    annot_ent_change = None
    old_parent_node = None
    belongs_to_tree = None
    revert_clade_ent_change = None
    revert_annot_ent_change = None
    include_as_cand = True

    # Variables necessary for the min-dist clustering
    ds_leafs_weighted = None # not used so far
    dsNodes = None # not used
    usNodes = None # not used
    level = None # not used
    max_dist_ds = None
    max_summed_dist_ds = None

    # Clustering information
    cluster_idx = 0 # per default all cells correspond to the same cluster at the beginning
    edge_length_rank = None # is this used?
    len_to_most_distant_leaf = 0 # For a cut set C of the tree, we define B(C,u) = the length of the path from u to the most distance connected leaf in U. U is the tree rooted at u
    edge_to_parent = True # set to false when we cut the tree
    is_deleted = False # check if it has already been processed

    # Coordinate information
    coords = None
    angle = None
    opt_angle = None
    thetaParent = None

    def __init__(self, vert_ind=None, childNodes=None, parentNode=None, isLeaf=False, isRoot=False, tParent=None,
                 nodeId=None):
        self.vert_ind = vert_ind
        self.childNodes = [] if (childNodes is None) else childNodes
        self.nodeId = nodeId
        self.tParent = tParent
        self.parentNode = parentNode
        self.isLeaf = isLeaf  # isLeaf
        self.isRoot = isRoot
        self.n_cells = None
        # self.opt_angle = opt_angle

    def to_newick_node(self, use_ids=True):
        nwk_children = []
        for child in self.childNodes:
            nwk_children.append(child.to_newick_node())
        if len(nwk_children) > 0:
            nwk = ','.join(nwk_children)
            nwk = '(' + nwk + ')'
        else:
            nwk = ''
        if self.isRoot:
            own_nwk = self.nodeId if use_ids else 'N' + str(self.nodeInd)
        else:
            own_nwk = self.nodeId + ':' + str(self.tParent) if use_ids else 'N' + str(self.nodeInd) + ':' + str(
                self.tParent)
        nwk = nwk + own_nwk
        if self.isRoot:
            return nwk + ';'
        else:
            return nwk

    def copy(self, minimal_copy=False):
        new_node = Cluster_TreeNode(vert_ind=self.vert_ind, nodeId=self.nodeId, tParent=self.tParent, isLeaf=self.isLeaf,
                                   isRoot=self.isRoot)
        if not minimal_copy:
            new_node.ds_leafs = self.ds_leafs
            new_node.ds_leafs_weighted = self.ds_leafs_weighted
            new_node.dsNodes = self.dsNodes.copy()
            new_node.usNodes = self.usNodes.copy()

            # Coordinate information
            new_node.coords = self.coords.copy()
            new_node.angle = self.angle
            new_node.opt_angle = self.opt_angle
            new_node.thetaParent = self.thetaParent
            new_node.vert_ind = self.vert_ind

        new_node.childNodes = [child.copy(minimal_copy=minimal_copy) for child in self.childNodes]
        for child in new_node.childNodes:
            child.parentNode = new_node
        return new_node

    def renumber_verts(self, vertIndToNode, vert_count):
        self.vert_ind = vert_count
        vertIndToNode[self.vert_ind] = self
        vert_count += 1
        for child in self.childNodes:
            vertIndToNode, vert_count = child.renumber_verts(vertIndToNode, vert_count)
        return vertIndToNode, vert_count

    def add_info_to_nodes(self, node_id_to_info=None, info_key=None, const_val=None):
        if node_id_to_info is not None:
            if self.nodeId in node_id_to_info:
                setattr(self, info_key, node_id_to_info[self.nodeId])
            else:
                setattr(self, info_key, None)
        elif const_val is not None:
            setattr(self, info_key, const_val)
        else:
            print("WARNING: Don't know what information to add in 'add_info_to_nodes()'")
            return
        for child in self.childNodes:
            child.add_info_to_nodes(node_id_to_info=node_id_to_info, info_key=info_key, const_val=const_val)

    def getDsLeafs_DForder(self):
        """
        TODO: store only at the node where this is invoked
        get downstream leafs of the current node in depth first (DF) order
        """

        dsNodes = [self.vert_ind]

        if self.isLeaf:
            self.ds_leafs = 1
        else:
            self.ds_leafs = 0
            for child in self.childNodes:
                ds_leafsCh, dsNodesCh = child.getDsLeafs_DForder()
                # ds_leafsCh = child.getDsLeafs_DForder()

                self.ds_leafs += ds_leafsCh
                # dsNodes += dsNodesChild
                # dsNodes += child.dsNodes


                dsNodes += dsNodesCh
        # self.dsNodes = dsNodes

        return self.ds_leafs, dsNodes

    def getDsLeafs_and_UsLeafs_DForder(self, nNodes):
        """
        get down stream nodes list and upstreames nodes list
        """
        dsNodes = [self.vert_ind]

        if self.isLeaf:
            self.ds_leafs = 1
        else:
            self.ds_leafs = 0
            for child in self.childNodes:
                ds_leafsCh, dsNodesCh, _ = child.getDsLeafs_and_UsLeafs_DForder(nNodes)
                # ds_leafsCh = child.getDsLeafs_DForder()

                self.ds_leafs += ds_leafsCh
                # dsNodes += dsNodesChild
                # dsNodes += child.dsNodes


                dsNodes += dsNodesCh

        usNodes = np.setdiff1d(range(nNodes), np.array(dsNodes + [self.vert_ind]))
        # self.dsNodes = np.array(dsNodes)
        # self.usNodes = np.setdiff1d(range(nNodes), np.array(dsNodes + [self.vert_ind]))

        # self.dsNodes = dsNodes

        return self.ds_leafs, dsNodes, usNodes

    def getPostOrder(self):
        """
        get list of nodes in post order (left, right, root) #depth first tree
        In the postorder traversal, first, we traverse the left child or left subtree of the current node and
        then we traverse the right child or right subtree of the current node. At last, we traverse the current node.
        """
        post_order_nodes = []

        if self.isLeaf:
            # print("my vert ind: {}".format(self.vert_ind))
            post_order_nodes.append(self.vert_ind)
            return post_order_nodes

        # define the left child to be the one with the lower vert ind
        left_child_index = np.argmin([self.childNodes[0].vert_ind, self.childNodes[1].vert_ind])
        left_child = self.childNodes[left_child_index]
        right_child = self.childNodes[set([0,1]).difference(set([left_child_index])).pop()]


        verts_left = left_child.getPostOrder()
        verts_right = right_child.getPostOrder()

        post_order_nodes = post_order_nodes + verts_left + verts_right + [self.vert_ind]

        return  post_order_nodes

    def getPostOrder_v2(self):
        """
        get list of nodes in post order (left, right, root) #depth first tree
        In the postorder traversal, first, we traverse the left child or left subtree of the current node and
        then we traverse the right child or right subtree of the current node. At last, we traverse the current node.

        I think this is more robust!! i.e. looks at all the children if there are more than two... (should not, but who know)
        """
        post_order_nodes = []

        if self.isLeaf:
            # print("my vert ind: {}".format(self.vert_ind))
            post_order_nodes.append(self.vert_ind)
            return post_order_nodes

        # define the left child to be the one with the lower vert ind
        # BUT i have see that the "root" has 3 children I think in tamaras tree...
        # Then the below definition does not work...
        for child_node in self.childNodes:
            grandchildren = child_node.getPostOrder_v2()
            post_order_nodes = post_order_nodes + grandchildren

        post_order_nodes = post_order_nodes + [self.vert_ind]

        return  post_order_nodes

    def getPostOrder_only_internalNodes(self):
        """
        get list of nodes in post order (left, right, root) #depth first tree
        In the postorder traversal, first, we traverse the left child or left subtree of the current node and
        then we traverse the right child or right subtree of the current node. At last, we traverse the current node.

        I think this is more robust!! i.e. looks at all the children if there are more than two... (should not, but who know)
        """
        post_order_nodes = []

        if self.isLeaf:
            return post_order_nodes

        # define the left child to be the one with the lower vert ind
        # BUT i have see that the "root" has 3 children I think in tamaras tree...
        # Then the below definition does not work...
        for child_node in self.childNodes:
            grandchildren = child_node.getPostOrder_only_internalNodes()
            post_order_nodes = post_order_nodes + grandchildren

        post_order_nodes = post_order_nodes + [self.vert_ind]

        return  post_order_nodes

    def find_longest_path_between_two_leafs(self):

        if self.isLeaf:
            return 0, 0

        long_dists = []
        short_dists = []
        for child in self.childNodes:
            long_to_ch, short_to_ch = child.find_longest_path_between_two_leafs()
            long_dists.append(long_to_ch + child.tParent)
            short_dists.append(short_to_ch + child.tParent)

        return np.max(long_dists), np.min(short_dists)

    def get_longest_path_ds(self, ignore_leaf_edge=True):
        if self.isLeaf:
            self.max_dist_ds = 0
            self.max_summed_dist_ds = 0
            return

        max_dists = []
        for child in self.childNodes:
            child.get_longest_path_ds()
            if child.isLeaf and ignore_leaf_edge:
                max_dists.append(0)
            else:
                max_dists.append(child.tParent + child.max_dist_ds)

        max_dists = np.array(max_dists)
        node_idxs_partition = np.argpartition(a=-max_dists, kth=1)  # all entries left of the kth element are smaller than the kth element, but not sorted
        nodes_with_top_edgelength = node_idxs_partition[:2]
        top_edge_lengths = max_dists[nodes_with_top_edgelength]
        self.max_summed_dist_ds = np.sum(top_edge_lengths)
        self.max_dist_ds = np.max(top_edge_lengths)

    def find_midpoint_pair(self, max_summed_dist):
        # First find child with maximum downstream path
        for child_ind, child in enumerate(self.childNodes):
            if abs(child.max_dist_ds + child.tParent - self.max_dist_ds) < 1e-9:
                break
        # Check if midpoint is between this node and child
        if child.max_dist_ds < max_summed_dist / 2:
            # If so, return this node and the index of the child
            return self, child_ind
        else:
            # Go down to child and repeat
            max_dist_node, child_ind = child.find_midpoint_pair(max_summed_dist)
            return max_dist_node, child_ind

    def get_vert_ind_to_node_node(self, vert_ind_to_node):
        vert_ind_to_node[self.vert_ind] = self
        for child in self.childNodes:
            vert_ind_to_node = child.get_vert_ind_to_node_node(vert_ind_to_node)
        return vert_ind_to_node

    def storeParent(self):
        for child in self.childNodes:
            child.parentNode = self
            if not child.isLeaf:
                child.storeParent()

    def reset_root_node(self, parent_ind=None, old_tParent=None):
        # Get all connecting nodes
        new_children = []
        new_parent = None
        old_tParents = []
        for node in self.childNodes:
            if node.vert_ind == parent_ind:
                new_parent = node
            else:
                new_children.append(node)
                old_tParents.append(node.tParent)
        if self.parentNode is not None:
            if self.parentNode.vert_ind == parent_ind:
                new_parent = self.parentNode
            else:
                old_tParents.append(self.parentNode.tParent)
                self.parentNode.tParent = old_tParent
                new_children.append(self.parentNode)
        if new_parent is not None:
            self.parentNode = new_parent
        else:
            self.parentNode = None
            self.tParent = None
            self.isRoot = True
        if len(new_children) == 0:
            # self.isLeaf = True
            self.childNodes = []
        else:
            self.isLeaf = False
            self.childNodes = new_children
        for ind, child in enumerate(self.childNodes):
            child.reset_root_node(parent_ind=self.vert_ind, old_tParent=old_tParents[ind])

    def get_ds_and_parent_info(self):
        if self.isLeaf:
            self.ds_leafs = 1
        else:
            self.ds_leafs = 0

        for child in self.childNodes:
            child.parentNode = self
            ds_leafs_ch = child.get_ds_and_parent_info()
            self.ds_leafs += ds_leafs_ch
        return self.ds_leafs

    def get_ds_and_parent_info_plus_dists(self):
        self.ds_dists = 0
        if self.n_cells is not None:
            self.ds_leafs = self.n_cells
            n_cell_nodes = min(self.n_cells, 1)
        elif self.isLeaf:
            self.ds_leafs = 1
            n_cell_nodes = 1
        else:
            self.ds_leafs = 0
            n_cell_nodes = 0

        for child in self.childNodes:
            child.parentNode = self
            ds_leafs_ch, ds_dists_ch, ds_n_cell_nodes = child.get_ds_and_parent_info_plus_dists()
            self.ds_dists += ds_dists_ch + child.tParent * child.ds_leafs
            self.ds_leafs += ds_leafs_ch
            n_cell_nodes += ds_n_cell_nodes
        return self.ds_leafs, self.ds_dists, n_cell_nodes

    def store_us_dists(self, total_leafs):
        # We first calculate the distance from each leaf to the current node from all sides, i.e., upstream and
        # downstream
        total_dists = self.us_dists + self.ds_dists
        for child in self.childNodes:
            # For each child, we take the total dists and subtract the contribution of the leaf-paths downstream of that
            # child
            total_dists_excl = total_dists - child.ds_dists
            # We then change the contribution of the edge from child to current node. This was traveled child.ds_leafs
            # times, but now is traveled total_leafs - child.ds_leafs
            child.us_dists = total_dists_excl - child.tParent * (child.ds_leafs - (total_leafs - child.ds_leafs))
            child.store_us_dists(total_leafs)

    def get_ds_annot_info(self):
        self.ds_annot_counts = self.own_annot_counts.copy()
        self.ds_n_annots = self.own_n_annots

        for child in self.childNodes:
            child.get_ds_annot_info()
            self.ds_annot_counts += child.ds_annot_counts
            self.ds_n_annots += child.ds_n_annots

    def set_include_as_cand(self, total_n_annots, total_annot_counts, clst_size_lb, clst_cat_capture_lbs,
                            clst_cat_capture_ubs):
        smallest_clst = min(total_n_annots - self.ds_n_annots, self.ds_n_annots)
        smallest_clst_counts = np.minimum(total_annot_counts - self.ds_annot_counts, self.ds_annot_counts)
        if smallest_clst < clst_size_lb:
            self.include_as_cand = False
        # elif np.all(smallest_clst_counts < clst_cat_capture_lbs):
        #     self.include_as_cand = False
        if np.any(smallest_clst_counts > clst_cat_capture_ubs):
            self.include_as_cand = True

        for child in self.childNodes:
            child.set_include_as_cand(total_n_annots, total_annot_counts, clst_size_lb, clst_cat_capture_lbs,
                                      clst_cat_capture_ubs)

    def subtract_or_add_ds_info_us(self, ds_node, add=False):
        if not add:
            self.ds_annot_counts -= ds_node.ds_annot_counts
            self.ds_n_annots -= ds_node.ds_n_annots
        else:
            self.ds_annot_counts += ds_node.ds_annot_counts
            self.ds_n_annots += ds_node.ds_n_annots
        if self.parentNode is not None:
            self.parentNode.subtract_or_add_ds_info_us(ds_node, add=add)

    def get_ent_contribs(self, total_n, total_annot_counts, num_annots=None):
        if self.include_as_cand or (self.parentNode is None):
            if num_annots is None:
                num_annots = len(total_annot_counts)
            # Store the contribution to the clade- and annotation-entropy given by the subtree downstream
            # This has the form n_c log n_c (for the clade ent.), where n_c is the total number of labeled cells downstream
            # or sum_r n_r log n_r (for the annot ent.) where n_r is the number of labeled cells of annotation r
            self.clade_ent_contrib_ds = - entropy_non_norm(self.ds_n_annots, num_annots=1)
            self.clade_ent_contrib_us = - entropy_non_norm(total_n - self.ds_n_annots, num_annots=1)
            self.annot_ent_contrib_ds = - entropy_non_norm(self.ds_annot_counts, num_annots=num_annots)
            self.annot_ent_contrib_us = - entropy_non_norm(total_annot_counts - self.ds_annot_counts, num_annots=num_annots)

        for child in self.childNodes:
            child.get_ent_contribs(total_n=total_n, total_annot_counts=total_annot_counts, num_annots=num_annots)

    def __repr__(self):
        return "vert_ind: {}\nnodeId: {}\ncluster idx: {}\nedge length: {}\nlen_to_most_distant_leaf: {}\nis_deleted: {}".format(self.vert_ind, self.nodeId, self.cluster_idx, self.tParent, self.len_to_most_distant_leaf, self.is_deleted)


def entropy_non_norm(n, num_annots=None):
    """
    Compute the Shannon entropy: - ∑_r n_r log(n_r) for a vector.
    *NOTE THAT THIS FUNCTION DOES NOT NORMALIZE THE VALUES TO SUM TO 1*
    :param n: int/float or numpy.ndarray of int/float
        A single non-negative integer or an array of non-negative integers
        representing counts n_r.
    :param num_annots: int
        Should indicate the array-length of n
    """
    if num_annots is None:
        num_annots = len(n)
    if num_annots == 1:
        return - n * np.log(n) if n > 0 else 0.0

    tmp = np.zeros(num_annots)
    np.log(n, out=tmp, where=n > 0)
    return - np.sum(n * tmp)


def entropy(n, num_annots=None):
    """
    Calculates the Shannon entropy for a vector n. *Does NOT normalize when n doesn't sum up to 1*.

    :param n: float or numpy.ndarray of float
        A single non-negative integer or an array of non-negative integers
        representing counts n_r.
    :param num_annots: int
        Should indicate the array-length of n
    """

def getEdgeDistVertNamesFromNode(node, edge_list, dist_list, orig_vert_names, intCounter, nodeIndToNode):
    nodeIndToNode[node.vert_ind] = node
    for child in node.childNodes:
        edge_list.append([node.vert_ind, child.vert_ind])
        dist_list.append(child.tParent)
        if (child.nodeId is None) or child.nodeId[:9] == 'internal_':
            orig_vert_names[child.vert_ind] = "internal_%d" % intCounter
            child.nodeId = "internal_%d" % intCounter
            intCounter += 1
        else:
            orig_vert_names[child.vert_ind] = child.nodeId
        edge_list, dist_list, orig_vert_names, intCounter, nodeIndToNode = getEdgeDistVertNamesFromNode(child,
                                                                                                        edge_list,
                                                                                                        dist_list,
                                                                                                        orig_vert_names,
                                                                                                        intCounter,
                                                                                                        nodeIndToNode)
    return edge_list, dist_list, orig_vert_names, intCounter, nodeIndToNode
