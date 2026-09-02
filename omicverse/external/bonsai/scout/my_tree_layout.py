import os
import csv
import numpy as np
import time
import pandas as pd
import itertools
from numpy import ndarray
from random import shuffle

import logging
FORMAT = '%(asctime)s %(funcName)s %(levelname)s %(message)s'
log_level = logging.WARNING
log_level = logging.DEBUG
logging.basicConfig(format=FORMAT,
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.WARNING)   # silence all libraries

# Create your app logger
logger = logging.getLogger("myapp")
logger.setLevel(log_level)

class Layout_Tree:
    # This is just a slim version of the scdata_tree, to make getting the equal-angle layout independent of bonsai and
    # make it as fast and lean as possible.
    root = None
    nNodes = None
    vert_ind_to_node_df = None
    vert_ind_to_node = None

    # Layout information
    coords = None
    nodeIds = None

    def __init__(self):
        self.root = Layout_TreeNode(vert_ind=-1, isRoot=True, opt_angle=360)
        self.nNodes = 1

    def copy(self, minimal_copy=False):
        new_tree = Layout_Tree()
        if not minimal_copy:
            new_tree.coords = self.coords.copy()
        new_tree.nNodes = self.nNodes
        new_tree.root = self.root.copy(minimal_copy=minimal_copy)
        for child in new_tree.root.childNodes:
            child.parentNode = new_tree.root
        return new_tree

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
                curr_node.nodeId = curr_str
                if (node_id_to_vert_ind is not None) and (curr_node.nodeId in node_id_to_vert_ind):
                    curr_node.vert_ind = node_id_to_vert_ind[curr_node.nodeId]
                nodeIdY_tParentF = False
                curr_str = ''
            elif char == ',':
                # This indicates the time is complete, new child coming
                new_node = True
            elif char == ';':
                curr_node.isRoot = True
                new_node = True
            else:
                if curr_node is None:
                    curr_node = Layout_TreeNode(isLeaf=True)
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

    def get_angles_from_coords(self):
        self.root.storeParent()
        self.root.angle = 360
        vert_ind_to_node = self.get_vert_ind_to_node_DF()
        node_list_wo_leafs = [node for vert_ind, node in vert_ind_to_node.items() if not node.isLeaf]
        for node in node_list_wo_leafs:
            # Get neighbouring coords
            coords = np.zeros((len(node.childNodes), 2)) if node.isRoot else np.zeros((len(node.childNodes) + 1, 2))
            for ind, child in enumerate(node.childNodes):
                coords[ind, :] = child.coords
            if not node.isRoot:
                coords[-1, :] = node.parentNode.coords
            # Use these coords to get angles as viewed from node
            rel_coords = coords - node.coords
            degs = np.degrees(np.arctan2(rel_coords[:, 1], rel_coords[:, 0])) % 360
            if node.isRoot:
                degs = np.append(degs, 180)
            # The angle that is stored in the treenode-object is this angle relative to the angle to the parent:
            angles = (degs[:-1] - degs[-1]) % 360

            # # Given these angles (theta), determine what part of the circle (s) is covered by each child. For this we
            # # solve the equations: s1/2 + s2/2 = theta2 - theta1
            # coeff_s = np.eye(len(degs)) / 2
            # for ind in range(len(degs) - 1):
            #     coeff_s[ind, ind + 1] = .5
            # coeff_s[-1, :] = 1
            #
            # angle_diffs = np.zeros(len(degs))
            # angle_diffs[:-1] = np.diff(degs) % 360
            # angle_diffs[-1] = 360
            # logger.debug(node.nodeId)
            # shades = np.linalg.solve(coeff_s, angle_diffs)
            for ind, child in enumerate(node.childNodes):
                child.angle = angles[ind]

    def equalAngle(self, get_nodelist=True, verbose=False):
        coords = np.zeros((self.nNodes, 2))
        _, _, _ = self.root.getDsLeafs(self.nNodes, verbose=verbose, get_nodelist=get_nodelist)
        if verbose:
            logger.debug("Obtained all downstream information.")
        self.root.coords = np.array([0., 0.])
        self.root.thetaParent = 180

        coords[self.root.vert_ind, :] = self.root.coords
        self.root.set_eq_angles(my_shade=360, use_weights=False)
        self.coords = self.root.positionChildren(coords, verbose=verbose)

    def get_dendrogram(self, verbose=False, xlims=(-.95, .95), ylims=(-.95, .95), ladderized=False, flipped_node_ids=[]):
        # get list of leafs, set y-coords
        if not ladderized:
            vert_ind_to_node = self.get_vert_ind_to_node_DF(update=True)
        else:
            if self.root.dsLeafs is None:
                self.root.getDsLeafs(get_nodelist=False)
            vert_ind_to_node = self.root.get_vert_ind_to_node_ladderized(vert_ind_to_node={}, flipped_node_ids=flipped_node_ids)
        leafs = [node for vert_ind, node in vert_ind_to_node.items() if node.isLeaf]
        n_leafs = len(leafs)
        leaf_y_coords = np.linspace(ylims[0], ylims[1], n_leafs)
        coords = np.zeros((self.nNodes, 2))
        for leaf_ind, leaf in enumerate(leafs):
            coords[leaf.vert_ind, 1] = leaf_y_coords[leaf_ind]
        coords[self.root.vert_ind, 0] = 0
        self.coords, _ = self.root.set_dendrogram_coords(coords, verbose=verbose)
        self.coords[:, 0] = self.coords[:, 0] / (np.max(self.coords[:, 0]) / (xlims[1] - xlims[0])) + xlims[0]

    def reset_root(self, new_root_ind):
        vert_ind_to_node = self.get_vert_ind_to_node_DF()
        if list(vert_ind_to_node.values())[1].parentNode is None:
            self.root.storeParent()
        self.root = vert_ind_to_node[new_root_ind]
        # self.root.isRoot = True
        # self.root.tParent = None
        self.root.reset_root_node(parent_ind=None, old_tParent=self.root.tParent)

    def get_edge_dict(self, nodeIdToVertInd=None):
        if nodeIdToVertInd is None:
            edge_dict = {'source': [], 'target': [], 'dist': []}
        else:
            edge_dict = {'source': [], 'source_ind': [], 'target': [], 'target_ind': [], 'dist': []}
        edge_dict = self.root.get_edge_dict_node(edge_dict, nodeIdToVertInd=nodeIdToVertInd)
        return edge_dict

    # def equalDaylight(self, ind, allowNegDaylight=False, verbose=False):
    #     nodeList = self.getNodeListBF()
    #     counter = 0
    #     for nodeId, node in nodeList.items():
    #         if (ind > 0) and (counter > 0):
    #             return
    #         if verbose and (counter % 1000 == 0):
    #             logger.debug("Setting equal-daylight angle for vertex %d" % counter)
    #         if not node.isLeaf:
    #             node.setequalDaylightAngles(self.coords, allowNegDaylight=allowNegDaylight)
    #             self.coords = node.positionChildren(self.coords, set_eq_angle=False)
    #         counter += 1

    def equalDaylightAll(self, verbose=False, print_dev=False, max_stepsize_changes=20, max_steps=100):
        nodeList = self.getNodeListBFVertInd()
        if print_dev:
            int_inds = np.array([vert_ind for vert_ind, node in nodeList.items() if not node.isLeaf])
            dev = np.zeros(self.nNodes)
        max_dev = 360
        step_counter = 0
        # First get for all nodes the current angles and equal-daylight angles
        stepsize = 0.10
        stepsize_changed = 0
        daylight_left = True
        new_angles = np.zeros(self.nNodes)
        new_angles[self.root.vert_ind] = 360
        reuse_calc = False
        # omicverse: old_tree is only assigned after a successful step, so a run
        # whose very first daylight pass already overshoots into negative daylight
        # hits UnboundLocalError below. Seeding it with the incoming (equal-angle)
        # tree is what the revert means on step zero.
        old_tree = self.copy()
        while daylight_left and (step_counter < max_steps) and (max_dev > 10) and (
                stepsize_changed < max_stepsize_changes) and (stepsize > 1e-3):
            max_dev = 0
            node_counter = 0
            for vert_ind, node in nodeList.items():
                if verbose and (node_counter % 1000 == 0):
                    logger.debug("Calculating equal-daylight angle for vertex %d" % node_counter)
                if not node.isLeaf:
                    # In the following function new angles are calculated (but not yet implemented) for the whole tree
                    # These new angles are based on the current position of the nodes
                    new_angles, dl_left, dev_curr = node.get_ed_angles(self.coords, new_angles, stepsize=stepsize,
                                                                       get_dev=True, reuse_calc=reuse_calc,
                                                                       set_angles=False, break_at_branch_overlap=True)
                    max_dev = dev_curr if dev_curr > max_dev else max_dev
                    if print_dev:
                        dev[vert_ind] = dev_curr
                    # We check after each node whether we already have negative daylight, in which case we revert to
                    # last tree for which each node had daylight
                    if dl_left < 0:
                        daylight_left = False
                        break
                node_counter += 1
            if daylight_left:
                if print_dev:
                    dev_int = dev[int_inds]
                    max_dev = np.max(dev_int)
                    # If daylight left, previous change can be processed. Current new_angles are tested in next round
                    logger.debug("Testing new angles equal daylight step {:d}. Stepsize {:.5f}, "
                          "total deviation {:.5f}, max deviation {:.5f}, "
                          "min deviation {:.5f}.".format(step_counter, stepsize, np.sum(dev_int),
                                                         max_dev, np.min(dev_int)))
                else:
                    logger.debug("Performed step {:d} of equal daylight. Still daylight left on all nodes.\n"
                          "Maximum deviation from equal-daylight across nodes is {:.2f}".format(step_counter, max_dev))
            # If min daylight < 0, go to last stored tree
            if not daylight_left:
                reuse_calc = True
                self = old_tree
                daylight_left = True
                stepsize_changed += 1
                stepsize /= 2
                nodeList = self.getNodeListBFVertInd()
                logger.debug("Last equal-daylight step was too large. Trying again with stepsize {}.".format(stepsize))
            else:
                step_counter += 1
                reuse_calc = False
                # Copy current tree before making changes, so that we can go back if too much daylight was removed
                old_tree = self.copy()
                for vert_ind, node in nodeList.items():
                    node.angle = new_angles[vert_ind]
                self.coords = self.root.positionChildren(self.coords, verbose=verbose)

        # Perform a last test if there is daylight on this tree
        for vert_ind, node in nodeList.items():
            if not node.isLeaf:
                # In the following function new angles are calculated (but not yet implemented) for the whole tree
                # These new angles are based on the current position of the nodes
                new_angles, dl_left, dev_curr = node.get_ed_angles(self.coords, new_angles, stepsize=stepsize,
                                                                   get_dev=True, reuse_calc=reuse_calc,
                                                                   set_angles=False, break_at_branch_overlap=True)
                # We check after each node whether we already have negative daylight, in which case we revert to
                # last tree for which each node had daylight
                if dl_left < 0:
                    daylight_left = False
                    break

        # If min daylight < 0, go to last stored tree
        if not daylight_left:
            self = old_tree
            logger.debug("Last equal-daylight step was too large. Trying again with stepsize {}.".format(stepsize))

        return self

    def equalDaylightSome(self, vert_inds=None, stepsize=1):
        if vert_inds is None:
            vert_inds = []
        vert_ind_to_node = self.get_vert_ind_to_node_DF()
        new_angles = np.zeros(self.nNodes)
        for vert_ind in vert_inds:
            node = vert_ind_to_node[vert_ind]
            node.get_ed_angles(self.coords, new_angles, stepsize=stepsize,
                               get_dev=False, reuse_calc=False,
                               set_angles=True)
            self.coords = node.positionChildren(self.coords)

    def increaseEqualAngle(self, vert_inds, multip_angle, origin):
        ancestor_ind = vert_inds[-2]
        ancestor = self.get_vert_ind_to_node_DF()[ancestor_ind]
        orig_coords_anc_rel_origin = ancestor.coords - origin
        ds_ind = vert_inds[-1]
        # Get indices of nodes in chosen subtree. Should be dsNodes that contains ds_ind
        to_be_changed = None
        for child in ancestor.childNodes:
            if ds_ind in child.dsNodes:
                to_be_changed = child.dsNodes
                break
        if to_be_changed is None:
            to_be_changed = ancestor.usNodes

        for vert_ind in to_be_changed:
            node = self.get_vert_ind_to_node_DF()[vert_ind]
            if node.isLeaf:
                node.dsLeafs_weighted *= multip_angle
        self.root.getDsLeafs(self.nNodes, verbose=False, get_nodelist=False)
        self.root.set_eq_angles(my_shade=360, use_weights=True)
        self.coords = self.root.positionChildren(self.coords)

        # Pick new origin such that ancestor node remains at same spot
        new_origin = ancestor.coords - orig_coords_anc_rel_origin
        return new_origin

    def resetEqualAngle(self):
        for vert_ind, node in self.get_vert_ind_to_node_DF().items():
            if node.isLeaf:
                node.dsLeafs_weighted = 1
        self.root.set_eq_angles(my_shade=360, use_weights=True)
        self.coords = self.root.positionChildren(self.coords)

    def getNodeListBF(self):
        nodeList = {self.root.nodeId: self.root}
        queue = [self.root]
        while len(queue):
            node = queue.pop(0)
            for child in node.childNodes:
                nodeList[child.nodeId] = child
                queue.append(child)
        return nodeList

    def getNodeListBFVertInd(self):
        nodeList = {self.root.vert_ind: self.root}
        queue = [self.root]
        while len(queue):
            node = queue.pop(0)
            for child in node.childNodes:
                nodeList[child.vert_ind] = child
                queue.append(child)
        return nodeList

    def get_vert_ind_to_node_DF(self, update=False):
        if (self.vert_ind_to_node_df is None) or update:
            vert_ind_to_node = {}
            self.vert_ind_to_node_df = self.root.get_vert_ind_to_node_node(vert_ind_to_node)
        return self.vert_ind_to_node_df


class Layout_TreeNode:
    vert_ind = None  # identifier index for the node
    nodeId = None
    tParent = None  # diffusion time to parent along tree
    childNodes = None  # list with pointers to child TreeNode objects
    parentNode = None
    isLeaf = None
    isRoot = None
    dsLeafs = None
    dsLeafs_weighted = None
    dsNodes = None
    usNodes = None
    level = None

    # Coordinate information
    coords = None
    angle = None
    opt_angle = None
    thetaParent = None

    def __init__(self, vert_ind=None, childNodes=None, parentNode=None, isLeaf=False, isRoot=False, tParent=None,
                 nodeId=None, opt_angle=None):
        self.vert_ind = vert_ind
        self.childNodes = [] if (childNodes is None) else childNodes
        self.nodeId = nodeId
        self.tParent = tParent
        self.parentNode = parentNode
        self.isLeaf = isLeaf  # isLeaf
        self.isRoot = isRoot
        self.opt_angle = opt_angle

    def copy(self, minimal_copy=False):
        new_node = Layout_TreeNode(vert_ind=self.vert_ind, nodeId=self.nodeId, tParent=self.tParent, isLeaf=self.isLeaf,
                                   isRoot=self.isRoot)
        if not minimal_copy:
            new_node.dsLeafs = self.dsLeafs
            new_node.dsLeafs_weighted = self.dsLeafs_weighted
            new_node.dsNodes = self.dsNodes.copy()
            new_node.usNodes = self.usNodes.copy()

            # Coordinate information
            new_node.coords = self.coords.copy()
            new_node.angle = self.angle
            new_node.opt_angle = self.opt_angle
            new_node.thetaParent = self.thetaParent
            new_node.vert_ind = self.vert_ind

        new_node.childNodes = [child.copy() for child in self.childNodes]
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

    def get_vert_ind_to_node_node(self, vert_ind_to_node):
        vert_ind_to_node[self.vert_ind] = self
        for child in self.childNodes:
            vert_ind_to_node = child.get_vert_ind_to_node_node(vert_ind_to_node)
        return vert_ind_to_node

    def get_vert_ind_to_node_ladderized(self, vert_ind_to_node, flipped_node_ids=[]):
        vert_ind_to_node[self.vert_ind] = self
        ds_leafs_ch = np.zeros(len(self.childNodes), dtype=int)
        for ind, child in enumerate(self.childNodes):
            ds_leafs_ch[ind] = child.dsLeafs
        if self.nodeId in flipped_node_ids:
            count = flipped_node_ids.count(self.nodeId)
            sorted_ch_inds_ladder = np.argsort(ds_leafs_ch)

            if len(sorted_ch_inds_ladder) <= 4:
                diff_orderings = list(itertools.permutations(sorted_ch_inds_ladder))
                sorted_ch_inds_ladder = list(diff_orderings[count % len(diff_orderings)])
            else:
                if count != 0:
                    shuffle(sorted_ch_inds_ladder)
            logger.info("The {} branches downstream of {} flipped {} times.".format(len(sorted_ch_inds_ladder),
                                                                                     self.nodeId, count))
        else:
            sorted_ch_inds_ladder = np.argsort(ds_leafs_ch)
        for ind in sorted_ch_inds_ladder:
            vert_ind_to_node = self.childNodes[ind].get_vert_ind_to_node_ladderized(vert_ind_to_node, flipped_node_ids=flipped_node_ids)
        return vert_ind_to_node

    def storeParent(self):
        for child in self.childNodes:
            child.parentNode = self
            if not child.isLeaf:
                child.storeParent()

    def is_vert_downstream(self, vert_ind):
        if self.vert_ind == vert_ind:
            return True
        for child in self.childNodes:
            if child.is_vert_downstream(vert_ind):
                return True
        # If we reach this point, apparently the vert_ind was not downstream
        return False

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
            self.isLeaf = True
            self.childNodes = new_children
        else:
            self.isLeaf = False
            self.childNodes = new_children
        for ind, child in enumerate(self.childNodes):
            child.reset_root_node(parent_ind=self.vert_ind, old_tParent=old_tParents[ind])

    def get_edge_dict_node(self, edge_dict, nodeIdToVertInd=None):
        for child in self.childNodes:
            if nodeIdToVertInd is None:
                edge_dict['source'].append(self.nodeId)
                edge_dict['target'].append(child.nodeId)
                edge_dict['dist'].append(child.tParent)
                # new_row = {'source': self.nodeId, 'target': child.nodeId, 'dist': child.tParent}
            else:
                edge_dict['source'].append(self.nodeId)
                edge_dict['target'].append(child.nodeId)
                edge_dict['source_ind'].append(nodeIdToVertInd[self.nodeId])
                edge_dict['target_ind'].append(nodeIdToVertInd[child.nodeId])
                edge_dict['dist'].append(child.tParent)
                # new_row = {'source': self.nodeId, 'source_ind': nodeIdToVertInd[self.nodeId], 'target': child.nodeId,
                #            'target_ind': nodeIdToVertInd[child.nodeId], 'dist': child.tParent}
            # edge_df = edge_df.append(new_row, ignore_index=True)
            if not child.isLeaf:
                edge_dict = child.get_edge_dict_node(edge_dict, nodeIdToVertInd=nodeIdToVertInd)
        return edge_dict

    def setequalDaylightAngles(self, allCoords, allowNegDaylight=False):
        # Calculate angles from this node with all other coordinates
        relCoords = allCoords - self.coords
        degs = np.degrees(np.arctan2(relCoords[:, 1], relCoords[:, 0])) % 360
        # Find the ranks of the size of the angle from all nodes to the current one
        argSortDegs = np.argsort(degs)
        ranks = np.empty_like(argSortDegs)
        ranks[argSortDegs] = np.arange(len(degs))
        # degsMinus = np.degrees(np.arctan2(-relCoords[:,1], -relCoords[:,0])) % 360
        # Loop over all children and make list of all ds-nodes grouped by child
        connectingNodes = [child.dsNodes for child in self.childNodes]
        if not self.isRoot:
            connectingNodes += [self.usNodes]
        # nShades = len(self.childNodes) if self.isRoot else (len(self.childNodes) + 1)
        nShades = len(connectingNodes)
        shades = np.zeros(nShades)
        findingDaylight = True
        while findingDaylight:
            for ind, dsNodes in enumerate(connectingNodes):
                childDegs = degs[dsNodes]
                childRanks = ranks[dsNodes]
                # Given the ranks of the nodes on this child, find where the biggest gap is. That's where you find the
                # max and min angle
                argSortRanks = np.argsort(childRanks)
                sortedRanks = childRanks[argSortRanks]
                # Get the jumps in ranks. A large jump indicates that in-between there are other branches
                # The additional - 1 is to compensate for the node itself always having rank 0
                rankJumps = np.append(np.diff(sortedRanks), sortedRanks[0] - sortedRanks[-1] - 1) % len(degs)
                maxJump = np.argmax(rankJumps)
                maxAngleInd, minAngleInd = argSortRanks[maxJump], argSortRanks[(maxJump + 1) % len(argSortRanks)]
                shades[ind] = (childDegs[maxAngleInd] - childDegs[minAngleInd]) % 360
            daylight = (360 - np.sum(shades)) / nShades
            if True:  # (daylight > 0) or allowNegDaylight:
                findingDaylight = False
            else:
                self.adjustShades(np.sum(shades))
                exit("Found negative daylight! Work on this!")
        angles = shades + daylight
        for ind, child in enumerate(self.childNodes):
            child.angle = angles[ind]
        # if not self.isRoot:
        #     self.angle = 360 - angles[-1]

    def get_ed_angles(self, allCoords, new_angles, stepsize=0.1, get_dev=False, reuse_calc=False, set_angles=False,
                      break_at_branch_overlap=False):
        if not reuse_calc:
            # Calculate angles from this node with all other coordinates
            relCoords = allCoords - self.coords
            degs = np.degrees(np.arctan2(relCoords[:, 1], relCoords[:, 0])) % 360
            # # Get only the degrees of nodes other than the current node itself
            # degs = degs[np.setdiff1d(np.arange(len(degs)), self.vert_ind)]
            # Find the ranks of the size of the angle from all nodes to the current one
            argSortDegs = np.argsort(degs)
            ranks = np.empty_like(argSortDegs)
            ranks[argSortDegs] = np.arange(len(degs))
            if ranks[self.vert_ind]:
                # In this case, some (or several) other node also has a degree of zero and they were ordered first. Fix
                # this by giving the node itself rank 0, and increasing the ranks that are below its current rank by 1
                ranks[ranks < ranks[self.vert_ind]] += 1
                ranks[self.vert_ind] = 0
            # Loop over all children and make list of all ds-nodes grouped by child
            connectingNodes = [child.dsNodes for child in self.childNodes]
            if not self.isRoot:
                connectingNodes += [self.usNodes]
            nShades = len(connectingNodes)
            shades = np.zeros(nShades)
            min_max_angle = np.zeros((nShades, 2))
            for ind, dsNodes in enumerate(connectingNodes):
                childDegs = degs[dsNodes]
                childRanks = ranks[dsNodes]
                # Given the ranks of the nodes on this child, find where the biggest gap is. That's where you find the
                # max and min angle
                argSortRanks = np.argsort(childRanks)
                sortedRanks = childRanks[argSortRanks]
                # Get the jumps in ranks. A large jump indicates that in-between there are other branches
                # The additional - 1 is to compensate for the node itself always having rank 0
                rankJumps = np.append(np.diff(sortedRanks), sortedRanks[0] - sortedRanks[-1] - 1) % len(degs)
                maxJump = np.argmax(rankJumps)
                if break_at_branch_overlap and (len(rankJumps) > 1):
                    if np.partition(rankJumps, -2)[-2] > 1:
                        if get_dev:
                            return np.zeros(shape=new_angles.shape), -360, 360
                        return np.zeros(shape=new_angles.shape), -360
                maxAngleInd, minAngleInd = argSortRanks[maxJump], argSortRanks[(maxJump + 1) % len(argSortRanks)]
                min_max_angle_curr = (childDegs[minAngleInd], childDegs[maxAngleInd])
                min_max_angle[ind, :] = min_max_angle_curr
                shades[ind] = (min_max_angle_curr[1] - min_max_angle_curr[0]) % 360
            daylight = (360 - np.sum(shades)) / nShades
            if daylight < 0:
                if get_dev:
                    return np.zeros(shape=new_angles.shape), daylight, 360
                return np.zeros(shape=new_angles.shape), daylight
            # Each child gets an equal portion of the remaining daylight, so we add half of that to their extreme angles
            min_max_angle[:, 1] += daylight / 2
            min_max_angle[:, 0] -= daylight / 2
            # Also, angles are measured with respect to the angle to the parent
            min_max_angle = (min_max_angle - self.thetaParent) % 360
            # Now we can compare that to the current angle of the child-edge to get the shade to the left and right of
            # that edge
            curr_angles = [child.angle for child in self.childNodes]
            if not self.isRoot:
                curr_angles.append(0)
            curr_angles = np.array(curr_angles)
            shade_left = (min_max_angle[:, 1] - curr_angles) % 360
            shade_right = (curr_angles - min_max_angle[:, 0]) % 360
            # By definition the angle for the parent edge from the parent edge is zero
            theta_prev = 0 if (not self.isRoot) else curr_angles[-1]
            shade_left_prev = shade_left[-1]
            for ind, child in enumerate(self.childNodes):
                child.opt_angle = (theta_prev + shade_left_prev + shade_right[ind]) % 360
                theta_prev = child.opt_angle
                shade_left_prev = shade_left[ind]
        else:
            daylight = 360
            # omicverse: this branch leaves opt_angle unset, and the loop below
            # subtracts it -> TypeError on None. With no shade to balance there
            # is nothing to correct, so the optimal angle is the current one.
            for child in self.childNodes:
                if child.opt_angle is None:
                    child.opt_angle = child.angle
        if get_dev:
            total_dev = 0
        for ind, child in enumerate(self.childNodes):
            dev = child.opt_angle - child.angle
            new_angles[child.vert_ind] = child.angle + stepsize * dev
            if set_angles:
                child.angle += stepsize * dev
            if get_dev:
                total_dev += np.abs(dev)
        if get_dev:
            return new_angles, daylight, total_dev
        return new_angles, daylight
        # if not self.isRoot:
        #     self.angle = 360 - angles[-1]

    def rearrange_branches_node(self, flipped_node_ids=[], ladderize_all=False, nNodes=None):
        if self.dsLeafs is None:
            self.getDsLeafs(nNodes=nNodes, get_nodelist=False, verbose=False)

        if self.isLeaf:
            return

        if (self.nodeId in flipped_node_ids) or ladderize_all:
            # First get ladderized order
            ds_leafs_ch = np.zeros(len(self.childNodes), dtype=int)
            for ind, child in enumerate(self.childNodes):
                ds_leafs_ch[ind] = child.dsLeafs
            sorted_ch_inds_ladder = np.argsort(ds_leafs_ch)

            # Then pick the n-th permutation given by how many times self.node_ids is in the flipped_node_ids
            count = flipped_node_ids.count(self.nodeId)
            if len(sorted_ch_inds_ladder) <= 4:
                diff_orderings = list(itertools.permutations(sorted_ch_inds_ladder))
                sorted_ch_inds_ladder = list(diff_orderings[count % len(diff_orderings)])
            else:
                if count != 0:
                    shuffle(sorted_ch_inds_ladder)
            if not ladderize_all:
                logger.info("The {} branches downstream of {} flipped {} times.".format(len(sorted_ch_inds_ladder),
                                                                                         self.nodeId, count))

            # Then arrange the children in that order
            self.childNodes = [self.childNodes[ind] for ind in sorted_ch_inds_ladder]

        # Move on to do children
        for child in self.childNodes:
            child.rearrange_branches_node(flipped_node_ids=flipped_node_ids, ladderize_all=ladderize_all)

    def getDsLeafs(self, nNodes=None, get_nodelist=True, verbose=False):
        if verbose and (self.vert_ind % 100000 == 0) and (self.vert_ind != 0):
            logger.debug("Getting downstream information at vertex number {c:d}.".format(c=self.vert_ind))
        dsNodes = [self.vert_ind] if get_nodelist else None
        if self.isLeaf:
            self.dsLeafs = 1
            if self.dsLeafs_weighted is None:
                self.dsLeafs_weighted = 1
        else:
            self.dsLeafs = 0
            self.dsLeafs_weighted = 0
            for child in self.childNodes:
                dsLeafsCh, dsLeafsChWeighted, dsNodesChild = child.getDsLeafs(nNodes=nNodes,
                                                                              verbose=verbose,
                                                                              get_nodelist=get_nodelist)
                self.dsLeafs += dsLeafsCh
                self.dsLeafs_weighted += dsLeafsChWeighted
                if get_nodelist:
                    dsNodes += dsNodesChild
        if get_nodelist:
            self.dsNodes = np.array(dsNodes)
            self.usNodes = np.setdiff1d(range(nNodes), np.array(dsNodes + [self.vert_ind]))
        return self.dsLeafs, self.dsLeafs_weighted, dsNodes

    def get_dsNodes(self):
        if self.dsNodes is not None:
            return list(self.dsNodes)
        dsNodes = [self.vert_ind]
        for child in self.childNodes:
            dsNodesCh = child.get_dsNodes()
            dsNodes += dsNodesCh
        self.dsNodes = np.array(dsNodes)
        return dsNodes

    # def getAnglesDs(self):
    #     for child in self.childNodes:
    #         test

    def set_eq_angles(self, my_shade, use_weights=False, verbose=False):
        shades = np.zeros(len(self.childNodes))
        for ind, child in enumerate(self.childNodes):
            if not use_weights:
                shades[ind] = my_shade * child.dsLeafs / self.dsLeafs
            else:
                shades[ind] = my_shade * child.dsLeafs_weighted / self.dsLeafs_weighted
            if verbose and (child.vert_ind % 100000 == 0):
                logger.debug("Calculating equal-angles for vertex number %d." % child.vert_ind)
        shades_sum = np.sum(shades)
        shade_parent = 360 - shades_sum
        theta_prev = 0
        shade_prev = shade_parent
        for ind, child in enumerate(self.childNodes):
            shade_curr = shades[ind]
            child.set_eq_angles(shade_curr, use_weights=use_weights, verbose=verbose)
            child.angle = (theta_prev + (shade_prev + shade_curr) / 2) % 360
            theta_prev = child.angle
            shade_prev = shade_curr

    def positionChildren(self, all_coords, verbose=False):
        theta_parent = self.thetaParent
        for child in self.childNodes:
            theta_child = (theta_parent + child.angle) % 360
            theta_child_rad = np.radians(theta_child)
            child.coords = self.coords + child.tParent * np.array([np.cos(theta_child_rad), np.sin(theta_child_rad)])
            all_coords[child.vert_ind, :] = child.coords
            child.thetaParent = (theta_child + 180) % 360
            all_coords = child.positionChildren(all_coords, verbose=verbose)
        return all_coords

    def set_thetaParent(self):
        theta_parent = self.thetaParent
        for child in self.childNodes:
            theta_child = (theta_parent + child.angle) % 360
            child.thetaParent = (theta_child + 180) % 360
            child.set_thetaParent()

    def set_dendrogram_coords(self, coords, verbose=False):
        if self.isLeaf:
            return coords, coords[self.vert_ind, 1]
        ch_ycoords = np.zeros(len(self.childNodes))
        my_coords = coords[self.vert_ind, :]
        for ind, child in enumerate(self.childNodes):
            coords[child.vert_ind, 0] = my_coords[0] + child.tParent
            coords, ch_ycoords[ind] = child.set_dendrogram_coords(coords, verbose=verbose)
        mean_ch_y = np.mean(ch_ycoords)
        coords[self.vert_ind, 1] = mean_ch_y
        return coords, mean_ch_y


def reconstructTreeFromEdgeVertInfo(mergerFolder):
    # Read reconstructed tree information
    edgeList = []
    distList = []
    edgeFile = os.path.join(mergerFolder, 'edgeInfo.txt')
    with open(edgeFile, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            edgeList.append((int(row[0]), int(row[1])))
            distList.append(float(row[2]))

    vertIndToNodeInd = {}
    vertIndToNodeId = {}
    vertFile = os.path.join(mergerFolder, 'vertInfo.txt')
    with open(vertFile, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        skipRow = True
        for row in reader:
            if skipRow:
                skipRow = False
                continue
            vertIndToNodeInd[int(row[0])] = int(row[1])
            vertIndToNodeId[int(row[0])] = row[2]

    tree = Layout_Tree()
    vertIndToNode = {}
    for ind, edge in enumerate(edgeList):
        if ind == 0:
            tree.root = Layout_TreeNode(vert_ind=vertIndToNodeInd[edge[0]], childNodes=[], isLeaf=False, isRoot=True,
                                        tParent=None, nodeId=vertIndToNodeId[edge[0]])
            vertIndToNode[edge[0]] = tree.root
            childNode = Layout_TreeNode(vert_ind=vertIndToNodeInd[edge[1]], childNodes=[], isLeaf=True, isRoot=False,
                                        tParent=distList[ind], nodeId=vertIndToNodeId[edge[1]])
            tree.root.childNodes.append(childNode)
            vertIndToNode[edge[1]] = childNode
            tree.nNodes = 2
            continue
        if edge[0] in vertIndToNode:
            parentVert = edge[0]
            childVert = edge[1]
        else:
            parentVert = edge[1]
            childVert = edge[0]
        childNode = Layout_TreeNode(vert_ind=vertIndToNodeInd[childVert], childNodes=[], isLeaf=True, isRoot=False,
                                    tParent=distList[ind], nodeId=vertIndToNodeId[childVert])
        parentNode = vertIndToNode[parentVert]
        parentNode.childNodes.append(childNode)
        parentNode.isLeaf = False
        vertIndToNode[childVert] = childNode
        tree.nNodes += 1

    logger.debug("\n\nReconstructed tree loaded from: \n%s \n%s" % (edgeFile, vertFile))

    return vertIndToNode, vertIndToNodeInd, edgeList, distList, tree


def getLayout(scdata_tree, eq_daylight=True, dendrogram=True, verbose=True, all_coords_dict=None,
              eq_dl_max_stepsize_changes=20, eq_dl_max_steps=100, flipped_node_ids=[]):
    if verbose:
        logger.debug("Starting reconstruction of tree.")
    ly_tree = get_ly_tree_from_scdata_tree(scdata_tree)
    if verbose:
        logger.debug("Starting equal angle algorithm.")
    start = time.time()
    ly_tree.equalAngle(verbose=verbose, get_nodelist=eq_daylight)
    if all_coords_dict is not None:
        all_coords_dict['ly_eq_angle'] = ly_tree.coords.copy()
    if verbose:
        logger.debug("Equal-angle algorithm took %f seconds." % (time.time() - start))
    if eq_daylight:
        ly_tree = ly_tree.equalDaylightAll(verbose=verbose, max_stepsize_changes=eq_dl_max_stepsize_changes,
                                 max_steps=eq_dl_max_steps)
        if all_coords_dict is not None:
            all_coords_dict['ly_eq_daylight'] = ly_tree.coords.copy()
    # if dendrogram:
    #     ly_tree.get_dendrogram(verbose=verbose)
    # if all_coords_dict is not None:
    #     all_coords_dict['ly_dendrogram'] = ly_tree.coords.copy()
    if dendrogram:
        ly_tree.get_dendrogram(ladderized=True, flipped_node_ids=flipped_node_ids)
        if all_coords_dict:
            all_coords_dict['ly_dendrogram_ladderized'] = ly_tree.coords.copy()

    # for ind in range(n_equal_daylight):
    #     start = time.time()
    #     ly_tree.equalDaylight(ind, allowNegDaylight=False, verbose=verbose)
    #     if verbose:
    #         logger.debug("Loop %d of equal-daylight took %f seconds." % (ind, time.time() - start))
    # with open(result_path, 'w') as f:
    #     f.write("label,x,y\n")
    #     for ind, vert_ind in enumerate(ly_tree.nodeIds):
    #         labelXY = "%s,%f,%f" % (ly_tree.nodeIds[vert_ind], ly_tree.coords[ind, 0], ly_tree.coords[ind, 1])
    #         f.write("%s\n" % labelXY)
    return ly_tree, all_coords_dict


if __name__ == '__main__':
    resultsFolder = '/Users/Daan/Documents/postdoc/waddington-code-github/python_waddington_code/results'
    dataset = 'bonsai_simbin_6gens_groundtruth_addedNoise_fracInformative0.02'
    mergerfile = 'mergers_zscore1.0_ellipsoidsize2.0_redoStarry_optTimes_nnnReorder_reorderedEdges'
    mergerFolder = os.path.join(resultsFolder, dataset, mergerfile)
    vertIndToNode, vertIndToNodeInd, edgeList, distList, tree = reconstructTreeFromEdgeVertInfo(mergerFolder)

    coords = tree.equalAngle()


def get_ly_tree_from_scdata_tree(scdata_tree):
    ly_tree = Layout_Tree()
    ly_tree.root = Layout_TreeNode(vert_ind=scdata_tree.root.vert_ind, childNodes=[], parentNode=None,
                                   isLeaf=scdata_tree.root.isLeaf, isRoot=True, tParent=None,
                                   nodeId=scdata_tree.root.nodeId, opt_angle=360)
    ly_tree.nNodes = scdata_tree.nNodes
    copy_from_scdata_tree(ly_tree.root, scdata_tree.root)
    # Ladderize all branches
    ly_tree.root.rearrange_branches_node(flipped_node_ids=[], ladderize_all=True, nNodes=ly_tree.nNodes)
    return ly_tree


def copy_from_scdata_tree(ly_tree_node, scdata_tree_node):
    for ind, child in enumerate(scdata_tree_node.childNodes):
        ly_child = Layout_TreeNode(vert_ind=child.vert_ind, parentNode=ly_tree_node, isLeaf=child.isLeaf,
                                   isRoot=child.isRoot, tParent=child.tParent, nodeId=child.nodeId)
        ly_tree_node.childNodes.append(ly_child)
        if not child.isLeaf:
            copy_from_scdata_tree(ly_child, child)


# TODO: Should go to tree layout file
def my_tree_layout(scData, calc=False, filepath='layout.csv', daylight_subset=None, eq_daylight=True, verbose=False,
                   return_all=False, eq_dl_max_stepsize_changes=20, eq_dl_max_steps=100):
    """
    This function assumes that a tree-object is already stored in self.
    :param calc: Boolean indicating whether new layout is calculated.
    :param filepath: Absolute path to file where layout coordinates will be stored
    :param daylight_subset: subset of nodes on which to perform equal-daylight. Will be updated later
    :param n_equal_daylight: Number of rounds of equal daylight, 1 is usually enough
    :param verbose:
    :return:
    """
    # Determine which nodes should be used for equal-daylight here. By default, use everything starting from root
    # and doing a breadth-first passage.
    # if only_branch_points:
    #     if self.branchingInds is None:
    #         self.find_branching_points()
    all_coords_dict = {} if return_all else None
    if calc:
        start = time.time()
        tree, all_coords_dict = getLayout(scData.tree, eq_daylight=eq_daylight,
                                          dendrogram=True, verbose=verbose, all_coords_dict=all_coords_dict,
                                          eq_dl_max_stepsize_changes=eq_dl_max_stepsize_changes,
                                          eq_dl_max_steps=eq_dl_max_steps)
        logger.debug("The layout algorithms took {:.2f} seconds.".format(time.time() - start))

        # Store calculated layouts in file
        coords = np.zeros((scData.nVerts, 2 * len(all_coords_dict)))
        node_ids = [''] * scData.nVerts
        for vert_ind, node in scData.tree.vert_ind_to_node.items():
            node_ids[vert_ind] = node.nodeId
        labels = []
        for type_ind, ly_type in enumerate(all_coords_dict):
            labels += [ly_type + '_x', ly_type + '_y']
            coords[:, type_ind * 2: type_ind * 2 + 2] = all_coords_dict[ly_type]
        coords_df = pd.DataFrame(columns=labels, index=node_ids, data=coords)
        coords_df.to_csv(filepath)
        if 'ly_dendrogram_ladderized' in all_coords_dict:
            ly_type = 'ly_dendrogram_ladderized'
        elif 'ly_eq_daylight' in all_coords_dict:
            ly_type = 'ly_eq_daylight'
        elif 'ly_eq_angle' in all_coords_dict:
            ly_type = 'ly_eq_angle'
        else:
            ly_type = list(all_coords_dict.keys())[0]
    # else:
    #     coords_df = pd.read_csv(filepath, index_col=0)
    #     # nodeIds = list(coords_df.index)
    #     # node_id_to_vert_ind = {node.nodeId: vert_ind for vert_ind, node in self.tree.vert_ind_to_node.items()}
    #     node_ids_ordered = [scData.tree.vert_ind_to_node[vert_ind].nodeId for vert_ind in range(self.nVerts)]
    #     coords_df.reindex(labels=node_ids_ordered)
    #     if 'ly_eq_daylight_x' in coords_df.columns:
    #         ly_type = 'ly_eq_daylight'
    #     elif 'ly_eq_angle_x' in coords_df.columns:
    #         ly_type = 'ly_eq_angle'
    #     else:
    #         ly_type = coords_df.columns[0][:-2]
    #     # vert_ind_order = [node_id_to_vert_ind[node_id] for node_id in nodeIds]
    #     # coords_df.values = coords_df.values[]
    #     # for ind, node_id in enumerate(nodeIds):
    #     #     if verbose and (ind % 100000 == 0):
    #     #         logger.debug("Loading %d-th coordinates." % ind)
    #     #     coords[ind, :] = tree_coords[nodeIdsToTreeInd[nodeId], :]
    #     # coords = np.array(list(coords))
    #     n_types = int(len(coords_df.columns) / 2)
    #     all_coords_dict = {}
    #     for type_ind in range(n_types):
    #         ly_type_curr = coords_df.columns[type_ind * 2][:-2]
    #         all_coords_dict[ly_type_curr] = coords_df.values[:, type_ind * 2: type_ind * 2 + 2]
    return all_coords_dict, ly_type
    # mst_layout = igraph.Layout(coord_list)
    # self.mstLayout = mst_layout

# # TODO: Should go to tree layout file
# def get_layout(scData, calc=False, filename='layout.csv', only_branch_points=False, myOwn=False,
#                vertIndToNodeId=None, treeFolder=None, n_equal_daylight=1, verbose=False):
#     # TODO: Clean this up
#     # if only_branch_points:
#     #     if self.branchingInds is None:
#     #         self.find_branching_points()
#     got_layout = False
#     if calc:
#         start = time.time()
#         tree = getLayout(treeFolder, outputFolder=self.result_path(), layoutFilename=filename,
#                          nEqualDaylight=n_equal_daylight, verbose=verbose)
#         logger.debug("My equal-daylight algorithm took %f seconds." % (time.time() - start))
#         got_layout = True
#         # else:
#         #     start = time.time()
#         #     # Note that we have to add one to all vertices here, because R-indexing starts at 1
#         #     edge_df_forR = self.mst.get_edge_dataframe()
#         #     edge_df_forR[['source', 'target']] = edge_df_forR[['source', 'target']] + 1
#         #     edge_df_forR.to_csv(self.result_path('edge_df.txt'), sep='\t', index=False)
#         #     vert_df_forR = self.mst.get_vertex_dataframe()
#         #     vert_df_forR.index += 1
#         #     if only_branch_points:
#         #         vert_df_forR['branch_point'] = [(ind in self.branchingInds) for ind in range(self.nVerts)]
#         #     vert_df_forR.to_csv(self.result_path('vert_df.txt'), sep=',', index=False, header=True)
#         #
#         #     Rscript_path = os.path.join(os.getcwd(), "tree_visualisation", "tree_vis.R")
#         #     command = ["Rscript", "--vanilla", Rscript_path, self.result_path(), filename, '0.05', '1']
#         #     run_output = subprocess.run(command)
#         #     run_output.check_returncode()
#         #     logger.debug("Equal-daylight algorithm from ggtree took %f seconds." % (time.time() - start))
#     if got_layout:
#         coords = np.zeros((len(vertIndToNodeId), 2))
#         nodeIdsToTreeInd = {nodeId: ind for ind, nodeId in tree.nodeIds.items()}
#         for ind, vert in enumerate(vertIndToNodeId):
#             if verbose and (ind % 100000 == 0):
#                 logger.debug("Loading %d-th coordinates." % ind)
#             nodeId = vertIndToNodeId[vert]
#             coords[ind, :] = tree.coords[nodeIdsToTreeInd[nodeId], :]
#         coord_list = list(coords)
#     elif myOwn:
#         imported_layout = pd.read_csv(self.result_path(filename))
#         coords = np.zeros((len(vertIndToNodeId), 2))
#         nodeIds = list(imported_layout['label'])
#         nodeIdsToTreeInd = {nodeId: ind for ind, nodeId in enumerate(nodeIds)}
#         tree_coords = imported_layout[['x', 'y']].values
#         for ind, vert in enumerate(vertIndToNodeId):
#             if verbose and (ind % 100000 == 0):
#                 logger.debug("Loading %d-th coordinates." % ind)
#             nodeId = vertIndToNodeId[vert]
#             coords[ind, :] = tree_coords[nodeIdsToTreeInd[nodeId], :]
#         coord_list = list(coords)
#     else:
#         imported_layout = pd.read_csv(self.result_path(filename))
#         coord_list = []
#         for vert in self.mst.vs:
#             coords = list(imported_layout[imported_layout["label"] == vert["name"]][['x', 'y']].values.flatten())
#             if len(coords) == 0:
#                 logger.debug("No vertex in R-output found with name:" + vert['name'])
#             coord_list.append(coords)
#     self.coords = np.array(coord_list)
#     # mst_layout = igraph.Layout(coord_list)
#     # self.mstLayout = mst_layout
