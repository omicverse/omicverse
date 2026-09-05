from typing import List, Tuple, Optional
import numpy as np
import scanpy as sc
import pandas as pd
import os

def construct_gem_expressions(adata: sc.AnnData,
                            gem_expr_key: str = 'X_gem',
                            scale_gem_expr: bool = True,
                            layer_key: Optional[str] = None):
    
    gem_expressions = adata.obsm[gem_expr_key]

    # Scale so that the GEM memberships sum to 1 per cell
    gem_sum = gem_expressions.sum(axis=0)
    gem_expressions = gem_expressions / gem_sum
    
    num_gems = gem_expressions.shape[1]
    flow_gems = ['GEM-' + str(i + 1) for i in range(num_gems)]

    adata_gem = sc.AnnData(X=gem_expressions)
    adata_gem.var.index = pd.Index(flow_gems)
    adata_gem.var['downstream_tfs'] = '' # For housekeeping for later
    adata_gem.var['type'] = 'module' # Define variable types
    adata_gem.var['interactions'] = '' # For housekeeping for later

    if scale_gem_expr:

        if layer_key is not None:

            scale_factor = adata.layers[layer_key].copy().sum(1).mean()

        else:
            scale_factor = np.expm1(adata.X).sum(1).mean()

        adata_gem.X *= scale_factor
        sc.pp.log1p(adata_gem)

    return adata_gem, flow_gems

def construct_inflow_signals_cellchat(adata: sc.AnnData,
                                    cellchat_output_key: str, 
                                    model_organism: str = 'human'):
    
    model_organisms = ['human', 'mouse']

    if model_organism not in model_organisms:
        raise ValueError ("Invalid model organism. Please select one of: %s" % model_organisms)
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_path = os.path.join(data_dir, 'cellchat_interactions_and_tfs_' + model_organism + '.csv')

    cellchat_interactions_and_tfs = pd.read_csv(data_path, index_col=0)

    ccc_output_merged = pd.concat([adata.uns[cellchat_output_key][sample] for sample in adata.uns[cellchat_output_key]])
    ccc_interactions = ccc_output_merged['interaction_name_2'].unique().tolist()

    unique_inflow_vars_and_interactions = {}

    # The inflow variables are constructed from receptor gene expression
    # that will be weighted by the average expression of their downstream
    # TF targets. These can be multi-units, but because of the weighting,
    # we hestitate to call them receptors in the context of cellular flows.
    for i, interaction in enumerate(ccc_interactions):
        receptor = interaction.split(' - ')[1].strip('()')
        receptor_split = receptor.split('+')
        receptors = []
        
        for i, rec in enumerate(receptor_split):
            if rec not in adata.var_names:
                receptor_v2_split = ccc_output_merged[ccc_output_merged['interaction_name_2'] == interaction]['receptor'].unique()[0].split('_')
                
                receptors.append(receptor_v2_split[i])
            else:
                receptors.append(rec)
        
        receptor = '+'.join(receptors)

        if receptor not in unique_inflow_vars_and_interactions:
            unique_inflow_vars_and_interactions[receptor] = [interaction]
            
        else:
            interactions_for_receptor = unique_inflow_vars_and_interactions[receptor]
            interactions_for_receptor.append(interaction)
            unique_inflow_vars_and_interactions[receptor] = interactions_for_receptor
            
            
    inflow_vars = sorted(list(unique_inflow_vars_and_interactions.keys()))
    num_inflow_vars = len(inflow_vars)

    inflow_expressions = np.zeros((adata.n_obs, num_inflow_vars)) 

    for i, receptor in enumerate(inflow_vars):
        
        receptor_expression = np.ones((adata.n_obs, ))
        
        split_receptor = receptor.split('+')
        considered_receptors = []
        
        for unit in split_receptor:
            
            if unit not in adata.var_names:
                print(unit)
                
            else:
                
                unit_expression = adata[:, unit].X.toarray().flatten()

                receptor_expression *= unit_expression
                considered_receptors.append(unit)
                
        if len(considered_receptors) != 0:
            inflow_expressions[:, i] = receptor_expression**(1.0 / len(considered_receptors))
        
    inflow_expressions_adjusted = inflow_expressions.copy()

    inflow_interactions = []
    unique_inflow_vars_and_tfs = {}

    for i, receptor in enumerate(inflow_vars):
        
        interactions_for_receptor = unique_inflow_vars_and_interactions[receptor]
        
        relevant_interactions = cellchat_interactions_and_tfs[cellchat_interactions_and_tfs['interaction_name_2'].isin(interactions_for_receptor)]
        
        joined_interactions = '/'.join(sorted(interactions_for_receptor))
        inflow_interactions.append(joined_interactions)
        
        downstream_tfs = []
        
        # Go through each category
        possible_downstream_tfs = relevant_interactions['Receptor-TF-combined'].dropna().tolist()\
                                    +  relevant_interactions['Ligand-TF-combined'].dropna().tolist()
        
        for unit in possible_downstream_tfs:
            split_unit = unit.split('_')
            for tf in split_unit:
                if tf not in downstream_tfs:
                    downstream_tfs.append(tf)
                    
        downstream_tfs = np.intersect1d(downstream_tfs, adata.var_names)
        
        unique_inflow_vars_and_tfs[receptor] = sorted(list(downstream_tfs))
        
        if len(downstream_tfs) != 0:
            
            average_tf_expression = np.zeros((adata.n_obs, ))
            
            for tf in downstream_tfs:
                
                average_tf_expression += adata[:, tf].X.toarray().flatten()
                
            average_tf_expression /= len(downstream_tfs)
            
            inflow_expressions_adjusted[:, i] *= average_tf_expression
            
    inflow_downstream_tfs = []
    for i, inflow_var in enumerate(inflow_vars):
        
        interactions_for_receptor = unique_inflow_vars_and_interactions[inflow_var]
        downstream_tfs = unique_inflow_vars_and_tfs[inflow_var]
        inflow_downstream_tfs.append('_'.join(sorted(downstream_tfs)))
        
    adata_inflow = sc.AnnData(X=inflow_expressions_adjusted)
    adata_inflow.var.index = pd.Index(inflow_vars)
    adata_inflow.var['downstream_tfs'] = inflow_downstream_tfs
    adata_inflow.var['type'] = 'inflow' # Define variable types
    adata_inflow.var['interactions'] = inflow_interactions

    return adata_inflow, inflow_vars

def construct_outflow_signals_cellchat(adata: sc.AnnData,
                                    cellchat_output_key: str, 
                                    ):
    
    cellchat_output_merged = pd.concat([adata.uns[cellchat_output_key][sample] for sample in adata.uns[cellchat_output_key]])
    cellchat_interactions = cellchat_output_merged['interaction_name_2'].unique().tolist()
    outflow_vars = []
    relevant_interactions = {}

    for inter in cellchat_interactions:

        ligand = inter.split(' - ')[0]

        if ligand not in outflow_vars:

            add_ligand = False

            # Sometimes this ligand not is not the actual symbol name (CellChat errors)
            if ligand not in adata.var_names:
                # Get the alternative name for the ligand
                ligand = cellchat_output_merged[cellchat_output_merged['interaction_name_2'] == inter]['ligand'].values[0]

                if (ligand in adata.var_names)&(ligand not in outflow_vars):
                    add_ligand = True
                    

            else:
                if ligand not in outflow_vars:
                    add_ligand = True

            if add_ligand:
                outflow_vars.append(ligand)

                relevant_interactions[ligand] = [inter]

        else:
            interactions_with_ligand = relevant_interactions[ligand]
            interactions_with_ligand.append(inter)
            relevant_interactions[ligand] = interactions_with_ligand

    outflow_expressions = np.zeros((adata.n_obs, len(outflow_vars)))

    for i, signal in enumerate(outflow_vars):
        outflow_expressions[:, i] = adata[:, signal].X.toarray().flatten()

    adata_outflow = sc.AnnData(X=outflow_expressions)
    adata_outflow.var.index = pd.Index(outflow_vars)
    adata_outflow.var['downstream_tfs'] = '' # For housekeeping for later
    adata_outflow.var['type'] = 'outflow' # Define variable types

    relevant_interactions_of_ligands = []
    for ligand in outflow_vars:
        interactions_of_ligand = relevant_interactions[ligand]
        relevant_interactions_of_ligands.append('/'.join(interactions_of_ligand))
        
    adata_outflow.var['interactions'] = relevant_interactions_of_ligands # Define variable types

    return adata_outflow, outflow_vars

def construct_flows_from_cellchat(adata: sc.AnnData,
                                cellchat_output_key: str,
                                gem_expr_key: str = 'X_gem',
                                scale_gem_expr: bool = True,
                                model_organism: str= 'mouse',
                                flowsig_network_key: str = 'flowsig_network',
                                flowsig_expr_key: str = 'X_flow'):


    # Define the expression
    adata_outflow, outflow_vars = construct_outflow_signals_cellchat(adata, cellchat_output_key)

    adata_inflow, inflow_vars = construct_inflow_signals_cellchat(adata, cellchat_output_key, model_organism)

    adata_gem, flow_gem_vars = construct_gem_expressions(adata, gem_expr_key, scale_gem_expr)

    # Determine the flow_variables
    flow_variables = outflow_vars + inflow_vars + flow_gem_vars

    flow_expressions = np.zeros((adata.n_obs, len(flow_variables)))

    for i, outflow_var in enumerate(outflow_vars):
        flow_expressions[:, i] = adata_outflow[:, outflow_var].X.toarray().flatten()

    for i, inflow_var in enumerate(inflow_vars):
        flow_expressions[:, len(outflow_vars) + i] = adata_inflow[:, inflow_var].X.toarray().flatten()

    for i, gem in enumerate(flow_gem_vars):
        flow_expressions[:, len(outflow_vars) + len(inflow_vars) + i] = adata_gem[:, gem].X.flatten()

    flow_variable_types = adata_outflow.var['type'].tolist() \
                            + adata_inflow.var['type'].tolist() \
                            + adata_gem.var['type'].tolist()
    
    flow_downstream_tfs = adata_outflow.var['downstream_tfs'].tolist() \
                            + adata_inflow.var['downstream_tfs'].tolist() \
                            + adata_gem.var['downstream_tfs'].tolist()
    
    flow_interactions = adata_outflow.var['interactions'].tolist() \
                            + adata_inflow.var['interactions'].tolist() \
                            + adata_gem.var['interactions'].tolist()

    # Store the type, relevant downstream_TF, and received interactions for each variable
    # Store all the information on the flow variables
    flow_var_info = pd.DataFrame(index = pd.Index(flow_variables),
                                 data = {'Type': flow_variable_types,
                                      'Downstream_TF': flow_downstream_tfs,
                                      'Interaction': flow_interactions})
    
    adata.obsm[flowsig_expr_key] = flow_expressions
    adata.uns[flowsig_network_key] = {'flow_var_info': flow_var_info}
    
def construct_inflow_signals_cellphonedb(adata: sc.AnnData,
                                    cellphonedb_output_key: str,
                                    cellphonedb_active_tfs_key: str):
    
    ccc_output_merged = pd.concat([adata.uns[cellphonedb_output_key][sample] for sample in adata.uns[cellphonedb_output_key]])
    cpdb_active_tfs_merged = pd.concat([adata.uns[cellphonedb_active_tfs_key][sample] for sample in adata.uns[cellphonedb_active_tfs_key]])
    ccc_interactions = ccc_output_merged['interacting_pair'].unique().tolist()

    unique_inflow_vars_and_interactions = {}

    # The inflow variables are constructed from receptor gene expression
    # that will be weighted by the average expression of their downstream
    # TF targets. These can be multi-units, but because of the weighting,
    # we hestitate to call them receptors in the context of cellular flows.
    for i, interaction in enumerate(ccc_interactions):
        receptors = ccc_output_merged[ccc_output_merged['interacting_pair'] == interaction]['gene_b'].dropna().unique().tolist()
        
        for receptor in receptors:
            if receptor in adata.var_names:
                if receptor not in unique_inflow_vars_and_interactions:
                    unique_inflow_vars_and_interactions[receptor] = [interaction]
                    
                else:
                    interactions_for_receptor = unique_inflow_vars_and_interactions[receptor]
                    interactions_for_receptor.append(interaction)  
                    unique_inflow_vars_and_interactions[receptor] = interactions_for_receptor
                
    inflow_vars = sorted(list(unique_inflow_vars_and_interactions.keys()))
    num_inflow_vars = len(inflow_vars)

    inflow_expressions = np.zeros((adata.n_obs, num_inflow_vars)) 

    for i, receptor in enumerate(inflow_vars):
        
        inflow_expressions[:, i] = adata[:, receptor].X.toarray().flatten()
        
    inflow_expressions_adjusted = inflow_expressions.copy()

    inflow_interactions = []
    unique_inflow_vars_and_tfs = {}

    for i, receptor in enumerate(inflow_vars):
        
        interactions_for_receptor = unique_inflow_vars_and_interactions[receptor]
        inflow_interactions.append('/'.join(sorted(interactions_for_receptor)))
        
        # Go through each category
        possible_downstream_tfs = cpdb_active_tfs_merged[cpdb_active_tfs_merged['gene_b'] == receptor]['active_TF'].dropna().unique().tolist()
                    
        if len(possible_downstream_tfs) != 0:
            downstream_tfs = np.intersect1d(possible_downstream_tfs, adata.var_names)
        
            unique_inflow_vars_and_tfs[receptor] = sorted(list(downstream_tfs))

        else:
            unique_inflow_vars_and_tfs[receptor] = ''
        
        if len(possible_downstream_tfs) != 0:
            
            average_tf_expression = np.zeros((adata.n_obs, ))
            
            for tf in possible_downstream_tfs:
                
                average_tf_expression += adata[:, tf].X.toarray().flatten()
                
            average_tf_expression /= len(possible_downstream_tfs)
            
            inflow_expressions_adjusted[:, i] *= average_tf_expression
            
    inflow_downstream_tfs = []
    for i, inflow_var in enumerate(inflow_vars):
        
        interactions_for_receptor = unique_inflow_vars_and_interactions[inflow_var]
        downstream_tfs = unique_inflow_vars_and_tfs[inflow_var]
        inflow_downstream_tfs.append('_'.join(sorted(downstream_tfs)))
        
    adata_inflow = sc.AnnData(X=inflow_expressions_adjusted)
    adata_inflow.var.index = pd.Index(inflow_vars)
    adata_inflow.var['downstream_tfs'] = inflow_downstream_tfs
    adata_inflow.var['type'] = 'inflow' # Define variable types
    adata_inflow.var['interactions'] = inflow_interactions

    return adata_inflow, inflow_vars

def construct_outflow_signals_cellphonedb(adata: sc.AnnData,
                                    cellphonedb_output_key: str):
    
    cellphonedb_output_merged = pd.concat([adata.uns[cellphonedb_output_key][sample] for sample in adata.uns[cellphonedb_output_key]])
    cellphonedb_interactions = cellphonedb_output_merged['interacting_pair'].unique().tolist()
    outflow_vars = []
    relevant_interactions = {}

    for inter in cellphonedb_interactions:

        ligands = cellphonedb_output_merged[cellphonedb_output_merged['interacting_pair'] == inter]['gene_a'].dropna().unique().tolist()

        for ligand in ligands:
            if (ligand in adata.var_names):


                if (ligand not in outflow_vars):
                    outflow_vars.append(ligand)

                    relevant_interactions[ligand] = [inter]

                else:
                    interactions_with_ligand = relevant_interactions[ligand]
                    interactions_with_ligand.append(inter)
                    relevant_interactions[ligand] = interactions_with_ligand

    outflow_expressions = np.zeros((adata.n_obs, len(outflow_vars)))

    for i, signal in enumerate(outflow_vars):
        outflow_expressions[:, i] = adata[:, signal].X.toarray().flatten()

    adata_outflow = sc.AnnData(X=outflow_expressions)
    adata_outflow.var.index = pd.Index(outflow_vars)
    adata_outflow.var['downstream_tfs'] = '' # For housekeeping for later
    adata_outflow.var['type'] = 'outflow' # Define variable types

    relevant_interactions_of_ligands = []
    for ligand in outflow_vars:
        interactions_of_ligand = relevant_interactions[ligand]
        relevant_interactions_of_ligands.append('/'.join(interactions_of_ligand))
        
    adata_outflow.var['interactions'] = relevant_interactions_of_ligands # Define variable types

    return adata_outflow, outflow_vars

def construct_flows_from_cellphonedb(adata: sc.AnnData,
                                cellphonedb_output_key: str,
                                cellphonedb_tfs_key: str,
                                gem_expr_key: str = 'X_gem',
                                scale_gem_expr: bool = True,
                                model_organism: str = 'human',
                                flowsig_network_key: str = 'flowsig_network',
                                flowsig_expr_key: str = 'X_flow'):

    if model_organism != 'human':
        ValueError("CellPhoneDB only supports human data.")

    # Define the expression
    adata_outflow, outflow_vars = construct_outflow_signals_cellphonedb(adata, cellphonedb_output_key)

    adata_inflow, inflow_vars = construct_inflow_signals_cellphonedb(adata, cellphonedb_output_key, cellphonedb_tfs_key)

    adata_gem, flow_gem_vars = construct_gem_expressions(adata, gem_expr_key, scale_gem_expr)

    # Determine the flow_variables
    flow_variables = outflow_vars + inflow_vars + flow_gem_vars

    flow_expressions = np.zeros((adata.n_obs, len(flow_variables)))

    for i, outflow_var in enumerate(outflow_vars):
        flow_expressions[:, i] = adata_outflow[:, outflow_var].X.toarray().flatten()

    for i, inflow_var in enumerate(inflow_vars):
        flow_expressions[:, len(outflow_vars) + i] = adata_inflow[:, inflow_var].X.toarray().flatten()

    for i, gem in enumerate(flow_gem_vars):
        flow_expressions[:, len(outflow_vars) + len(inflow_vars) + i] = adata_gem[:, gem].X.flatten()

    flow_variable_types = adata_outflow.var['type'].tolist() \
                            + adata_inflow.var['type'].tolist() \
                            + adata_gem.var['type'].tolist()
    
    flow_downstream_tfs = adata_outflow.var['downstream_tfs'].tolist() \
                            + adata_inflow.var['downstream_tfs'].tolist() \
                            + adata_gem.var['downstream_tfs'].tolist()
    
    flow_interactions = adata_outflow.var['interactions'].tolist() \
                            + adata_inflow.var['interactions'].tolist() \
                            + adata_gem.var['interactions'].tolist()

    # Store the type, relevant downstream_TF, and received interactions for each variable
    # Store all the information on the flow variables
    flow_var_info = pd.DataFrame(index = pd.Index(flow_variables),
                                 data = {'Type': flow_variable_types,
                                      'Downstream_TF': flow_downstream_tfs,
                                      'Interaction': flow_interactions})
    
    adata.obsm[flowsig_expr_key] = flow_expressions
    adata.uns[flowsig_network_key] = {'flow_var_info': flow_var_info}
    
def construct_inflow_signals_liana(adata: sc.AnnData,
                                    liana_output_key: str, 
                                    use_tfs: bool = False,
                                    model_organism: str = 'human'):
    
    model_organisms = ['human', 'mouse']

    if model_organism not in model_organisms:
        raise ValueError ("Invalid model organism. Please select one of: %s" % model_organisms)
    
    if use_tfs:
    
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        data_path = os.path.join(data_dir, 'cellchat_interactions_and_tfs_' + model_organism + '.csv')

        cellchat_interactions_and_tfs = pd.read_csv(data_path, index_col=0)

        ccc_output_merged = pd.concat([adata.uns[liana_output_key][sample] for sample in adata.uns[liana_output_key]])
        inflow_vars = sorted(ccc_output_merged['receptor_complex'].unique().tolist())
        inflow_vars = [var.replace('_', '+') for var in inflow_vars]

        unique_inflow_vars_and_interactions = {}

        # The inflow variables are constructed from receptor gene expression
        # that will be weighted by the average expression of their downstream
        # TF targets. These can be multi-units, but because of the weighting,
        # we hestitate to call them receptors in the context of cellular flows.
        for inflow_var in inflow_vars:
            relevant_rec = inflow_var.replace('+', '_')

            relevant_ligands = sorted(ccc_output_merged[ccc_output_merged['receptor_complex'] == relevant_rec]['ligand_complex'].unique().tolist())
            interactions_for_receptor = []

            for lig in relevant_ligands:
                if '+' in inflow_var:
                    interactions_for_receptor.append(lig + ' - ' + '(' + inflow_var + ')')
                else:
                    interactions_for_receptor.append(lig + ' - ' + inflow_var)

            unique_inflow_vars_and_interactions[inflow_var] = interactions_for_receptor
                
        inflow_vars = sorted(list(unique_inflow_vars_and_interactions.keys()))
        num_inflow_vars = len(inflow_vars)

        inflow_expressions = np.zeros((adata.n_obs, num_inflow_vars)) 

        for i, receptor in enumerate(inflow_vars):
            
            receptor_expression = np.ones((adata.n_obs, ))
            
            split_receptor = receptor.split('+')
            considered_receptors = []
            
            for unit in split_receptor:
                
                if unit not in adata.var_names:
                    print(unit)
                    
                else:
                    
                    unit_expression = adata[:, unit].X.toarray().flatten()

                    receptor_expression *= unit_expression
                    considered_receptors.append(unit)
                    
            if len(considered_receptors) != 0:
                inflow_expressions[:, i] = receptor_expression**(1.0 / len(considered_receptors))
            
        inflow_expressions_adjusted = inflow_expressions.copy()

        inflow_interactions = []
        unique_inflow_vars_and_tfs = {}

        for i, receptor in enumerate(inflow_vars):
            
            interactions_for_receptor = unique_inflow_vars_and_interactions[receptor]
            
            relevant_interactions = cellchat_interactions_and_tfs[cellchat_interactions_and_tfs['interaction_name_2'].isin(interactions_for_receptor)]
            
            joined_interactions = '/'.join(sorted(interactions_for_receptor))
            inflow_interactions.append(joined_interactions)
            
            downstream_tfs = []
            
            # Go through each category
            possible_downstream_tfs = relevant_interactions['Receptor-TF-combined'].dropna().tolist()
            
            for unit in possible_downstream_tfs:
                split_unit = unit.split('_')
                for tf in split_unit:
                    if tf not in downstream_tfs:
                        downstream_tfs.append(tf)
                        
            downstream_tfs = np.intersect1d(downstream_tfs, adata.var_names)
            
            unique_inflow_vars_and_tfs[receptor] = sorted(list(downstream_tfs))
            
            if len(downstream_tfs) != 0:
                
                average_tf_expression = np.zeros((adata.n_obs, ))
                
                for tf in downstream_tfs:
                    
                    average_tf_expression += adata[:, tf].X.toarray().flatten()
                    
                average_tf_expression /= len(downstream_tfs)
                
                inflow_expressions_adjusted[:, i] *= average_tf_expression
                
        inflow_downstream_tfs = []
        for i, inflow_var in enumerate(inflow_vars):
            
            interactions_for_receptor = unique_inflow_vars_and_interactions[inflow_var]
            downstream_tfs = unique_inflow_vars_and_tfs[inflow_var]
            inflow_downstream_tfs.append('_'.join(sorted(downstream_tfs)))
            
        adata_inflow = sc.AnnData(X=inflow_expressions_adjusted)
        adata_inflow.var.index = pd.Index(inflow_vars)
        adata_inflow.var['downstream_tfs'] = inflow_downstream_tfs
        adata_inflow.var['type'] = 'inflow' # Define variable types
        adata_inflow.var['interactions'] = inflow_interactions

        return adata_inflow, inflow_vars
    
def construct_outflow_signals_liana(adata: sc.AnnData,
                                    liana_output_key: str, 
                                    ):
    liana_output_merged = pd.concat([adata.uns[liana_output_key][sample] for sample in adata.uns[liana_output_key]])
   
    outflow_vars = sorted(liana_output_merged['ligand_complex'].unique().tolist())
    outflow_vars = [var.replace('_', '+') for var in outflow_vars]

    relevant_interactions = {}

    for outflow_var in outflow_vars:
            
        relevant_lig = outflow_var.replace('+', '_')

        interactions_with_ligand = []

        relevant_receptors = sorted(liana_output_merged[liana_output_merged['ligand_complex'] == relevant_lig]['receptor_complex'].unique().tolist())

        for rec in relevant_receptors:
            
            inter = outflow_var + ' - ' + rec.replace('_', '+')
            interactions_with_ligand.append(inter)

        relevant_interactions[outflow_var] = interactions_with_ligand

    outflow_expressions = np.zeros((adata.n_obs, len(outflow_vars)))

    for i, ligand in enumerate(outflow_vars):
            
        ligand_expression = np.ones((adata.n_obs, ))

        split_ligand = ligand.split('+')
        considered_ligands = []
        
        for unit in split_ligand:
            
            if unit not in adata.var_names:
                print(unit)
                
            else:
                
                unit_expression = adata[:, unit].X.toarray().flatten()

                ligand_expression *= unit_expression
                considered_ligands.append(unit)
                
        if len(considered_ligands) != 0:
            outflow_expressions[:, i] = ligand_expression**(1.0 / len(considered_ligands))

    adata_outflow = sc.AnnData(X=outflow_expressions)
    adata_outflow.var.index = pd.Index(outflow_vars)
    adata_outflow.var['downstream_tfs'] = '' # For housekeeping for later
    adata_outflow.var['type'] = 'outflow' # Define variable types

    relevant_interactions_of_ligands = []
    for ligand in outflow_vars:
        interactions_of_ligand = relevant_interactions[ligand]
        relevant_interactions_of_ligands.append('/'.join(interactions_of_ligand))
        
    adata_outflow.var['interactions'] = relevant_interactions_of_ligands # Define variable types

    return adata_outflow, outflow_vars
   
def construct_flows_from_liana(adata: sc.AnnData,
                                liana_output_key: str,
                                gem_expr_key: str = 'X_gem',
                                scale_gem_expr: bool = True,
                                use_tfs: bool = False,
                                model_organism: str= 'mouse',
                                flowsig_network_key: str = 'flowsig_network',
                                flowsig_expr_key: str = 'X_flow'):

    # Define the expression
    adata_outflow, outflow_vars = construct_outflow_signals_liana(adata, liana_output_key)

    adata_inflow, inflow_vars = construct_inflow_signals_liana(adata, liana_output_key, use_tfs, model_organism)

    adata_gem, flow_gem_vars = construct_gem_expressions(adata, gem_expr_key, scale_gem_expr)

    # Determine the flow_variables
    flow_variables = outflow_vars + inflow_vars + flow_gem_vars

    flow_expressions = np.zeros((adata.n_obs, len(flow_variables)))

    for i, outflow_var in enumerate(outflow_vars):
        flow_expressions[:, i] = adata_outflow[:, outflow_var].X.toarray().flatten()

    for i, inflow_var in enumerate(inflow_vars):
        flow_expressions[:, len(outflow_vars) + i] = adata_inflow[:, inflow_var].X.toarray().flatten()

    for i, gem in enumerate(flow_gem_vars):
        flow_expressions[:, len(outflow_vars) + len(inflow_vars) + i] = adata_gem[:, gem].X.flatten()

    flow_variable_types = adata_outflow.var['type'].tolist() \
                            + adata_inflow.var['type'].tolist() \
                            + adata_gem.var['type'].tolist()
    
    flow_downstream_tfs = adata_outflow.var['downstream_tfs'].tolist() \
                            + adata_inflow.var['downstream_tfs'].tolist() \
                            + adata_gem.var['downstream_tfs'].tolist()
    
    flow_interactions = adata_outflow.var['interactions'].tolist() \
                            + adata_inflow.var['interactions'].tolist() \
                            + adata_gem.var['interactions'].tolist()

    # Store the type, relevant downstream_TF, and received interactions for each variable
    # Store all the information on the flow variables
    flow_var_info = pd.DataFrame(index = pd.Index(flow_variables),
                                 data = {'Type': flow_variable_types,
                                      'Downstream_TF': flow_downstream_tfs,
                                      'Interaction': flow_interactions})
    
    adata.obsm[flowsig_expr_key] = flow_expressions
    adata.uns[flowsig_network_key] = {'flow_var_info': flow_var_info}

def _commot_ligand_interactions(adata, commot_output_key, ligand, direction):
    """Resolve exact LR columns from database identities, not gene prefixes."""
    database = adata.uns[commot_output_key + '-info']['df_ligrec']
    columns = adata.obsm[commot_output_key + '-sum-' + direction].columns
    prefix = 'r-' if direction == 'receiver' else 's-'
    interactions = dict.fromkeys(
        f"{row.ligand}-{row.receptor}"
        for row in database.loc[database['ligand'] == ligand].itertuples()
    )
    return [interaction for interaction in interactions if prefix + interaction in columns]


def construct_inflow_signals_commot(adata: sc.AnnData,
                                    commot_output_key: str):

    # Inflow variables are inferred from outflow variables
    outflow_vars = sorted(adata.uns[commot_output_key + '-info']['df_ligrec']['ligand'].unique().tolist())
    inflow_vars = ['inflow-' + outflow_var for outflow_var in outflow_vars]


    inflow_interactions = []
    inflow_expressions = np.zeros((adata.n_obs, len(inflow_vars)))
    for i, inflow_var in enumerate(inflow_vars):
        lig = outflow_vars[i]
        inferred_interactions = _commot_ligand_interactions(
            adata, commot_output_key, lig, 'receiver'
        )
        inflow_interactions.append('/'.join(sorted(inferred_interactions)))

        # We sum the total received signal across each interaction at each spot
        for inter in inferred_interactions:
            inflow_expressions[:, i] += adata.obsm[commot_output_key + '-sum-receiver']['r-' + inter]

    adata_inflow = sc.AnnData(X=inflow_expressions)
    adata_inflow.var.index = pd.Index(inflow_vars)
    adata_inflow.var['downstream_tfs'] = ''
    adata_inflow.var['type'] = 'inflow' 
    adata_inflow.var['interactions'] = inflow_interactions

    return adata_inflow, inflow_vars

def construct_outflow_signals_commot(adata: sc.AnnData,
                                    commot_output_key: str, 
                                    ):
    
    # Inflow variables are inferred from outflow variables
    outflow_vars = sorted(adata.uns[commot_output_key + '-info']['df_ligrec']['ligand'].unique().tolist())
    outflow_vars = [var for var in outflow_vars if var in adata.var_names]

    outflow_interactions = []
    outflow_expressions = np.zeros((adata.n_obs, len(outflow_vars)))

    for i, outflow_var in enumerate(outflow_vars):

        inferred_interactions = _commot_ligand_interactions(
            adata, commot_output_key, outflow_var, 'sender'
        )
        outflow_interactions.append('/'.join(sorted(inferred_interactions)))

        # Outflow signal expression is simply ligand gene expression
        outflow_expressions[:, i] += adata[:, outflow_var].X.toarray().flatten()

    adata_outflow = sc.AnnData(X=outflow_expressions)
    adata_outflow.var.index = pd.Index(outflow_vars)
    adata_outflow.var['downstream_tfs'] = ''
    adata_outflow.var['type'] = 'outflow' 
    adata_outflow.var['interactions'] = outflow_interactions

    return adata_outflow, outflow_vars

def construct_flows_from_commot(adata: sc.AnnData,
                                commot_output_key: str,
                                gem_expr_key: str = 'X_gem',
                                scale_gem_expr: bool = True,
                                flowsig_network_key: str = 'flowsig_network',
                                flowsig_expr_key: str = 'X_flow'):
    
    # Define the expression
    adata_outflow, outflow_vars = construct_outflow_signals_commot(adata, commot_output_key)

    adata_inflow, inflow_vars = construct_inflow_signals_commot(adata, commot_output_key)

    adata_gem, flow_gem_vars = construct_gem_expressions(adata, gem_expr_key, scale_gem_expr)

    # Determine the flow_variables
    flow_variables = outflow_vars + inflow_vars + flow_gem_vars

    flow_expressions = np.zeros((adata.n_obs, len(flow_variables)))

    for i, outflow_var in enumerate(outflow_vars):
        flow_expressions[:, i] = adata_outflow[:, outflow_var].X.toarray().flatten()

    for i, inflow_var in enumerate(inflow_vars):
        flow_expressions[:, len(outflow_vars) + i] = adata_inflow[:, inflow_var].X.toarray().flatten()

    for i, gem in enumerate(flow_gem_vars):
        flow_expressions[:, len(outflow_vars) + len(inflow_vars) + i] = adata_gem[:, gem].X.flatten()

    flow_variable_types = adata_outflow.var['type'].tolist() \
                            + adata_inflow.var['type'].tolist() \
                            + adata_gem.var['type'].tolist()
    
    flow_downstream_tfs = adata_outflow.var['downstream_tfs'].tolist() \
                            + adata_inflow.var['downstream_tfs'].tolist() \
                            + adata_gem.var['downstream_tfs'].tolist()
    
    flow_interactions = adata_outflow.var['interactions'].tolist() \
                            + adata_inflow.var['interactions'].tolist() \
                            + adata_gem.var['interactions'].tolist()

    # Store the type, relevant downstream_TF, and received interactions for each variable
    # Store all the information on the flow variables
    flow_var_info = pd.DataFrame(index = pd.Index(flow_variables),
                                 data = {'Type': flow_variable_types,
                                      'Downstream_TF': flow_downstream_tfs,
                                      'Interaction': flow_interactions})
    
    adata.obsm[flowsig_expr_key] = flow_expressions
    adata.uns[flowsig_network_key] = {'flow_var_info': flow_var_info}

