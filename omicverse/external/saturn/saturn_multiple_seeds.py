import subprocess
import os
import time

import numpy as np
import argparse

import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train many SATURN runs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run', help='run file path')
    parser.add_argument('--seeds', help='How many seeds to run?')
    parser.add_argument('--macrogenes', help='Number of macrogenes, default is 2000')
    parser.add_argument('--gpus', nargs='+', help='<Required> Which GPUs to use', required=True)
    parser.add_argument('--embedding_model', type=str, choices=['ESM1b', 'MSA1b', 'protXL', 'ESM1b_protref', 'ESM2'],
                    help='Gene embedding model whose embeddings should be loaded if using gene_embedding_method')
    parser.add_argument('--in_label_col', help='which column to use as labels')
    parser.add_argument('--ref_label_col', help='which column to use as ref labels')
    parser.add_argument('--batch_label_col', help='which column to use as batch variable for autoencoder')
    
    
    parser.add_argument('--l1_penalty', type=float,
                        help='L1 Penalty hyperparameter Default is 0')
    parser.add_argument('--pe_sim_penalty', type=float,
                        help='Protein Embedding similarity to Macrogene loss, weight hyperparameter. Default is 1.0')
    parser.add_argument('--hv_genes', help='How many hv genes to subset to?')
    parser.add_argument('--centroid_score_func', type=str, choices=['default', 'one_hot', 'smoothed'],
                    help='Gene embedding model whose embeddings should be loaded if using gene_embedding_method')
    # python saturn_multiple_seeds.py --run=frog_zebrafish_run.csv --seeds=30 --embedding_model=protXL --gpus 2 3 4 5 6 7
    # python saturn_multiple_seeds.py --run=frog_zebrafish_run_rand.csv --in_label_col=random_cell_type --ref_label_col=cell_type --seeds=10 --gpus 3 8 9
    
    
    
    parser.set_defaults(
        run="frog_zebrafish_run.csv",
        seeds=30,
        embedding_model="ESM2",
        macrogenes=2000,
        in_label_col="cell_type",
        ref_label_col="cell_type", # not used for F/Z, just a duplicate column name
        l1_penalty=0,
        pe_sim_penalty=1.0,
        batch_label_col=None,
        hv_genes=8000,
        centroid_score_func="default"
    )
    
    
    args = parser.parse_args()
    
    available_gpus = args.gpus
    embedding_model = args.embedding_model
    print(available_gpus)
    
    seeds = np.arange(int(args.seeds))
    command_part_1 = f"python train-saturn.py --in_data={args.run} --device_num="
    
    command_part_2 = f" --in_label_col={args.in_label_col} --ref_label_col={args.ref_label_col} --work_dir=./Vignettes/multiple_seeds_results/ --num_macrogenes={args.macrogenes} --pretrain --model_dim=256 --polling_freq=201 --ref_label_col=cell_type --epochs=50 --hv_genes={args.hv_genes} --pretrain_epochs=200 --pe_sim_penalty={args.pe_sim_penalty} --l1_penalty={args.l1_penalty} --centroid_score_func={args.centroid_score_func} --seed="
    
    org = args.run.split("/")[-1].replace(".csv", "")
    org += f"_l1_{args.l1_penalty}_pe_{args.pe_sim_penalty}"
    org += f"_{embedding_model}"
    org += f"_macrogenes_{args.macrogenes}"
    org += f"_hv_genes_{args.hv_genes}"
    org += f"_centroid_score_func_{args.centroid_score_func}"
    
    if args.batch_label_col is not None:
        org += f"_batch_label_{args.batch_label_col}"
        command_part_3 = f" --non_species_batch_col={args.batch_label_col} --org={org} "
    else:
        command_part_3 = f" --org={org} "
        
    
    
    if embedding_model is not None:
        command_part_3 += f"--embedding_model={embedding_model}  "
    command_part_3 += f"--centroids_init_path=./Vignettes/multiple_seeds_results/saturn_{org}_seed_"
        
    processes = {}


    max_processes = len(available_gpus)

    for seed in tqdm.tqdm(seeds):
        # take my device off of the list of available gpus
        device = available_gpus.pop(0)    
        print(f"RUNNING SEED: {seed} ON GPU:{device}")
        processes[device] = subprocess.Popen((command_part_1 + str(device) + command_part_2 + str(seed) + command_part_3 + str(seed)).split(), stdout=subprocess.DEVNULL) # add the process hide all output
        # , stdout=subprocess.DEVNULL

        if len(processes) >= max_processes:
            os.wait() # waits until a child proc terminates

            devices_done = []
            for device, p in processes.items():
                if p.poll() is not None:
                    devices_done.append(device)
                    if p.returncode != 0:
                        # There was an error using this GPU / seed
                        print(f"ERROR on GPU: {device} for seed {seed}")
                    # poll returns none if process han't completed
            for device in devices_done:
                available_gpus.append(device) # add the device as an option again
                processes.pop(device) # remove this device : proc pair
            # if 

    #Check if all the child processes were closed
    for p in processes.values():
        if p.poll() is None:
            p.wait()      
