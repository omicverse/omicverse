import anndata
import os
import torch
import time
import torch as th
import torch.distributed as dist
import numpy as np
class scDiffusion(object):

    def __init__(
            self,
            adata,
            device='cuda:0'
    ):
        from ..utils import check_dependencies
        check_dependencies(
            ['blobfile','mpi4py']
        )
        self.adata=adata
        self.device=device
        self.num_genes=self.adata.shape[1]

    def prepare_VAE(
            self,
            state_dict=None,
            cell_type='celltype',
            batch_size=128,
            num_genes=18996,
            seed=1234,
            loss_ae="mse",
            decoder_activation='ReLU',
    ):
        from ..external.scdiffusion.VAE.VAE_model import VAE
        from ..external.scdiffusion.guided_diffusion.cell_datasets_loader import load_data,prepare_data
        print('device: ',self.device)

        datasets = prepare_data(
            adata=self.adata,
            batch_size=batch_size,
            train_vae=True,
            cell_type=cell_type,
        )

        autoencoder = VAE(
            num_genes=num_genes,
            device=self.device,
            seed=seed,
            loss_ae=loss_ae,
            hidden_dim=128,
            decoder_activation=decoder_activation,
        )
        if state_dict is not None:
            print('loading pretrained model from: \n',state_dict)
            use_gpu = "cuda" in self.device
            autoencoder.encoder.load_state(state_dict["encoder"], use_gpu)
            autoencoder.decoder.load_state(state_dict["decoder"], use_gpu)

        return autoencoder, datasets



    def train_VAE(
            self,
            state_dict=None,
            cell_type='celltype',
            loss_ae="mse",
            decoder_activation='ReLU',
            local_rank=0,
            split_seed=1234,
            num_genes=18996,
            seed=1234,
            hparams="",
            batch_size=128,
            max_steps=200000,
            max_minutes=3000,
            checkpoint_freq=50000,
            save_dir=None,
            sweep_seeds=200,
            return_model=False,
    ):
        
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "result/VAE_model")
            os.makedirs(save_dir, exist_ok=True)
        
        if state_dict is not None:
            filenames = {}
            checkpoint_path = {
                "encoder": os.path.join(
                    state_dict, filenames.get("model", "encoder.ckpt")
                ),
                "decoder": os.path.join(
                    state_dict, filenames.get("model", "decoder.ckpt")
                ),
                "gene_order": os.path.join(
                    state_dict, filenames.get("gene_order", "gene_order.tsv")
                ),
            }
            autoencoder, datasets = self.prepare_VAE(
                state_dict=checkpoint_path,
                cell_type=cell_type,
                batch_size=batch_size,
                num_genes=num_genes,
                seed=seed,
                loss_ae=loss_ae,
                decoder_activation=decoder_activation)
        else:
            autoencoder, datasets = self.prepare_VAE(
                state_dict=None,
                cell_type=cell_type,
                batch_size=batch_size,
                num_genes=num_genes,
                seed=seed,
                loss_ae=loss_ae,
                decoder_activation=decoder_activation)
        hparams = autoencoder.hparams

        start_time = time.time()
        from tqdm import tqdm
        for step in tqdm(range(max_steps)):

            genes, _ = next(datasets)

            minibatch_training_stats = autoencoder.train(genes)

            if step % 1000 == 0:
                for key, val in minibatch_training_stats.items():
                    print('step ', step, 'loss ', val)

            ellapsed_minutes = (time.time() - start_time) / 60

            stop = ellapsed_minutes > max_minutes or (
                step == max_steps - 1
            )

            if ((step % checkpoint_freq) == 0 or stop):

                os.makedirs(save_dir,exist_ok=True)
                torch.save(
                    autoencoder.state_dict(),
                    os.path.join(
                        save_dir,
                        "model_seed={}_step={}.pt".format(seed, step),
                    ),
                )

                if stop:
                    break

        if return_model:
            return autoencoder, datasets
        

    def train_cellmodel(
            self,
            cell_type='celltype',
            schedule_sampler="uniform",
            lr=1e-4,
            weight_decay=0.0001,
            lr_anneal_steps=500000,
            batch_size=12,
            microbatch=-1,  # -1 disables microbatches
            ema_rate="0.9999",  # comma-separated list of EMA values
            log_interval=100,
            save_interval=200000,
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
            vae_path = 'output/Autoencoder_checkpoint/muris_AE/model_seed=0_step=0.pt',
            model_name="muris_diffusion",
            save_dir='output/diffusion_checkpoint'
    ):
        from ..external.scdiffusion.guided_diffusion import dist_util, logger
        from ..external.scdiffusion.guided_diffusion.cell_datasets_loader import load_data
        from ..external.scdiffusion.guided_diffusion.resample import create_named_schedule_sampler
        from ..external.scdiffusion.guided_diffusion.script_util import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
            args_to_dict,
            add_dict_to_argparser,
        )
        from ..external.scdiffusion.guided_diffusion.train_util import TrainLoop
        from ..external.scdiffusion.guided_diffusion.cell_datasets_loader import load_data,prepare_data

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dist_util.setup_dist()
        logger.configure(dir='../output/logs/'+model_name)  # log file

        logger.log("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(
                input_dim = 128,
                hidden_dim = [512,512,256,128],
                dropout = 0.0,
                learn_sigma=False,
                diffusion_steps=1000,
                noise_schedule="linear",
                timestep_respacing="",
                use_kl=False,
                predict_xstart=False,
                rescale_timesteps=False,
                rescale_learned_sigmas=False,
                class_cond=False,
        )
        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(schedule_sampler, diffusion)

        logger.log("creating data loader...")
        data = prepare_data(
            adata=self.adata, 
            batch_size=batch_size,
            vae_path=vae_path,
            train_vae=False,
            cell_type=cell_type,
        )


        logger.log("training...")
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=batch_size,
            microbatch=microbatch,
            lr=lr,
            ema_rate=ema_rate,
            log_interval=log_interval,
            save_interval=save_interval,
            resume_checkpoint=resume_checkpoint,
            use_fp16=use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=weight_decay,
            lr_anneal_steps=lr_anneal_steps,
            model_name=model_name,
            save_dir=save_dir
        ).run_loop()


    def train_classifier(
            self,
            model_name="classifier_muris",
            save_dir='output/classifier_checkpoint/classifier_muris',
            noised=True,
            resume_checkpoint="",
            use_fp16=False,
            batch_size=128,
            vae_path='output/Autoencoder_checkpoint/muris_AE/model_seed=0_step=0.pt',
            cell_type='celltype',
            val_adata=None,
            input_dim=128,
            lr=3e-4,
            weight_decay=0.0,
            start_guide_time=500,
            microbatch=-1,
            model_path='output/classifier_checkpoint/classifier_muris',
            iterations=500000,
            log_interval=100,
            eval_interval=100,
            save_interval=100000,
            anneal_lr=False,
            num_class=12,
            schedule_sampler="uniform",
            
    ):
        from ..external.scdiffusion.guided_diffusion import dist_util, logger
        from ..external.scdiffusion.guided_diffusion.cell_datasets_loader import load_data
        from ..external.scdiffusion.guided_diffusion.resample import create_named_schedule_sampler
        from ..external.scdiffusion.guided_diffusion.script_util import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
            args_to_dict,
            add_dict_to_argparser,
            create_classifier_and_diffusion
        )
        from ..external.scdiffusion.guided_diffusion.train_util import log_loss_dict,parse_resume_step_from_filename
        from ..external.scdiffusion.guided_diffusion.cell_datasets_loader import load_data,prepare_data
        from torch.nn.parallel.distributed import DistributedDataParallel as DDP
        from torch.optim import AdamW
        import torch.nn.functional as F

        from ..external.scdiffusion.guided_diffusion.fp16_util import MixedPrecisionTrainer

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dist_util.setup_dist()
        logger.configure(dir='../output/logs/'+model_name)  # log file

        logger.log("creating model and diffusion...")
        model, diffusion = create_classifier_and_diffusion(
                input_dim = input_dim,
                hidden_dim = [512,512,256,128],
                dropout = 0.1,
                classifier_use_fp16=use_fp16,
                num_class = num_class,
                learn_sigma=False,
                diffusion_steps=1000,
                noise_schedule="linear",
                timestep_respacing="",
                use_kl=False,
                predict_xstart=False,
                rescale_timesteps=False,
                rescale_learned_sigmas=False,
                class_cond=False,
        )
        model.to(dist_util.dev())
        if noised:
            schedule_sampler = create_named_schedule_sampler(
                schedule_sampler, diffusion
            )
        resume_step = 0
        if resume_checkpoint:
            resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(
                    f"loading model from checkpoint: {resume_checkpoint}... at {resume_step} step"
                )
                model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        # Needed for creating correct EMAs and fp16 parameters.
        dist_util.sync_params(model.parameters())

        mp_trainer = MixedPrecisionTrainer(
            model=model, use_fp16=use_fp16, initial_lg_loss_scale=16.0
        )

        model = DDP(
            model,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        logger.log("creating data loader...")
        data = prepare_data(
            adata=self.adata, 
            batch_size=batch_size,
            vae_path=vae_path,
            train_vae=False,
            cell_type=cell_type,
        )
        if val_adata is not None:
            val_data = prepare_data(
                adata=val_adata, 
                batch_size=batch_size,
                vae_path=vae_path,
                hidden_dim=128,
                train_vae=False,
                cell_type=cell_type
            )
        else:
            val_data = None
        logger.log(f"creating optimizer...")
        opt = AdamW(mp_trainer.master_params, lr=lr, weight_decay=weight_decay)
        if resume_checkpoint:
            opt_checkpoint = os.path.join(
                os.path.dirname(resume_checkpoint), f"opt{resume_step:06}.pt"
            )
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            opt.load_state_dict(
                dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            )

        logger.log("training classifier model...")

        def forward_backward_log(data_loader, prefix="train"):
            batch, extra = next(data_loader)
            labels = extra["y"].to(dist_util.dev())

            batch = batch.to(dist_util.dev())
            # Noisy cells
            if noised:
                t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev(), start_guide_time=start_guide_time)
                batch = diffusion.q_sample(batch, t)
            else:
                t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

            for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(microbatch, batch, labels, t)
            ):
                logits = model(sub_batch, sub_t)
                loss = F.cross_entropy(logits, sub_labels, reduction="none")

                losses = {}
                losses[f"{prefix}_loss"] = loss.detach()
                losses[f"{prefix}_acc@1"] = compute_top_k(
                    logits, sub_labels, k=1, reduction="none"
                )

                log_loss_dict(diffusion, sub_t, losses)
                del losses
                loss = loss.mean()
                if loss.requires_grad:
                    if i == 0:
                        mp_trainer.zero_grad()
                    mp_trainer.backward(loss * len(sub_batch) / len(batch))
        model_path = save_dir
        for step in range(iterations - resume_step):
            logger.logkv("step", step + resume_step)
            logger.logkv(
                "samples",
                (step + resume_step + 1) * batch_size * dist.get_world_size(),
            )
            if anneal_lr:
                set_annealed_lr(opt, lr, (step + resume_step) / iterations)
            forward_backward_log(data)
            mp_trainer.optimize(opt)
            if val_data is not None and not step % eval_interval:
                with th.no_grad():
                    with model.no_sync():
                        model.eval()
                        forward_backward_log(val_data, prefix="val")
                        model.train()
            if not step % log_interval:
                logger.dumpkvs()
            if (
                step
                and dist.get_rank() == 0
                and not (step + resume_step) % save_interval
            ):
                logger.log("saving model...")
                save_model(mp_trainer, opt, step + resume_step, model_path)

        if dist.get_rank() == 0:
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step, model_path)
        dist.barrier()




    def load_VAE(self,
            vae_path='output/checkpoint/AE/open_problem/model_seed=1234_step=150000.pt',
            loss_ae="mse",
            decoder_activation='ReLU',
    ):
        from ..external.scdiffusion.VAE.VAE_model import VAE
        autoencoder = VAE(
            num_genes=self.num_genes,
            device=self.device,
            seed=0,
            loss_ae=loss_ae,
            hidden_dim=128,
            decoder_activation=decoder_activation,
        )
        autoencoder.load_state_dict(torch.load(vae_path))
        return autoencoder

    def generate_cell(
            self,
            input_dim=128,
            clip_denoised=False,
            num_samples=3000,
            batch_size=3000,
            use_ddim=False,
            model_path="output/diffusion_checkpoint/muris_diffusion/model600000.pt",
            vae_path='output/Autoencoder_checkpoint/muris_AE/model_seed=0_step=0.pt',
            sample_dir="output/simulated_samples/muris",
            loss_ae="mse",
            decoder_activation='ReLU',
    ):
        from ..external.scdiffusion.guided_diffusion import dist_util, logger
        from ..external.scdiffusion.guided_diffusion.cell_datasets_loader import load_data
        from ..external.scdiffusion.guided_diffusion.resample import create_named_schedule_sampler
        from ..external.scdiffusion.guided_diffusion.script_util import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
            args_to_dict,
            add_dict_to_argparser,
        )
        from ..external.scdiffusion.guided_diffusion.train_util import TrainLoop
        from ..external.scdiffusion.guided_diffusion.cell_datasets_loader import load_data,prepare_data

        dist_util.setup_dist()
        logger.configure(dir='checkpoint/sample_logs')

        logger.log("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(
                input_dim = input_dim,
                hidden_dim = [512,512,256,128],
                dropout = 0.0,
                learn_sigma=False,
                diffusion_steps=1000,
                noise_schedule="linear",
                timestep_respacing="",
                use_kl=False,
                predict_xstart=False,
                rescale_timesteps=False,
                rescale_learned_sigmas=False,
                class_cond=False,
        )
        model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        model.eval()

        logger.log("sampling...")
        all_cells = []
        while len(all_cells) * batch_size < num_samples:
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
            )
            sample, traj = sample_fn(
                model,
                (batch_size, input_dim), 
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                start_time=diffusion.betas.shape[0],
            )

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_cells.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log(f"created {len(all_cells) * batch_size} samples")

        arr = np.concatenate(all_cells, axis=0)
        dist.barrier()
        autoencoder = self.load_VAE(
            vae_path=vae_path,
            loss_ae=loss_ae,
            decoder_activation=decoder_activation,
        )
        cell_gen = autoencoder(torch.tensor(arr).cuda(),return_decoded=True).cpu().detach().numpy()

        adata=anndata.AnnData(cell_gen)
        adata.var.index=self.adata.var_names
        logger.log("sampling complete")
        return adata

        
        

    def generate_classifier(
            self,
            cell_type=[0], multi=False, inter=False, weight=[10,10],
            input_dim=128,
            hidden_dim = [512,512,256,128],
            dropout=0.0,
            clip_denoised=True,
            num_samples=9000,
            batch_size=3000,
            use_ddim=False,
            class_cond=False, 
            loss_ae="mse",
            decoder_activation='ReLU',

            model_path="output/diffusion_checkpoint/muris_diffusion/model000000.pt", 

            # ***if commen conditional generation & gradiante interpolation, use this path***
            classifier_path="output/classifier_checkpoint/classifier_muris/model000100.pt",
            # ***if multi-conditional, use this path. replace this to your own classifiers***
            classifier_path1="output/classifier_checkpoint/classifier_muris_ood_type/model200000.pt",
            classifier_path2="output/classifier_checkpoint/classifier_muris_ood_organ/model200000.pt",
            num_class1 = 2,  # set this to the number of classes in your own dataset. this is the first condition (for example cell organ).
            num_class2 = 2,  # this is the second condition (for example cell type).

            # ***if commen conditional generation, use this scale***
            classifier_scale=2,
            # ***in multi-conditional, use this scale. scale1 and scale2 are the weights of two classifiers***
            # ***in Gradient Interpolation, use this scale, too. scale1 and scale2 are the weights of two gradients***
            

            # ***if gradient interpolation, replace these base on your own situation***
            vae_path='output/Autoencoder_checkpoint/WOT/model_seed=0_step=150000.pt', 
            num_gene=19423,
            num_class=12,
            init_time = 600,    # initial noised state if interpolation
            init_cell_adata = None,   #input initial noised cell state

            sample_dir=f"output/simulated_samples/muris",
            start_guide_steps = 500,     # the time to use classifier guidance
            filter = False,   # filter the simulated cells that are classified into other condition, might take long time

    ):
        classifier_scale1=weight[0]*2/10,
        classifier_scale2=weight[1]*2/10,
        import torch.nn.functional as F
        from ..external.scdiffusion.guided_diffusion import dist_util, logger
        from ..external.scdiffusion.guided_diffusion.cell_datasets_loader import load_data
        from ..external.scdiffusion.guided_diffusion.resample import create_named_schedule_sampler
        from ..external.scdiffusion.guided_diffusion.script_util import (
            NUM_CLASSES,
            model_and_diffusion_defaults,
            classifier_and_diffusion_defaults,
            create_model_and_diffusion,
            create_classifier,
            add_dict_to_argparser,
            args_to_dict,
        )
        from ..external.scdiffusion.guided_diffusion.train_util import TrainLoop
        from ..external.scdiffusion.guided_diffusion.cell_datasets_loader import load_data,prepare_data

        dist_util.setup_dist()
        logger.configure(dir='checkpoint/sample_logs')

        logger.log("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(
                input_dim = input_dim,
                hidden_dim = [512,512,256,128],
                dropout = 0.0,
                learn_sigma=False,
                diffusion_steps=1000,
                noise_schedule="linear",
                timestep_respacing="",
                use_kl=False,
                predict_xstart=False,
                rescale_timesteps=False,
                rescale_learned_sigmas=False,
                class_cond=False,
        )
        model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        model.eval()

        logger.log("loading classifier...")
        if multi:
            num_class = num_class1 # how many classes in this condition
            classifier1 = create_classifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_class=num_class1,
                dropout=dropout,
            )
            classifier1.load_state_dict(
                dist_util.load_state_dict(classifier_path1, map_location="cpu")
            )
            classifier1.to(dist_util.dev())
            classifier1.eval()

            num_class = num_class2 # how many classes in this condition
            classifier2 = create_classifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_class=num_class2,
                dropout=dropout,
            )
            classifier2.load_state_dict(
                dist_util.load_state_dict(classifier_path2, map_location="cpu")
            )
            classifier2.to(dist_util.dev())
            classifier2.eval()

        else:
            classifier = create_classifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_class=num_class,
                dropout=dropout,
            )
            classifier.load_state_dict(
                dist_util.load_state_dict(classifier_path, map_location="cpu")
            )
            classifier.to(dist_util.dev())
            classifier.eval()

        '''
        control function for Gradient Interpolation Strategy
        '''
        def cond_fn_inter(x, t, y=None, init=None, diffusion=None):
            assert y is not None
            y1 = y[:,0]
            y2 = y[:,1]
            # xt = diffusion.q_sample(th.tensor(init,device=dist_util.dev()),t*th.ones(init.shape[0],device=dist_util.dev(),dtype=torch.long),)
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected1 = log_probs[range(len(logits)), y1.view(-1)]
                selected2 = log_probs[range(len(logits)), y2.view(-1)]
                
                grad1 = th.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * classifier_scale1
                grad2 = th.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * classifier_scale2

                # l2_loss = ((x_in-xt)**2).mean()
                # grad3 = th.autograd.grad(-l2_loss, x_in, retain_graph=True)[0] * 100

                return grad1+grad2#+grad3

        '''
        control function for multi-conditional generation
        Two conditional generation here
        '''
        def cond_fn_multi(x, t, y=None):
            assert y is not None
            y1 = y[:,0]
            y2 = y[:,1]
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits1 = classifier1(x_in, t)
                log_probs1 = F.log_softmax(logits1, dim=-1)
                selected1 = log_probs1[range(len(logits1)), y1.view(-1)]

                logits2 = classifier2(x_in, t)
                log_probs2 = F.log_softmax(logits2, dim=-1)
                selected2 = log_probs2[range(len(logits2)), y2.view(-1)]
                
                grad1 = th.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * classifier_scale1
                grad2 = th.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * classifier_scale2
                
                return grad1+grad2

        '''
        control function for one conditional generation
        '''
        def cond_fn_ori(x, t, y=None):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                grad = th.autograd.grad(selected.sum(), x_in, retain_graph=True)[0] * classifier_scale
                return grad
            
        def model_fn(x, t, y=None, init=None, diffusion=None):
            assert y is not None
            if class_cond:
                return model(x, t, y if class_cond else None)
            else:
                return model(x, t)

        if inter:
            # input real cell expression data as initial noise
            ori_adata = init_cell_adata

        logger.log("sampling...")
        all_cell = []
        sample_num = 0
        while sample_num < num_samples:
            model_kwargs = {}

            if not multi and not inter:
                classes = (cell_type[0])*th.ones((batch_size,), device=dist_util.dev(), dtype=th.long)

            if multi:
                classes1 = (cell_type[0])*th.ones((batch_size,), device=dist_util.dev(), dtype=th.long)
                classes2 = (cell_type[1])*th.ones((batch_size,), device=dist_util.dev(), dtype=th.long)
                # classes3 = ... if more conditions
                classes = th.stack((classes1,classes2), dim=1)

            if inter:
                classes1 = (cell_type[0])*th.ones((batch_size,), device=dist_util.dev(), dtype=th.long)
                classes2 = (cell_type[1])*th.ones((batch_size,), device=dist_util.dev(), dtype=th.long)
                classes = th.stack((classes1,classes2), dim=1)

            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
            )

            if inter:
                celltype = ori_adata.obs['period'].cat.categories.tolist()[cell_type[0]]
                adata = ori_adata[ori_adata.obs['period']==celltype].copy()

                start_x = adata.X
                autoencoder = self.load_VAE(vae_path, self.num_genes)
                start_x = autoencoder(torch.tensor(start_x,device=dist_util.dev()),return_latent=True).detach().cpu().numpy()

                n, m = start_x.shape  
                if n >= batch_size:  
                    start_x = start_x[:batch_size, :]  
                else:  
                    repeat_times = batch_size // n  
                    remainder = batch_size % n  
                    start_x = np.concatenate([start_x] * repeat_times + [start_x[:remainder, :]], axis=0)  
                
                noise = diffusion.q_sample(th.tensor(start_x,device=dist_util.dev()),
                            init_time*th.ones(start_x.shape[0],device=dist_util.dev(),dtype=torch.long),)
                model_kwargs["init"] = start_x
                model_kwargs["diffusion"] = diffusion

            if multi:
                sample, traj = sample_fn(
                    model_fn,
                    (batch_size, input_dim),
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn_multi,
                    device=dist_util.dev(),
                    noise = None,
                    start_time=diffusion.betas.shape[0],
                    start_guide_steps=start_guide_steps,
                )
            elif inter:
                sample, traj = sample_fn(
                    model_fn,
                    (batch_size, input_dim),
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn_inter,
                    device=dist_util.dev(),
                    noise = noise,
                    start_time=diffusion.betas.shape[0],
                    start_guide_steps=start_guide_steps,
                )
            else:
                sample, traj = sample_fn(
                    model_fn,
                    (batch_size, input_dim),
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn_ori,
                    device=dist_util.dev(),
                    noise = None,
                )

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            if filter:
                for sample in gathered_samples:
                    if multi:
                        logits1 = classifier1(sample, torch.zeros((sample.shape[0]), device=sample.device))
                        logits2 = classifier2(sample, torch.zeros((sample.shape[0]), device=sample.device))
                        prob1 = F.softmax(logits1, dim=-1)
                        prob2 = F.softmax(logits2, dim=-1)
                        type1 = torch.argmax(prob1, 1)
                        type2 = torch.argmax(prob2, 1)
                        select_index = ((type1 == cell_type[0]) & (type2 == cell_type[1]))
                        all_cell.extend([sample[select_index].cpu().numpy()])
                        sample_num += select_index.sum().item()
                    elif inter:
                        logits = classifier(sample, torch.zeros((sample.shape[0]), device=sample.device))
                        prob = F.softmax(logits, dim=-1)
                        left = (prob[:,cell_type[0]] > weight[0]/10-0.15) & (prob[:,cell_type[0]] < weight[0]/10+0.15)
                        right = (prob[:,cell_type[1]] > weight[1]/10-0.15) & (prob[:,cell_type[1]] < weight[1]/10+0.15)
                        select_index = left & right
                        all_cell.extend([sample[select_index].cpu().numpy()])
                        sample_num += select_index.sum().item()
                    else:
                        logits = classifier(sample, torch.zeros((sample.shape[0]), device=sample.device))
                        prob = F.softmax(logits, dim=-1)
                        type = torch.argmax(prob, 1)
                        select_index = (type == cell_type[0])
                        all_cell.extend([sample[select_index].cpu().numpy()])
                        sample_num += select_index.sum().item()
                logger.log(f"created {sample_num} samples")
            else:
                all_cell.extend([sample.cpu().numpy() for sample in gathered_samples])
                sample_num = len(all_cell) * batch_size
                logger.log(f"created {len(all_cell) * batch_size} samples")

        arr = np.concatenate(all_cell, axis=0)
        dist.barrier()
        autoencoder = self.load_VAE(
            vae_path=vae_path,
            loss_ae=loss_ae,
            decoder_activation=decoder_activation,
        )
        cell_gen = autoencoder(torch.tensor(arr).cuda(),return_decoded=True).cpu().detach().numpy()

        adata=anndata.AnnData(cell_gen)
        adata.var.index=self.adata.var_names
        logger.log("sampling complete")
        return adata
            


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr

def save_model(mp_trainer, opt, step, model_path):
    if dist.get_rank() == 0:
        model_dir = model_path
        os.makedirs(model_dir,exist_ok=True)
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(model_dir, f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(model_dir, f"opt{step:06d}.pt"))
