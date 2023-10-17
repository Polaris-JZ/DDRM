import world
import utils
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import dataloader
from os.path import join
from parse import parse_args
import torch.utils.data as data
import diffusion as gd
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register

# define dataset
args = parse_args()
dataset = dataloader.DiffData(path = args.data_path) 
train_loader = data.DataLoader(dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)

# define rec mdoel
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

# define diffusion reverse model
out_dims = eval(args.dims) + [args.recdim]
in_dims = out_dims[::-1]
# w_out_dims = eval(args.w_dims) + [args.recdim]
# w_in_dims = w_out_dims[::-1]
# user_reverse_model = register.DIFF_MODELS['transformer'](in_dims, out_dims, w_in_dims, w_out_dims, norm=args.norm)
# user_reverse_model = user_reverse_model.to(world.device)

# item_reverse_model = register.DIFF_MODELS['transformer'](in_dims, out_dims, w_in_dims, w_out_dims, norm=args.norm)
# item_reverse_model = item_reverse_model.to(world.device)

user_reverse_model = register.DIFF_MODELS['dnn'](in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm)
user_reverse_model = user_reverse_model.to(world.device)

item_reverse_model = register.DIFF_MODELS['dnn'](in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm)
item_reverse_model = item_reverse_model.to(world.device)

# define Gaussian Diffusion
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)
diffusion = gd.GaussianDiffusion(world.config, mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, world.device).to(world.device)

# define bpr
bpr = utils.BPRLoss(Recmodel, user_reverse_model, item_reverse_model, diffusion, world.config)

weight_file, user_weight_file, item_weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load('./pretrain_checkpoint/SGL_checkpoint',map_location=torch.device('cpu')))
        print(f"loaded model weights from ./pretrain_checkpoint/SGL_checkpoint")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# get config
config = world.config

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    print("not enable tensorflowboard")

try:
    best_recall = 0
    best_epoch = 0
    recall_list = []
    cnt = 0
    iter = 0
    for param in Recmodel.parameters():
        param.requires_grad = False
    for epoch in range(world.TRAIN_epochs):
        Recmodel.train()
        user_reverse_model.train()
        item_reverse_model.train()
        # print(user_reverse_model)
        bpr_: utils.BPRLoss = bpr
        train_loader.dataset.get_pair_bpr()
        aver_loss = 0.
        idx = 0
        for batch_users, batch_pos, batch_neg in train_loader:
            batch_users = batch_users.to(world.device)
            batch_pos = batch_pos.to(world.device)
            batch_neg = batch_neg.to(world.device)
            loss = bpr.call_bpr(batch_users, batch_pos, batch_neg, iter)
            aver_loss += loss
            idx += 1
            iter += 1

        aver_loss = aver_loss / idx
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] loss:{aver_loss}')
        
        if (epoch+1) % 5 == 0:
            results = Procedure.Test(dataset, Recmodel, user_reverse_model, item_reverse_model, diffusion, epoch, w, world.config['multicore'])
            Procedure.print_results(results)
            if results[1][0] > best_recall:
                best_epoch = epoch
                best_recall = results[1][0]
                best_v = results
                torch.save(Recmodel.state_dict(), weight_file)
                torch.save(user_reverse_model.state_dict(), user_weight_file)
                torch.save(item_reverse_model.state_dict(), item_weight_file)
            if epoch == 30:
                recall_list.append((epoch, results[1][0]))
            if epoch > 30:  # epoch20以后如果出现连续40个recall@10不涨，直接停止训练
                recall_list.append((epoch, results[1][0]))
                if results[1][0] < best_recall:
                    cnt += 1
                else:
                    cnt = 1
                if cnt >= 6:
                    break

    print("End train and valid. Best validation epoch is {:03d}.".format(best_epoch))
    Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))  
    user_reverse_model.load_state_dict(torch.load(user_weight_file,map_location=torch.device('cpu')))   
    item_reverse_model.load_state_dict(torch.load(item_weight_file,map_location=torch.device('cpu')))    
    best_results_valid = Procedure.Test_all(dataset, Recmodel, user_reverse_model, item_reverse_model, diffusion, epoch, w, world.config['multicore'], 0)
    best_results_test = Procedure.Test_all(dataset, Recmodel, user_reverse_model, item_reverse_model, diffusion, epoch, w, world.config['multicore'], 1)
    print("Validation:")
    Procedure.print_results_all(None, best_results_valid, None)
    print("Test:")
    Procedure.print_results_all(None, None, best_results_test)

finally:
    if world.tensorboard:
        w.close()