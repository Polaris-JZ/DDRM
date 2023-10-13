import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--batch_size', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=float,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=256,
                        help="the batch size of users for testing")
    parser.add_argument('--data_path', type=str, default='/storage/jjzhao/jujia_ws/diff/data/yelp_noisy',
                        help='the path to dataset')
    parser.add_argument('--dataset', type=str,default='yelp_noisy',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--data_type', type=str, default='time',
                        help='time or random')
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10, 20, 50, 100]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=50)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')

    parser.add_argument('--log_name', type=str, default='log', help='log name')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log', type=str)
    parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")

    # diff reverse params (DNN)
    parser.add_argument('--dims', type=str, default='[200,600]', help='the dims for the DNN')
    parser.add_argument('--act', type=str, default='tanh', help='the activate function for the DNN')
    parser.add_argument('--w_dims', type=str, default='[200,600]', help='the dims for the W DNN')
    parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
    parser.add_argument('--diff_lr', type=float,default=0.001, help="the learning rate")
    # diff params
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=2, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=5e-3, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.005, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=2, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')    

    parser.add_argument('--alpha', type=float, default=0.1, help='balance rec loss and reconstruct loss')

    # denoising 
    parser.add_argument('--drop_rate', 
        type = float,
        help = 'drop rate',
        default = 0.2)
    parser.add_argument('--num_gradual', 
        type = int, 
        default = 30000,
        help='how many epochs to linearly increase drop_rate')
    parser.add_argument('--exponent', 
        type = float, 
        default = 1, 
        help='exponent of the drop rate {0.5, 1, 2}')
    parser.add_argument('--beta', 
        type = float,
        help = 'drop rate',
        default = 0.2)
    return parser.parse_args()