import os
import errno
import numpy
import time
import subprocess
# import ipdb
import sys
import re

dataset = 'ml-1m'
log_path = './log/'
ckpt_path = './code/checkpoints/'
data_path = './data/ml-1m'
epoch_num = 1000
# noise_schedule = 'linear'



def pid_exists(pid):
	"""Check whether pid exists in the current process table.
	UNIX only.
	"""
	if pid < 0:
		return False
	if pid == 0:
		# According to "man 2 kill" PID 0 refers to every process
		# in the process group of the calling process.
		# On certain systems 0 is a valid PID but we have no way
		# to know that in a portable fashion.
		raise ValueError('invalid PID 0')
	try:
		os.kill(pid, 0)
	except OSError as err:
		if err.errno == errno.ESRCH:
			# ESRCH == No such process
			return False
		elif err.errno == errno.EPERM:
			# EPERM clearly means there's a process to deny access to
			return True
		else:
			# According to "man 2 kill" possible error values are
			# (EINVAL, EPERM, ESRCH)
			raise
	else:
		return True


# 1_ml-1m_clean_lr0.001_wd0_bs400_dims[200,600]_emb10_mse_x0_small_steps5_scale0.005_min0.001_max0.01_sample0_levellarge_reweight1_log.txt
def check_log_finish(lr, dlr, layer, alpha, steps, sample, min_, max_, dims, scale, act, beta,log, log_num):
	
	file_name = f'{log_num}_load_{dataset}_lr{lr}_difflr{dlr}_layer{layer}_alpha{alpha}_steps{steps}_sample{sample}_min{min_}_max{max_}_dims{dims}_scale{scale}_act{act}_beta{beta}_{log}.txt'
	file_path = log_path+file_name
	
	if os.path.exists(file_path):
		with open(file_path, "r") as file:
			lines = file.readlines()
			if lines:  # 确保文件不为空
				last_line = lines[-1].strip()
				match = re.search(r"Recall: (\d+\.\d+)", last_line)
				if match:
					return True
				else:
					return False 
	else:
		file = f'{log_num}_{dataset}_lr{lr}_difflr{dlr}_layer{layer}_alpha{alpha}_steps{steps}_sample{sample}_min{min_}_max{max_}_dims{dims}_scale{scale}_act{act}_beta{beta}_{log}.txt'
		file_path = log_path+file
		if os.path.exists(file_path):
			with open(file_path, "r") as file:
				lines = file.readlines()
				if lines:  # 确保文件不为空
					last_line = lines[-1].strip()
					match = re.search(r"Recall: (\d+\.\d+)", last_line)
					if match:
						return True
					else:
						return False 
		else:
			return False

def check_ckpt(lr, dlr, layer, alpha, steps, sample, min_, max_, dims, scale, act, beta, log, log_num):
	
	file_name = f'lgn-ml-1m_noisy-{log}.pth.tar'
	file_path = ckpt_path+file_name
	print(file_path)
	if os.path.exists(file_path):
		return True
	else:
		return False
	
# nohup python -u main.py --model_name=$1 --data_path=$2 --batch_size=$3 --l_r=$4 --reg_weight=$5 --num_neg=$6 --has_v=$7 --has_t=$8 --lr_lambda=$9 --num_sample=$10 --temp_value=$11 --dim_E=$12 --gpu=$14
def run_program(lr, dlr, layer, alpha, steps, sample, min_, max_, dims, scale, act, beta, log, log_num, gpu):
	
	log_name = f'{log_num}_{dataset}_lr{lr}_difflr{dlr}_layer{layer}_alpha{alpha}_steps{steps}_sample{sample}_min{min_}_max{max_}_dims{dims}_scale{scale}_act{act}_beta{beta}_{log}.txt'
	# cmd = f'python -u main.py --model_name={model_name} --data_path={dataset} --batch_size=256 --l_r={lr} --reg_weight={reg} --num_neg={num_neg} --has_v=True --has_t=True --lr_lambda={lam} --num_sample={rou} --temp_value={temp} --dim_E={dim_E} --gpu={gpu} > {log_path}{log_name}'
	cmd = f'python -u main.py --load=1 --act={act} --beta={beta} --dataset={dataset} --data_path={data_path}  --lr={lr} --diff_lr={dlr} --layer={layer} --alpha={alpha} --steps={steps} --epochs=1000 --sampling_steps={sample} --noise_scale={scale} --noise_min={min_} --noise_max={max_}  --sampling_steps={sample} --dims={dims} --log_name={log} --gpu={gpu} > {log_path}{log_name}'
#     res = os.popen(cmd).readlines()
	p = subprocess.Popen(cmd, shell=True)
	return p

def load_program(lr, dlr, layer, alpha, steps, sample, min_, max_, dims, scale, act, beta, log, log_num, gpu):
	log_name = f'{log_num}_load_{dataset}_lr{lr}_difflr{dlr}_layer{layer}_alpha{alpha}_steps{steps}_sample{sample}_min{min_}_max{max_}_dims{dims}_act{act}_scale{scale}_beta{beta}_{log}.txt'
	# cmd = f'python -u main.py --model_name={model_name} --data_path={dataset} --batch_size=256 --l_r={lr} --reg_weight={reg} --num_neg={num_neg} --has_v=True --has_t=True --lr_lambda={lam} --num_sample={rou} --temp_value={temp} --dim_E={dim_E} --gpu={gpu} > {log_path}{log_name}'
	cmd = f'python -u main.py --load=1 --act={act} --beta={beta} --dataset={dataset} --data_path={data_path}  --lr={lr} --diff_lr={dlr} --layer={layer} --alpha={alpha} --steps={steps} --epochs=1000 --sampling_steps={sample} --noise_scale={scale} --noise_min={min_} --noise_max={max_}  --sampling_steps={sample} --dims={dims} --log_name={log} --gpu={gpu} > {log_path}{log_name}'
#     res = os.popen(cmd).readlines()
	p = subprocess.Popen(cmd, shell=True)
	return p

def check_program_done(pid_list):
	
	flag = False
	while(flag==False):
		print("Start : %s" % time.ctime())
		time.sleep(1*60)
		print("End : %s" % time.ctime())
		finished_pid = 0
		for pid in pid_list:
			print(pid.poll())
			if pid.poll() is not None:
				finished_pid += 1
		if finished_pid == len(pid_list):
			flag = True


# DNN
# dims_list = ['[200,600]', '[300,600]', '[300,1000]', '[200,1000]']
# '[200]', '[300]', '[600]', '[1000]'
dims_list = ['[200,1100]']
act_list = ['relu']
# act_list = ['relu', 'sigmoid', 'tanh']

# diff
# steps_list = [10,30,50,70,90]
steps_list = [50]

# sample_steps_list = [1, 1.5, 1.25]
sample_steps_list = [1.25]

# min_list = [1e-4, 1e-3, 1e-2]
min_list = [1e-4]

# max_list = [1e-3, 1e-2, 1e-1]
max_list = [1e-3]

# scale_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
scale_list = [1e-2]

lr_list = [1e-5]
# diff_lr_list = [1e-3, 1e-4, 1e-5]
# lr_list = [1e-5]
diff_lr_list = [1e-3]

gpu_list = ['0']

# GCN
# layer_list = [2,3,4]
layer_list = [2]

# alpha
# alpha_list = [0.1, 0.2, 0.3, 0.4]
alpha_list = [0.1]

# denoising
# drop_rate_list = [0.05, 0.1, 0.2, 0.3, 0.4]
# num_gradual_list = [1000,5000,10000,20000]
# beta_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ,1.0]
beta_list = [0]

wd = 0
log = 'log'


log_num = 0
pid_list = []
cnt = 0
for beta in beta_list:
	for act in act_list:
		for alpha in alpha_list:
			for layer in layer_list:
				for steps in steps_list:
					for lr in lr_list:
						for dlr in diff_lr_list:
								for sample in sample_steps_list:
									for min_ in min_list:
										for max_ in max_list:
											if min_>=max_:
												continue
											for dims in dims_list:
												for i, scale in enumerate(scale_list):
													log = f'log_{log_num}'
													print(f'run ==>  {dataset}_lr{lr}_difflr{dlr}_layer{layer}_alpha{alpha}_steps{steps}_sample{int(steps*sample)}_min{min_}_max{max_}_dims{dims}_scale{scale}_{log}')
													if check_log_finish(lr, dlr, layer, alpha, steps, int(steps*sample), min_, max_, dims, scale, act, beta, log, log_num):
														log_num += 1
														continue
													if check_ckpt(lr, dlr, layer, alpha, steps, int(steps*sample), min_, max_, dims, scale, act, beta, log, log_num):
														print(f'exist check ckpt{log}')
														pid = load_program(lr, dlr, layer, alpha, steps, int(steps*sample), min_, max_, dims, scale, act, beta, log, log_num, gpu=gpu_list[cnt%1])
													else:
														pid = run_program(lr, dlr, layer, alpha, steps, int(steps*sample), min_, max_, dims, scale, act, beta, log, log_num, gpu=gpu_list[cnt%1])
													
													pid_list.append(pid)
													log_num += 1
													cnt += 1
													if cnt == 1:
														cnt = 0
														check_program_done(pid_list)
														pid_list = []
