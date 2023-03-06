# SimpleDiffusion

Train command:

'''
accelerate launch --multi_gpu --main_process_port 25640 --gpu_ids=0,1 train.py -c ./config/simple_diffusion.yaml --results_folder=./results
accelerate launch --multi_gpu --main_process_port 25641 --gpu_ids=2,3 train_dwt.py -c ./config/simple_diffusion_dwt.yaml --results_folder=./results_dwt

'''

Note:
dwt version generate bad quality sample.
