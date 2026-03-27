import os
import toml
import torch
from collections import OrderedDict
from tqdm import tqdm
import datetime
import torchaudio
from open_source.models.fastsde import SeldNetSubbandFast

# real audio 
from utils.record_audio import MyDataset   

def remove_module_prefix(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = k[len("module."):]
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict

if __name__ == '__main__':
    os.environ['PATH'] = '/sbin:' + os.environ.get('PATH', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # configs
    config = toml.load('./configs/real_audio/config.toml')
    checkpoint_path = 'PATH/dataset/FramRIRs/RIR_100room_100mic_50src_UltraFast/checkpoints/model_0400_loss_0.0178.tar'

    # resample
    resampler = torchaudio.transforms.Resample(orig_freq=config['test_gene_setting']['sample_rate'],
                                            new_freq=config['test_gene_setting']['target_sample_rate'])
    resampler.to(device)

    test_dataset = MyDataset(**config['FFT'], **config['test_gene_setting'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                **config['test_dataloader'])

    # model = SeldNetSubbandFast(
    #     features_set="all",
    #     n_subbands=8,  
    #     c_mid=24,
    #     n_blocks=3,
    #     fuse_c=64,
    #     use_gru=True,  
    #     gru_hidden=96,
    #     att_conf="Nothing",
    # ).to(device)

    # UltraFast
    model = SeldNetSubbandFast(
    features_set="all",
    n_subbands=6,   
    c_mid=16,
    n_blocks=2,
    fuse_c=48,
    use_gru=False,  
    att_conf="Nothing",).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_weights = checkpoint['model']
    model_weights = remove_module_prefix(model_weights)
    model.load_state_dict(model_weights, strict=True)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params',total_params)

    data_sample_rate = config['test_gene_setting']['sample_rate']
    target_sample_rate = config['test_gene_setting']['target_sample_rate']
    resample = False
    if data_sample_rate != target_sample_rate:
        resample = True
        resampler = torchaudio.transforms.Resample(orig_freq=data_sample_rate,
                                                        new_freq=target_sample_rate)
        resampler.to(device)
    test_bar = tqdm(test_dataloader, ncols=100)
    total_error = 0
    total_num = 0
    total_truth = []
    total_esti = []


    for step, data in enumerate(test_bar):
        input_audio = data[0].to(device)              # [b, wav length]
        target_dis = data[1].unsqueeze(-1).to(device) # [b, 1]
        
        # model
        output = model(input_audio, 0)                # [b, 1]  
        
        # data
        total_truth.append(target_dis.item())
        total_esti.append(output.item())
            
        # error
        error = torch.mean(torch.abs(output - target_dis))
        total_error += error.item()
        total_num += 1
        
        print('mean error', round(total_error/total_num, 3))
            
    mean_error = total_error/total_num


