import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa import STFT
import pandas as pd
    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubbandSplitNoOverlap(nn.Module):
    # x: [B, Cin, T, F] -> [B, N, Cin, T, Fb]
    def __init__(self, n_subbands=8):
        super().__init__()
        self.n_subbands = n_subbands

    def forward(self, x):
        B, Cin, T, Freq = x.shape
        N = self.n_subbands
        Fb = math.ceil(Freq / N)
        need_F = Fb * N
        if need_F > Freq:
            x = F.pad(x, (0, need_F - Freq, 0, 0))
        # [B, Cin, T, N, Fb] -> [B, N, Cin, T, Fb]
        x = x.view(B, Cin, T, N, Fb).permute(0, 3, 1, 2, 4).contiguous()
        return x

class FastFilterBlock(nn.Module):
    """
    input:  [B*, C, T, Fb]
    output:  [B*, C, T, Fb]
    depthwise + pointwise
    """
    def __init__(self, ch):
        super().__init__()
        self.pw1 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.PReLU(),
        )
        # depthwise convs on (T,Fb) -> kernel=(time,freq)
        self.dw_a = nn.Conv2d(ch, ch, kernel_size=(1, 3), padding=(0, 1), groups=ch, bias=False)
        self.dw_b = nn.Conv2d(ch, ch, kernel_size=(3, 7), padding=(1, 3), groups=ch, bias=False)

        self.pw2 = nn.Sequential(
            nn.Conv2d(2 * ch, ch, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.PReLU(),
        )

    def forward(self, x):
        res = x
        h = self.pw1(x)
        a = self.dw_a(h)
        b = self.dw_b(h)
        h = torch.cat([a, b], dim=1)
        h = self.pw2(h)
        return h + res

class SharedSubbandEncoder(nn.Module):
    """
    input:  [B*N, Cin, T, Fb]
    output:  [B*N, Cmid, T, Fb]
    """
    def __init__(self, c_in, c_mid=24, n_blocks=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.PReLU(),
        )
        self.blocks = nn.Sequential(*[FastFilterBlock(c_mid) for _ in range(n_blocks)])

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        return x

class SeldNetSubbandFast(nn.Module):
    def __init__(
        self,
        features_set="all",     
        n_subbands=8,           
        c_mid=24,              
        n_blocks=3,             
        fuse_c=64,              
        use_gru=True,
        gru_hidden=96,          
        n_fft=512,
        hop_length=128,
        att_conf="Nothing",     
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.features_set = features_set
        self.att_conf = att_conf

        self.STFT = STFT(n_fft=self.n_fft, hop_length=self.hop_length)

        if features_set == "stft":
            c_in = 1
        elif features_set == "sincos":
            c_in = 2
        elif features_set == "all":
            c_in = 3
        else:
            raise ValueError

        if att_conf != "Nothing":
            raise NotImplementedError("For speed, set att_conf='Nothing' first.")

        self.split = SubbandSplitNoOverlap(n_subbands=n_subbands)
        self.enc = SharedSubbandEncoder(c_in=c_in, c_mid=c_mid, n_blocks=n_blocks)

        self.fuse = nn.Sequential(
            nn.Conv1d(n_subbands * c_mid, fuse_c, kernel_size=1, bias=False),
            nn.BatchNorm1d(fuse_c),
            nn.PReLU()
        )

        self.use_gru = use_gru
        if use_gru:
            self.gru = nn.GRU(
                input_size=fuse_c,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=False 
            )
            head_in = gru_hidden
        else:
            head_in = fuse_c

        self.head = nn.Sequential(
            nn.Linear(head_in, head_in),
            nn.ELU(),
            nn.Linear(head_in, 1)
        )

    @staticmethod
    def normalize_tensor(x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), unbiased=False, keepdim=True) + 1e-6
        return (x - mean) / std

    def extract_features(self, wav):
        x_real, x_im = self.STFT(wav)  # [B,1,T,F]
        magn = torch.sqrt(x_real**2 + x_im**2)
        magn = torch.log(magn**2 + 1e-7)

        cos = torch.cos(torch.angle(x_real + 1j * x_im))
        sin = torch.sin(torch.angle(x_real + 1j * x_im))

        magn = magn[..., :-1]
        cos = cos[..., :-1]
        sin = sin[..., :-1]

        if self.features_set == "stft":
            x = magn
        elif self.features_set == "sincos":
            x = torch.cat([cos, sin], dim=1)
        else:
            x = torch.cat([magn, cos, sin], dim=1)

        return self.normalize_tensor(x)  # [B,Cin,T,F]

    def forward(self, wav, info=0 ,return_seq=False):
        x = self.extract_features(wav)        # [B,Cin,T,F]
        B, Cin, T, Freq = x.shape

        x_sb = self.split(x)                 # [B,N,Cin,T,Fb]
        B, N, Cin, T, Fb = x_sb.shape

        u = x_sb.view(B*N, Cin, T, Fb)       # [B*N,Cin,T,Fb]
        u = self.enc(u)                      # [B*N,Cmid,T,Fb]
        Cmid = u.shape[1]
        u = u.view(B, N, Cmid, T, Fb)        # [B,N,Cmid,T,Fb]

        u_mean = u.mean(dim=-1)              # [B,N,Cmid,T]
        u_max = u.amax(dim=-1)               # [B,N,Cmid,T]
        z = 0.5 * (u_mean + u_max)           # [B,N,Cmid,T] 

        z = z.view(B, N*Cmid, T)
        z = self.fuse(z)                     # [B,fuse_c,T]
        z = z.permute(0, 2, 1).contiguous()  # [B,T,fuse_c]

        if self.use_gru:
            y, _ = self.gru(z)               # [B,T,H]
            if return_seq:
                return self.head(y).squeeze(-1)   # [B,T]
            y_pool = 0.5 * (y.mean(dim=1) + y.amax(dim=1))  # [B,H]
            return (self.head(y_pool),False) if info == 1 else  self.head(y_pool)       # [B,1]
        else:
            z_pool = 0.5 * (z.mean(dim=1) + z.amax(dim=1))  # [B,fuse_c]
            return (self.head(z_pool),False) if info == 1 else  self.head(z_pool)        # [B,1]

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
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

    model = SeldNetSubbandFast(
    features_set="all",
    n_subbands=6,   
    c_mid=16,
    n_blocks=2,
    fuse_c=48,
    use_gru=False,
    att_conf="Nothing",).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params',total_params)

    batch = 1
    x = torch.randn(batch, 3200).to(device)
    
    out = model(x, 1)

    from thop import profile
    import time

    macs, params = profile(model, inputs=(x,))
    print("MACs: ", macs)
    print("Params:", params)

    model.eval()
    input_data = torch.randn(1, 3200).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    num_runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()

    average_time = (end_time - start_time) / num_runs
    print(f"Average inference time: {average_time*1000:.2f} ms")