"""Timing only on existing34-image cache; NOT actual1500-data training."""
from pathlib import Path
import json,time
import numpy as np
import torch
from model import ImageBatchNNDMD,optimizer_step


def main():
    root=Path(__file__).parent
    source=root.parent/'maly_vit_pykoopman_30_obrazow_128/LATENT_768.npy'
    started=time.perf_counter();cache=np.load(source,mmap_mode='r')
    window=np.array(cache[:11,:32],dtype=np.float32,copy=True)
    read_seconds=time.perf_counter()-started
    scale=float(np.std(window[0].reshape(-1,768),axis=0).max())
    results=[]
    for micro in [512,1024]:
        model=ImageBatchNNDMD(device='cuda');model.scale=scale
        torch.cuda.reset_peak_memory_stats();timings=[]
        for j in range(3):
            torch.cuda.synchronize();start=time.perf_counter()
            stats=optimizer_step(model,window,micro_tokens=micro)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            print(f'BENCH_ONLY micro{micro}, logical32IMAGES step{j+1}: {elapsed:.3f}s',flush=True)
            if j:timings.append(elapsed)
        results.append({'micro_tokens':micro,'seconds_logical32images':timings,
                        'mean_seconds':float(np.mean(timings)),
                        'peak_tensor_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'prediction_RMSE_H10':float(np.sqrt(stats['prediction_mse_raw_by_horizon'][-1]))})
        del model;torch.cuda.empty_cache()
    output={'benchmark_only':True,'not_1500training':True,'data':'old34cache first32images, checkpoints0..10',
            'image_batch':32,'particles_per_image':600,'dimensions':[768,768,768],
            'encoder_decoder':'both swish768hidden768; actual PyKoopman neural core',
            'H':10,'normalization_scalar':scale,'RAM_window_bytes':window.nbytes,
            'old_cache_read_seconds':read_seconds,'GPU':torch.cuda.get_device_name(0),'results':results}
    (root/'BENCHMARK_NN.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
