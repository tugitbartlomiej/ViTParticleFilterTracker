import torch, os
def inspect(d, files):
    for f in files:
        p = os.path.join(d, f)
        try:
            ck = torch.load(p, map_location='cpu', weights_only=False)
            if isinstance(ck, dict):
                ep = ck.get('epoch', 'NO_EPOCH_KEY')
                bf = ck.get('best_fitness', None)
                ta = ck.get('train_args', {})
                ep_plan = ta.get('epochs','?') if isinstance(ta, dict) else '?'
                print(f'{f}: epoch={ep}  best_fitness={bf}  epochs_planned={ep_plan}')
            else:
                print(f'{f}: NOT_DICT type={type(ck)}')
        except Exception as e:
            print(f'{f}: ERROR {e}')
print('=== final_ep170_lr0001_weights (PRODUKCYJNE) ===')
inspect(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\YoloTreningSesnions\20KTreningFrom170epochStart\final_ep170_lr0001_weights', ['last.pt','best.pt','epoch270.pt','epoch260.pt','epoch170.pt'])
print('=== exp/weights (wczesny przebieg) ===')
inspect(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\YoloTreningSesnions\20KTreningFrom170epochStart\exp\weights', ['last.pt','best.pt','epoch190.pt','epoch180.pt'])
