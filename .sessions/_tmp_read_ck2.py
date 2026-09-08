import zipfile, os, pickletools, io, re

# Ultralytics .pt is a torch zip. We read data.pkl and disassemble to find 'epoch' int.
def find_epoch_via_pickle(path):
    try:
        with zipfile.ZipFile(path) as z:
            # find */data.pkl
            names = [n for n in z.namelist() if n.endswith('data.pkl')]
            if not names:
                return 'no data.pkl in zip; names=' + str(z.namelist()[:5])
            data = z.read(names[0])
        out = io.StringIO()
        try:
            pickletools.dis(io.BytesIO(data), out)
            text = out.getvalue()
        except Exception as e:
            text = ''
        # Look for SHORT_BINUNICODE 'epoch' followed soon by an int opcode
        results = {}
        for key in ['epoch','best_fitness','epochs','train_args','date','version']:
            idx = text.find(\"'\" + key + \"'\")
            if idx != -1:
                snippet = text[idx: idx+200].replace(chr(10),' | ')
                results[key] = snippet
        return results
    except zipfile.BadZipFile:
        return 'NOT A ZIP (old torch format)'
    except Exception as e:
        return 'ERR ' + repr(e)

for label, d, files in [
    ('PROD', r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\YoloTreningSesnions\20KTreningFrom170epochStart\final_ep170_lr0001_weights', ['last.pt','best.pt','epoch270.pt']),
    ('EXP', r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\YoloTreningSesnions\20KTreningFrom170epochStart\exp\weights', ['last.pt','epoch190.pt']),
]:
    print('==== ' + label + ' ====')
    for f in files:
        p = os.path.join(d,f)
        print('---', f)
        r = find_epoch_via_pickle(p)
        if isinstance(r, dict):
            for k,v in r.items():
                print('   ', k, '=>', v[:160])
        else:
            print('   ', r)
