"""Zbiera ramiona; promień wybiera WYŁĄCZNIE po zamrożonej walidacji."""
from pathlib import Path
import json,argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
    radii=['0.25','0.5','1.0']
    selection={r:json.loads((a.results/f'r{r}/SELECTION.json').read_text()) for r in radii}
    selected=min(radii,key=lambda r:(selection[r]['val_loss'],float(r)))
    result={r:json.loads((a.results/f'r{r}/RESULT.json').read_text()) for r in radii}
    assert all(x['status']=='complete' and not x['smoke'] for x in result.values())
    report={'status':'complete','selected_radius':selected,'selection':'validation loss only, smaller radius on ties','results':result}
    (a.results/'SUMMARY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Optymalizacja 128 kanałów: wynik','',f'Promień wybrany wyłącznie na walidacji: {selected} SD.','',
        'Oracle oznacza osobną optymalizację obrazu z jego etykietą; nie jest testem generalizacji. Shared oznacza jedną korektę uczoną na train i zamrożoną po val.','',
        '| Promień | Metoda | AP50 test | Strata test | Confidence×IoU test |','|---|---|---:|---:|---:|']
    for r in radii:
        for m in ('baseline','shared','shared_random11','shared_random22','shared_random33','oracle','oracle_random11','oracle_random22','oracle_random33'):
            v=result[r]['methods'][m]['test'];lines.append(f"| {r} | {m} | {v['AP50']:.6f} | {v['loss']:.6f} | {v['q']:.6f} |")
    v=result[selected];ci=v['paired_film_CI95']['shared'];base=v['methods']['baseline']['test'];shared=v['methods']['shared']['test']
    passed=ci['loss'][1]<0 and shared['AP50']>base['AP50'] and shared['q']>base['q']
    lines+=['',f'Kryterium wstępnej użyteczności wspólnej korekty: {"spełnione" if passed else "niespełnione"}.',f'95% CI różnicy straty po filmach: {ci["loss"]}.',f'95% CI różnicy confidence×IoU po filmach: {ci["q"]}.','',
        'Wartości kanałów: TARGETS_ORACLE.npz (per obraz/token), SHARED.npz (korekta wspólna). Optymalizacja lokalna nie gwarantuje maksimum globalnego ani istnienia naturalnego obrazu z takim latentem. Jeden checkpoint, znane wcześniej filmy, etykieta jednej ramki narzędzia. Nie wykonano generowania obrazów ani aktualizacji wag ViT.']
    (a.results/'RAPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('selected',selected,'criterion',passed)
if __name__=='__main__':main()
