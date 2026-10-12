"""CLI izolujące pamięć optymalizacji wejścia generatora."""
import argparse
from generowanie import generate
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('detector','generator','vendor','probe','directions','target','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--anchor',type=int,required=True);ap.add_argument('--device',default='cuda')
    ap.add_argument('--steps',type=int,default=40);ap.add_argument('--restarts',type=int,default=2)
    ap.add_argument('--expected-current',help='NPY [K] z zapisanym kodem wybranej kotwicy; kontrola zgodności przed optymalizacją')
    a=ap.parse_args()
    generate(a.detector,a.generator,a.vendor,(a.probe,a.anchor),a.directions,a.target,a.out,a.device,a.steps,a.restarts,
             expected_current_npy=a.expected_current)
