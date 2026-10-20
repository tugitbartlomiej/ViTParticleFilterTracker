"""Atlas from frozen measurements. No training and no parameter changes."""
from pathlib import Path
import json, hashlib, textwrap, html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "Dane", ROOT / "Wykresy"
OUT.mkdir(exist_ok=True)
S = json.loads((DATA / "SUMMARY.json").read_text(encoding="utf-8"))
CFG = json.loads((DATA / "config.json").read_text(encoding="utf-8"))
DONE = json.loads((DATA / "DONE.json").read_text())
for p in DATA.iterdir():
    key = "results/" + p.name
    if key in DONE["files"]:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == DONE["files"][key], p
A = pd.read_csv(DATA / "ACTIVITY.csv")
T = pd.read_csv(DATA / "ANGLE_TRAJECTORY.csv")
F = pd.read_csv(DATA / "ANGLE_BY_FILM.csv")
D = pd.read_csv(DATA / "DETECTION.csv")
TR = pd.read_csv(DATA / "TRANSITIONS.csv")
H = np.load(DATA / "ANGLE_HISTOGRAMS.npz")
EX = np.load(DATA / "DETAILS.npz")
META = json.loads((DATA / "DETAILS_METADATA.json").read_text())
STEPS = np.array(S["steps"])
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC3377", "#7A55A3", "#86712D"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": .17,
    "legend.fontsize": 8.5, "lines.linewidth": 2, "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none"})
INDEX = []
BOOT = np.random.default_rng(20261020).integers(25, size=(2000, 25))
ATLAS = PdfPages(OUT / "ATLAS_WYKRESOW.pdf")
BASE = "50 klatek TEST / 25 filmów | 600 tokenów | seed3047 | 55 000 kroków = 50,29 epoki"
COHORT = "Stałe pary: wiązki >5% na początku i końcu, niezerowe w 11 checkpointach; ten sam obraz i token."

def canvas(title, subtitle="", rows=2, cols=3):
    fig, ax = plt.subplots(rows, cols, figsize=(15, 9.4), squeeze=False)
    fig.subplots_adjust(left=.075, right=.96, bottom=.15, top=.82, wspace=.32, hspace=.50)
    fig.suptitle(title, x=.075, y=.965, ha="left", fontsize=20, fontweight="bold", color="#18314B")
    fig.text(.075, .906, "\n".join(textwrap.wrap(subtitle, 152)), fontsize=10.3, color="#44556A", va="top")
    return fig, ax.ravel()

def timeaxis(ax, ylabel="", ylim=None):
    ax.set_xscale("symlog", linthresh=300)
    ax.set_xticks([0, 300, 1000, 3000, 10000, 55000])
    ax.set_xticklabels(["0", "300", "1k", "3k", "10k", "55k"])
    ax.set_xlabel("Krok treningu (oś symlog)")
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlim(-45, 65000)

def arr(df, key, block=None, cohort=None):
    v = df
    if block is not None: v = v[v.block == block]
    if cohort is not None: v = v[v.cohort == cohort]
    return v.groupby(["step", "film"])[key].mean().unstack("film").reindex(STEPS).to_numpy()

def curve(ax, df, key, block, color=None, label=None, cohort=None, band=False):
    x = arr(df, key, block, cohort)
    y = x.mean(1)
    c = color or COLORS[block-1]
    ax.plot(STEPS, y, "o-", ms=3.5, color=c, label=label)
    if band:
        lo, hi = np.quantile(x[:, BOOT].mean(2), [.025, .975], axis=1)
        ax.fill_between(STEPS, lo, hi, color=c, alpha=.16, linewidth=0)
    return y

def save(fig, slug, title, note, group):
    n = len(INDEX)+1
    stem = f"{n:02d}_{slug}"
    fig.text(.075, .075, "\n".join(textwrap.wrap(note, 168)), fontsize=9, color="#263D52", va="top")
    fig.text(.075, .018, BASE, fontsize=8, color="#68788A")
    fig.text(.96, .018, f"{n:02d}", ha="right", fontsize=8, color="#68788A")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=190)
    ATLAS.savefig(fig)
    plt.close(fig)
    INDEX.append(dict(number=n, stem=stem, title=title, note=note, group=group))
    (OUT / "INDEX.json").write_text(json.dumps(INDEX, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PLOT", n, slug, flush=True)

# 1. Immediately useful overview.
fig, ax = canvas("Co zmienia trening w pojedynczych wiązkach?", BASE, 2, 2)
for b in range(1, 7):
    for a, df, key, ylabel, cohort in [
        (ax[0], A, "active_rel_0.05", "Wiązki >5% maksimum / token", None),
        (ax[1], A, "effective", "Efektywna liczba / token", None),
        (ax[2], F, "mean_angle", "Średni kąt [°]", "persistent"),
        (ax[3], F, "mean_distance_from90", "Średnie |kąt - 90°| [°]", "persistent")]:
        curve(a, df, key, b, label=f"Blok {b}", cohort=cohort)
        timeaxis(a, ylabel)
ax[0].legend(ncol=3)
save(fig, "podsumowanie", "Podsumowanie treningu",
     "Mniej wiązek dominuje, a rozkład kątów się poszerza. Spadek liczby ponad progiem względnym nie oznacza wyzerowania. " + COHORT, "Przegląd")

# Detailed trajectories, shared units and axes across all blocks.
specs = [
 ("aktywnosc_wzgledna", "Ile wiązek przekracza próg względny?",
  [("active_rel_0.01", "1% maksimum", "#A5BED0"), ("active_rel_0.05", "5% maksimum", "#0072B2"), ("active_rel_0.1", "10% maksimum", "#D55E00")],
  "Liczba wiązek / token", (0,768), "Próg jest liczony osobno dla każdego tokena. Silniejsza najsilniejsza wiązka podnosi próg dla pozostałych."),
 ("aktywnosc_absolutna", "Czy wiązki naprawdę tracą siłę?",
  [("active_abs_0.01", "Siła >0,01", "#A5BED0"), ("active_abs_0.05", "Siła >0,05", "#0072B2"), ("active_abs_0.1", "Siła >0,1", "#D55E00")],
  "Liczba wiązek / token", (0,768), "Progi absolutne są stałe w całym treningu. Wynik pokazuje, dlaczego utrata znaczenia względnego nie jest równoważna wygaszeniu."),
 ("efektywna_liczba", "Koncentracja siły bez wybierania progu",
  [("effective", "Efektywna liczba", None)], "Efektywna liczba wiązek / token", (0,400),
  "N_eff = (suma sił)^2 / suma kwadratów sił. N równych wiązek daje N_eff=N. Wstęga: 95% bootstrap 25 filmów, dla jednego modelu."),
 ("kwantyle_sily", "Jak zmieniają się słabe, typowe i silne wiązki?",
  [("q10", "Kwantyl 10%", "#A5BED0"), ("median", "Mediana", "#0072B2"), ("q90", "Kwantyl 90%", "#D55E00")],
  "Siła wkładu (jednostki modelu)", None,
  "Kwantyle liczone wśród 768 wiązek osobno dla każdego tokena, następnie uśrednione. To nie trajektorie trzech konkretnych neuronów."),
 ("suma_i_wypadkowa", "Suma sił a długość wypadkowego wkładu",
  [("total", "Suma norm wiązek", "#0072B2"), ("signal", "Norma sumy wektorów", "#D55E00")],
  "Norma (jednostki modelu)", None,
  "Suma norm nie uwzględnia kierunków. Norma sumy uwzględnia kierunki i znaki; bias fc2 jest wyłączony z obu miar."),
 ("zera", "Ile wkładów jest zerowych?",
  [("exactzero", "Aktywacja dokładnie 0", "#0072B2"), ("nearzero", "Siła ≤ 1e-8", "#D55E00")],
  "Liczba wiązek / token", None,
  "Zera dotyczą wyników obliczeń zmiennoprzecinkowych. Liczba zer jest znacznie mniejsza niż liczba wiązek, które spadły poniżej względnego progu 5%."),
]
for slug, title, metrics, ylabel, ylim, note in specs:
    fig, axes = canvas(title, "Każdy panel: osobny blok; wszystkie 768 wkładów w każdym tokenie.")
    for b, a in enumerate(axes, 1):
        a.set_title(f"Blok {b}")
        for key, label, color in metrics:
            curve(a, A, key, b, color, label, band=(key=="effective"))
        timeaxis(a, ylabel, ylim)
        a.legend()
    save(fig, slug, title, note, "Siła i aktywność")

# Endpoint membership: true transitions, rather than a difference of counts.
fig, axes = canvas("Które wiązki pozostają, tracą próg lub go przekraczają?", "Krok 0 -> 55 000; próg 5% maksimum dla danego tokena.", 1, 2)
x = np.arange(1,7)
groups = [("persist","Ponad progiem na obu końcach","#009E73"),("lost","Ponad progiem tylko na początku","#D55E00"),
          ("new","Ponad progiem tylko na końcu","#0072B2"),("inactive","Poniżej progu na obu końcach","#B8C0C9")]
bottom=np.zeros(6)
for key,label,c in groups:
    y=np.array([s["endpoint_transitions"][key]["mean_beams_per_token"] for s in S["blocks"]])
    axes[0].bar(x,y,bottom=bottom,color=c,label=label)
    bottom+=y
axes[0].set(xticks=x,xlabel="Blok",ylabel="Średnia liczba wiązek / token",ylim=(0,768))
axes[0].legend(loc="lower left",fontsize=8)
for j,(key,label,c) in enumerate([("fraction_absolute_decrease","Siła spadła","#0072B2"),("fraction_decrease10x","Siła spadła co najmniej 10×","#D55E00")]):
    y=[100*s["endpoint_transitions"]["lost"][key] for s in S["blocks"]]
    axes[1].bar(x+(j-.5)*.34,y,width=.34,color=c,label=label)
    for xx,yy in zip(x+(j-.5)*.34,y): axes[1].text(xx,yy+1,f"{yy:.1f}",ha="center",fontsize=8)
axes[1].set(xticks=x,xlabel="Blok",ylabel="% wśród wiązek tracących próg",ylim=(0,100))
axes[1].legend()
save(fig,"przejscia","Przejścia przez próg i rzeczywiste osłabienie",
     "Próg na obu końcach nie oznacza aktywności w każdym kroku pośrednim. W prawym panelu spadek co najmniej 10× jest podzbiorem wszystkich spadków.", "Siła i aktywność")

for key,title,ylabel,ylim in [
 ("mean_angle","Średni kąt: te same pary przez cały trening","Kąt [°]",(86.7,90.3)),
 ("mean_distance_from90","Kąty oddalają się od 90° w obie strony","Średnie |kąt - 90°| [°]",(2.5,10))]:
    fig,axes=canvas(title,COHORT)
    for b,a in enumerate(axes,1):
        curve(a,F,key,b,cohort="persistent",band=True)
        timeaxis(a,ylabel,ylim);a.set_title(f"Blok {b}")
    save(fig,key,title,"Wstęgi: 95% bootstrap 25 filmów; jeden seed. Punkty to zapisane checkpointy, linie tylko je łączą. Węższa oś Y ujawnia niewielkie zmiany średniej.", "Kąty")

fig,axes=canvas("Rozkład kątów na początku i w najlepszym checkpoincie",COHORT)
for b,a in enumerate(axes,1):
    for step,c,lab in [(0,"#98A4B0","Krok 0"),(55000,COLORS[b-1],"Krok 55 000")]:
        h=H[f"b{b}_k{step}_persistent"]
        assert abs(h.sum()-1)<1e-8
        a.plot(np.arange(180)+.5,100*h,color=c,label=lab)
    a.axvline(90,c="#666",ls=":",lw=1);a.set(xlim=(35,145),xlabel="Kąt [°]",ylabel="% w przedziale 1°",title=f"Blok {b}");a.legend()
save(fig,"rozklad_katow","Rozkłady kątów 0 i 55k",
     "Główna część rozkładu jest powiększona do 35-145°. Ogony poza kadrem pozostają w danych i na wykresie rzadkich kątów; histogram nie został renormalizowany po przycięciu.", "Kąty")

fig,axes=canvas("Rozkład kątów w 11 checkpointach",COHORT)
for b,a in enumerate(axes,1):
    mat=np.array([H[f"b{b}_k{s}_persistent"] for s in STEPS])*100
    im=a.imshow(mat[:,40:141],aspect="auto",origin="lower",extent=(40,141,-.5,10.5),cmap="magma",norm=LogNorm(.001,15))
    a.set(title=f"Blok {b}",xlabel="Kąt [°]",yticks=range(11),yticklabels=[f"{s:,}".replace(","," ") for s in STEPS],ylabel="Zapisany krok")
    a.grid(False)
fig.colorbar(im,ax=axes.tolist(),shrink=.7,pad=.02,label="% par w przedziale 1° (kolor log)")
save(fig,"mapy_katow","Mapy rozkładów kątów",
     "Wiersze mają równą wysokość, ale odstępy między krokami nie są równe. Wspólna skala kolorów. Zera są puste; dodatnie wartości poniżej 0,001% mają najciemniejszy kolor.", "Kąty")

fig,axes=canvas("O ile zmienił się kąt konkretnej pary?", "Zmiana = kąt w kroku 55 000 minus kąt w kroku 0; stałe pary obraz-token-neurony.")
for b,a in enumerate(axes,1):
    h=H[f"b{b}_delta_persistent"]
    assert abs(h.sum()-1)<1e-8
    a.plot(np.arange(360)-179.5,h*100,color=COLORS[b-1])
    a.axvline(0,c="#333",ls=":",lw=1)
    a.set(title=f"Blok {b}",xlim=(-70,70),xlabel="Zmiana kąta [°]",ylabel="% w przedziale 1°")
save(fig,"rozklad_zmian_katow","Rozkład zmian kąta tych samych par",
     "Ujemna zmiana: kierunki się zbliżają. Dodatnia: oddalają. Widok -70° do +70°; dane zawierają pełne -180° do +180° bez renormalizacji.", "Kąty")

fig,axes=canvas("Ile par zbliża się, a ile oddala?", "Stałe pary obraz-token-neurony. Rozróżnienie zmian >1° i >5°.",1,2)
fig.subplots_adjust(bottom=.27)
for a,threshold in zip(axes,[1,5]):
    btm=np.zeros(6)
    down=np.array([s["angles"]["persistent"][f"decreasing_gt{threshold}_fraction"] for s in S["blocks"]])*100
    up=np.array([s["angles"]["persistent"][f"increasing_gt{threshold}_fraction"] for s in S["blocks"]])*100
    for y,c,lab in [(down,"#0072B2",f"Spadek >{threshold}°"),(100-down-up,"#DCE2E8",f"Zmiana ≤{threshold}°"),(up,"#D55E00",f"Wzrost >{threshold}°")]:
        a.bar(x,y,bottom=btm,color=c,label=lab)
        for xx,yy,bb in zip(x,y,btm):a.text(xx,bb+yy/2,f"{yy:.1f}%",ha="center",va="center",fontsize=9)
        btm+=y
    a.set(xticks=x,xlabel="Blok",ylabel="% par",ylim=(0,100));a.legend(loc="upper center",bbox_to_anchor=(.5,-.16),ncol=3)
save(fig,"kierunek_zmian","Odsetki wzrostów i spadków kąta",
     "Oba kierunki zmian są częste. Niewielki spadek średniej nie oznacza, że wszystkie pary się zbliżają. Procenty mają równe wagi 25 filmów.", "Kąty")

fig,axes=canvas("Czy wynik zależy od progu aktywności?", "Dla każdego progu osobno: stała grupa wiązek ponad progiem na obu końcach.",1,2)
for b,record in enumerate(S["blocks"],1):
    primary=record["angles"]["persistent"]
    vals={.05:primary["paired_delta"]["mean"]}
    vals.update({r["threshold"]:r["mean_delta"] for r in record["angle_sensitivity"]})
    axes[0].plot([1,5,10],[vals[k] for k in [.01,.05,.1]],"o-",color=COLORS[b-1],label=f"Blok {b}")
    stats=primary["paired_delta"];m=stats["mean"];lo,hi=stats["ci95"]
    axes[1].errorbar(m,b,xerr=[[m-lo],[hi-m]],fmt="o",color=COLORS[b-1],capsize=4)
axes[0].set(xticks=[1,5,10],xlabel="Próg względny [% maksimum]",ylabel="Średnia zmiana kąta [°]");axes[0].axhline(0,c="#777",lw=1);axes[0].legend(ncol=2)
axes[1].axvline(0,c="#777",lw=1);axes[1].set(yticks=x,ylabel="Blok",xlabel="Zmiana kąta [°]; 95% bootstrap filmów")
save(fig,"progi_i_niepewnosc","Czułość na próg i niepewność",
     "Po lewej zmiana progu zmienia też skład grupy. Po prawej próg 5%; przedziały opisują zróżnicowanie 25 filmów przy jednym seedzie, nie powtarzalność między treningami.", "Kontrole")

fig,axes=canvas("Kąty wag, aktywacji i kontrola stałego znaku",COHORT)
for b,a in enumerate(axes,1):
    for key,lab,c in [("mean_angle","Rzeczywiste wkłady",COLORS[b-1]),("mean_weight_angle","Same kierunki wag","#A1ABB5"),("mean_angle_fixed_initial_sign","Wagi + znaki z kroku 0","#222222")]:
        curve(a,F,key,b,c,lab,"persistent")
    timeaxis(a,"Kąt [°]",(86,90.5));a.set_title(f"Blok {b}");a.legend(fontsize=8)
save(fig,"wagi_i_znaki","Rola wag i znaku aktywacji",
     "Ujemna aktywacja GELU odwraca wkład neuronu. Kąty wkładów zależą od wag i znaków; kontrola nie jest osobnym treningiem ani dowodem przyczynowym.", "Kontrole")

fig,axes=canvas("Jak rzadkie są duże kąty między znaczącymi wiązkami?",COHORT)
for b,a in enumerate(axes,1):
    rr=T[(T.block==b)&(T.cohort=="persistent")].set_index("step").reindex(STEPS)
    for key,lab,c in [("fraction_ge_120","≥120°","#0072B2"),("fraction_ge_135","≥135°","#009E73"),("fraction_ge_150","≥150°","#D55E00")]:
        a.plot(STEPS,100*rr[key],"o-",ms=3,color=c,label=lab)
    timeaxis(a,"% par");a.set_yscale("symlog",linthresh=.0001);a.set_title(f"Blok {b}");a.legend()
save(fig,"rzadkie_katy","Rzadkie kąty 120°, 135° i 150°",
     "Mianownik: obserwacje par dla tego samego tokena w stałej grupie. To nie liczba unikalnych par neuronów w modelu. Oś Y symlog pozwala pokazać również dokładne zera.", "Kąty")

fig,axes=canvas("Stała grupa i grupa zmieniająca się z treningiem", "Porównanie trzech definicji zbioru par.")
for b,a in enumerate(axes,1):
    for cohort,lab,c in [("persistent","Próg na obu końcach",COLORS[b-1]),("final_active","Próg tylko na końcu","#222222"),("current_active","Próg w bieżącym kroku","#9AA6B2")]:
        curve(a,F,"mean_angle",b,c,lab,cohort)
    timeaxis(a,"Średni kąt [°]");a.set_title(f"Blok {b}");a.legend(fontsize=8)
save(fig,"porownanie_grup","Porównanie definicji grup par",
     "Dwie pierwsze grupy zachowują tożsamości par między checkpointami. Trzecia zmienia skład, więc jej krzywa miesza zmianę kierunków i zmianę uczestniczących wiązek.", "Kontrole")

fig,axes=canvas("IoU i score na tych samych 50 klatkach", "Wybrany checkpoint 55k pochodzi z osobnego VAL. Tutaj pokazany jest TEST.",1,2)
for a,key,c in zip(axes,["iou","score"],["#0072B2","#D55E00"]):
    y=curve(a,D,key,None,c,key.upper(),band=True)
    timeaxis(a,key.upper(),(0,1));a.set_title("Dopasowanie ramki: IoU" if key=="iou" else "Pewność modelu: score")
    a.text(.97,.10,f"55k: {y[-1]:.4f}",transform=a.transAxes,ha="right",fontweight="bold")
save(fig,"iou_score","IoU i score w treningu",
     "IoU dotyczy ramki tokena z najwyższym logitem i najlepiej pasującego GT. Score to sigmoid tego logitu. Wstęgi: bootstrap 25 filmów; score nie jest skalibrowanym prawdopodobieństwem poprawności.", "Detekcja")

fig,axes=canvas("Czy tendencja występuje w różnych filmach?", "Każdy wiersz to jeden z 25 filmów; średnia z dwóch klatek.",1,2)
films=sorted(A.film.unique())
for a,key,title in [(axes[0],"mean_angle","Zmiana średniego kąta [°]"),(axes[1],"mean_distance_from90","Zmiana odległości od 90° [°]")]:
    mat=[]
    for b in range(1,7):
        q=F[(F.block==b)&(F.cohort=="persistent")].pivot(index="film",columns="step",values=key).reindex(films)
        mat.append((q[55000]-q[0]).to_numpy())
    mat=np.array(mat).T
    limit=abs(mat).max();im=a.imshow(mat,aspect="auto",cmap="RdBu_r",vmin=-limit,vmax=limit)
    a.set(xticks=range(6),xticklabels=range(1,7),xlabel="Blok",yticks=range(25),yticklabels=films,title=title)
    a.tick_params(axis="y",labelsize=7);a.grid(False);fig.colorbar(im,ax=a,pad=.03)
save(fig,"filmy","Zmiany w 25 filmach",
     "Równe wagi filmów. Każdy pomiar dotyczy tego samego wytrenowanego modelu; obrazy i tokeny nie są niezależnymi treningami.", "Kontrole")

fig,axes=canvas("Rzeczywiste siły wszystkich wiązek: początek i koniec", "Każdy blok: 50 klatek × 600 tokenów × 768 pojedynczych wkładów.")
edges=EX["strength_log10_edges"]
for b,a in enumerate(axes,1):
    for tag,c,label in [("initial","#98A4B0","Krok 0"),("best",COLORS[b-1],"Krok 55 000")]:
        a.stairs(EX[f"b{b}_strength_hist_{tag}"]*100,10**edges,color=c,label=label)
    a.set_xscale("log");a.set(xlim=(1e-8,100),xlabel="Siła pojedynczej wiązki",ylabel="% w przedziale log10",title=f"Blok {b}");a.legend()
save(fig,"rozklad_sil","Rozkłady bezwzględnej siły",
     "Zera nie mają położenia na osi logarytmicznej i są pokazane na osobnym wykresie. Odsetki odnoszą się do wszystkich wkładów; nie renormalizowano po odrzuceniu zer ani po powiększeniu osi.", "Siła i aktywność")

fig,axes=canvas("Zmiany między kolejnymi checkpointami", "Liczba wiązek przekraczających próg 5% w obie strony.")
for b,a in enumerate(axes,1):
    for group,c,label in [("lost","#D55E00","Spadły poniżej progu"),("new","#0072B2","Weszły ponad próg")]:
        r=TR[(TR.block==b)&(TR.group==group)].groupby("to_step")["count"].mean().reindex(STEPS[1:])/600
        a.plot(STEPS[1:],r,"o-",ms=3,color=c,label=label)
    timeaxis(a,"Średnio wiązek / token");a.set_title(f"Blok {b}");a.legend()
save(fig,"przejscia_krokowe","Przejścia między checkpointami",
     "Każdy punkt porównuje bieżący zapis z poprzednim. Odstępy czasu są różne: to liczba zmian w przedziale, nie szybkość zmian na krok.", "Siła i aktywność")

# Separate block dashboards and real, fixed-token examples.
for b,record in enumerate(S["blocks"],1):
    fig,axes=canvas(f"Blok {b}: siła, kąty i zmiany pojedynczych wkładów",COHORT,2,2)
    curve(axes[0],A,"active_rel_0.05",b,label="Liczba >5%")
    curve(axes[0],A,"effective",b,"#222222","Efektywna liczba");timeaxis(axes[0],"Wiązki / token");axes[0].legend()
    for key,c,lab in [("q10","#98A4B0","Kwantyl 10%"),("median","#0072B2","Mediana"),("q90","#D55E00","Kwantyl 90%")]:
        curve(axes[1],A,key,b,c,lab)
    timeaxis(axes[1],"Siła");axes[1].legend()
    curve(axes[2],F,"mean_angle",b,cohort="persistent",band=True);timeaxis(axes[2],"Średni kąt [°]")
    for step,c,lab in [(0,"#98A4B0","Krok 0"),(55000,COLORS[b-1],"Krok 55k")]:
        axes[3].plot(np.arange(180)+.5,H[f"b{b}_k{step}_persistent"]*100,color=c,label=lab)
    axes[3].set(xlim=(35,145),xlabel="Kąt [°]",ylabel="% / przedział 1°");axes[3].legend()
    act=record["activity"];ang=record["angles"]["persistent"]
    note=f"Blok {b}: liczba >5% {act['active_rel_0.05']['initial']:.1f} -> {act['active_rel_0.05']['best']:.1f}; efektywna {act['effective']['initial']:.1f} -> {act['effective']['best']:.1f}; średni kąt {ang['mean_angle_start']:.2f}° -> {ang['mean_angle_best']:.2f}°."
    save(fig,f"blok_{b}_szczegoly",f"Blok {b}: szczegóły",note,f"Blok {b}")

for b in range(1,7):
    subtitle=f"Przykład diagnostyczny: {META['frame']}, stały token {META['fixed_token']} (wiersz {META['token_row']}, kolumna {META['token_col']}; indeksy od 0)."
    fig,axes=canvas(f"Blok {b}: pojedyncze wiązki tego samego tokena",subtitle,2,2)
    s0,s1=EX[f"b{b}_strength0"],EX[f"b{b}_strength1"]
    order=np.argsort(-s1,kind="stable")
    im=axes[0].imshow(EX[f"b{b}_active"][:,order].T,aspect="auto",origin="upper",cmap=ListedColormap(["#EEF1F5",COLORS[b-1]]),interpolation="nearest")
    axes[0].set(xticks=range(11),xticklabels=["0",".3k","1k","3k","6k","10k","16k","24k","33k","44k","55k"],xlabel="Krok (równe kolumny)",ylabel="Ranga wiązki wg siły w 55k",title="Wszystkie 768 wiązek: ponad progiem = kolor")
    axes[0].tick_params(axis="x",rotation=45,labelsize=8);axes[0].grid(False)
    axes[1].scatter(s0,s1,s=10,alpha=.65,color=COLORS[b-1]);axes[1].set_xscale("symlog",linthresh=1e-8);axes[1].set_yscale("symlog",linthresh=1e-8)
    lim=max(s0.max(),s1.max())*1.2;axes[1].plot([0,lim],[0,lim],":",color="#444",lw=1)
    axes[1].set(xlabel="Siła w kroku 0",ylabel="Siła w kroku 55k",title="Każdy punkt = jedna wiązka; nad linią wzrost")
    pairs=EX[f"b{b}_pairs"];history=EX[f"b{b}_pair_angles"]
    drawn={}
    for j,(ids,lab,c) in enumerate(zip(pairs,["Największy kąt w 55k","Największy spadek","Największy wzrost"],["#0072B2","#009E73","#D55E00"])):
        pair=tuple(ids)
        if pair in drawn:
            line=drawn[pair]
            line.set_label(line.get_label()+"\n"+lab)
        else:
            drawn[pair],=axes[2].plot(STEPS,history[:,j],"o-",ms=3,color=c,label=f"{ids[0]} + {ids[1]}: {lab}")
    timeaxis(axes[2],"Kąt [°]",(0,180));axes[2].axhline(90,c="#888",ls=":",lw=1);axes[2].legend(fontsize=7.5)
    k=order[:24]
    axes[3].bar(np.arange(24)-.18,s0[k],width=.36,color="#AAB4BE",label="Krok 0")
    axes[3].bar(np.arange(24)+.18,s1[k],width=.36,color=COLORS[b-1],label="Krok 55k")
    axes[3].set(xticks=range(24),xticklabels=k,xlabel="ID neuronu: 24 najsilniejsze w 55k",ylabel="Siła",title="Te same ID na początku i końcu")
    axes[3].tick_params(axis="x",rotation=90,labelsize=7);axes[3].legend()
    save(fig,f"blok_{b}_pojedyncze_wiazki",f"Blok {b}: pojedynczy token",
         "Stałe ID obrazu, tokena i neuronów. Trzy kryteria wyboru skrajnych par mogą wskazać tę samą parę. Ranking mapy ustalony raz w 55k. Pełne ID i pomiary: SINGLE_TOKEN_BEAMS.csv i SINGLE_TOKEN_PAIRS.csv.",f"Blok {b}")

fig,axes=canvas("Koncentracja siły zależy także od położenia tokena",f"{META['frame']} | Różnica efektywnej liczby: 55k minus 0; wszystkie 600 pozycji.")
maps=[EX[f"b{b}_spatial_effective_55000"]-EX[f"b{b}_spatial_effective_0"] for b in range(1,7)]
lim=max(abs(m).max() for m in maps)
for b,(a,m) in enumerate(zip(axes,maps),1):
    im=a.imshow(m,cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto")
    a.plot(META["token_col"],META["token_row"],"s",mfc="none",mec="black",ms=9)
    a.set(title=f"Blok {b}",xlabel="Kolumna tokena",ylabel="Wiersz tokena");a.grid(False)
fig.colorbar(im,ax=axes.tolist(),shrink=.7,pad=.02,label="Zmiana efektywnej liczby wiązek")
save(fig,"mapy_tokenow","Mapy przestrzenne 600 tokenów",
     "Ujemny wynik oznacza silniejszą koncentrację. Czarny kwadrat wskazuje stały token z poprzednich sześciu wykresów. Jest to pojedyncza klatka diagnostyczna.", "Pojedyncze tokeny")
ATLAS.close()

# Offline gallery, no remote assets.
cards=[]
for p in INDEX:
    stem=p["stem"]
    cards.append(f'<article data-group="{html.escape(p["group"])}"><h2>{p["number"]:02d}. {html.escape(p["title"])}</h2>'
        f'<a href="{stem}.png" target="_blank"><img src="{stem}.png" loading="lazy" alt="{html.escape(p["title"])}"></a>'
        f'<p>{html.escape(p["note"])}</p><nav><a href="{stem}.png">PNG</a><a href="{stem}.svg">SVG</a><a href="{stem}.pdf">PDF</a></nav></article>')
groups=list(dict.fromkeys(p["group"] for p in INDEX))
page='''<!doctype html><html lang="pl"><meta charset="utf-8"><title>Wiązki ViT - atlas</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{margin:0;background:#eef2f6;color:#18314b;font:16px system-ui}header,main{max-width:1280px;margin:auto;padding:24px}
h1{font-size:32px;margin-bottom:8px}header p{line-height:1.6}select,input{padding:12px;border:1px solid #bac8d4;border-radius:8px;font:inherit}
article{background:white;padding:24px;margin:0 0 28px;border-radius:12px;box-shadow:0 3px 18px #1434540d}h2{font-size:22px}
img{width:100%;height:auto}p{line-height:1.55}nav{display:flex;gap:24px}a{color:#0069a5}footer{padding:30px} [hidden]{display:none!important}
</style><header><h1>Wiązki MLP podczas treningu</h1><p>37 szczegółowych plansz · 6 bloków · 11 checkpointów · 50 klatek / 25 filmów.<br>
Kliknij wykres, aby otworzyć pełną rozdzielczość. <b>55k = około 50,29 epoki</b> (35 000 obrazów TRAIN; batch 32).</p>
<p>Siła wiązki = norma pojedynczego wkładu neuronu. Kąty w pełnych 192D, ze znakiem aktywacji. Spadek poniżej 5% maksimum nie oznacza zera.</p>
<p><a href="ATLAS_WYKRESOW.pdf">Pobierz cały atlas PDF</a> · <a href="../Raport/RAPORT.html">Raport</a> · <a href="../Krótki_Wniosek/WNIOSEK.txt">Krótki wniosek</a></p>
<select id="group"><option value="">Wszystkie tematy</option>OPTIONS</select> <input id="search" placeholder="Szukaj w tytułach i opisach" aria-label="Szukaj">
<p id="count"></p></header><main>CARDS</main><footer>Źródło: zapisane pomiary seed3047; bez ponownego treningu. Wstęgi dotyczą filmów, nie niezależnych seedów.</footer>
<script>function filter(){let n=0;document.querySelectorAll('article').forEach(x=>{x.hidden=(group.value&&x.dataset.group!==group.value)||!x.textContent.toLowerCase().includes(search.value.toLowerCase());if(!x.hidden)n++});document.getElementById('count').textContent=n+' plansz'}const group=document.getElementById('group'),search=document.getElementById('search');group.onchange=filter;search.oninput=filter;filter()</script></html>'''
page=page.replace("37 szczegółowych",f"{len(INDEX)} szczegółowych").replace("OPTIONS","".join(f'<option>{html.escape(g)}</option>' for g in groups)).replace("CARDS","".join(cards))
(OUT/"GALERIA.html").write_text(page,encoding="utf-8")
# Export every plotted mean with explicit equal-film aggregation.
for name,df in [("ACTIVITY",A),("ANGLES",F),("DETECTION",D)]:
    group=["block","step","cohort"] if name=="ANGLES" else (["block","step"] if name=="ACTIVITY" else ["step"])
    numeric=df.select_dtypes(include="number").columns.difference(group).tolist()
    film_means=df.groupby(group+["film"])[numeric].mean()
    film_means.groupby(level=group).mean().to_csv(DATA/f"PLOT_MEANS_{name}.csv")
beam_rows,pair_rows=[],[]
for b in range(1,7):
    rank=np.argsort(np.argsort(-EX[f"b{b}_strength1"],kind="stable"),kind="stable")
    for c in range(768):
        beam_rows.append(dict(block=b,frame=META["frame"],token=META["fixed_token"],neuron=c,rank_at_55000=int(rank[c]),
            strength0=float(EX[f"b{b}_strength0"][c]),strength55000=float(EX[f"b{b}_strength1"][c]),
            **{f"above05_k{k}":bool(EX[f"b{b}_active"][j,c]) for j,k in enumerate(STEPS)}))
    for choice,(a,bid) in enumerate(EX[f"b{b}_pairs"]):
        for j,k in enumerate(STEPS):
            pair_rows.append(dict(block=b,frame=META["frame"],token=META["fixed_token"],selection=choice,
                neuron_a=int(a),neuron_b=int(bid),step=int(k),angle=float(EX[f"b{b}_pair_angles"][j,choice])))
pd.DataFrame(beam_rows).to_csv(DATA/"SINGLE_TOKEN_BEAMS.csv",index=False)
pd.DataFrame(pair_rows).to_csv(DATA/"SINGLE_TOKEN_PAIRS.csv",index=False)
(DATA/"PLOT_QC.json").write_text(json.dumps(dict(status="PASS",plots=len(INDEX),input_hashes_verified=True,
    histogram_normalization_verified=True,seed=3047,films=25,bootstrap_replicates=2000,
    epoch_equivalent_55000=55000*32/35000,source_training_batch=32,source_train_images=35000),indent=2))
print("ALL_PLOTS_DONE",len(INDEX),flush=True)
