"""Eksport dokumentu 45 do PDF, z samodzielnym dowodem przedziału poprawy.

Wymaga pandoc i XeLaTeX w PATH. Nie zmienia źródłowych dokumentów Markdown.
Pliki składu pozostają w tmp/pdfs/langevin_kalman do kontroli.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


APPENDIX = r"""
\clearpage

## Dodatek A. Pełny dowód przejścia do poprawy straty

Ten dodatek rozwija dowód nierówności (16), aby PDF można było czytać bez
otwierania dokumentu 44. Odtwarza jego twierdzenia T5, T6 i T7.
Nie dodaje założenia, że rzeczywisty detektor już je spełnia.

### A.1. Założenia i znaczenie symboli

Pracujemy w skończenie wymiarowej rzeczywistej przestrzeni parametrów z normą
euklidesową i iloczynem skalarnym. Macierze parametrów identyfikujemy z wektorami;
ich norma jest wtedy normą Frobeniusa.
Niech \(V\) będzie różniczkowalną stratą na osobnych danych sterujących.
Zakładamy, że jej gradient jest \(L\)-Lipschitz na rozważanym odcinku aktualizacji:

\[
\|\nabla V(x)-\nabla V(y)\|\le L\|x-y\|.
\]

Niech \(g=\nabla V(\theta)\), \(\eta\ge0\), \(\theta^+=\theta-\eta d\).
Wektor \(d\) to faktyczny kierunek aktualizacji, obejmujący stan optymalizatora.
Mamy przybliżenia \(\widehat g,\widehat d\) i ważne ograniczenia
\(\|g-\widehat g\|\le\epsilon\), \(\|d-\widehat d\|\le\delta\),
gdzie \(L,\epsilon,\delta\ge0\).

### A.2. Reszta rozwinięcia straty

Z podstawowego twierdzenia rachunku całkowego:

\[
V(\theta-\eta d)-V(\theta)
=-\eta\int_0^1\langle\nabla V(\theta-s\eta d),d\rangle\,ds.
\]

Po odjęciu liniowego składnika i zastosowaniu nierówności Cauchy'ego-Schwarza:

\[
\begin{aligned}
&\left|V(\theta-\eta d)-V(\theta)+\eta\langle g,d\rangle\right|\\
&\quad\le\eta\int_0^1
 \|\nabla V(\theta-s\eta d)-\nabla V(\theta)\|\,\|d\|\,ds\\
&\quad\le L\eta^2\|d\|^2\int_0^1s\,ds
 =\frac{L\eta^2}{2}\|d\|^2.
\end{aligned}\tag{A1}
\]

Definiując \(G=V(\theta)-V(\theta^+)\), otrzymujemy
\(\left|G-\eta\langle g,d\rangle\right|\le L\eta^2\|d\|^2/2\).
Dla \(\eta=0\) równość i ograniczenie są bezpośrednie.

### A.3. Błąd przewidywanej korzyści

Dokładna tożsamość brzmi:

\[
\begin{aligned}
\langle g,d\rangle-\langle\widehat g,\widehat d\rangle
&=\langle g-\widehat g,\widehat d\rangle\\
&\quad+\langle g,d-\widehat d\rangle.
\end{aligned}
\]

Zatem

\[
\begin{aligned}
\left|\langle g,d\rangle-\langle\widehat g,\widehat d\rangle\right|
&\le\epsilon\|\widehat d\|+\|g\|\delta\\
&\le\epsilon\|\widehat d\|+(\|\widehat g\|+\epsilon)\delta.
\end{aligned}\tag{A2}
\]

Ponadto \(\|d\|\le\|\widehat d\|+\delta\). Łącząc (A1), (A2)
i nierówność trójkąta, dostajemy

\[
|G-\widehat a|\le E,\qquad
\widehat a=\eta\langle\widehat g,\widehat d\rangle,
\]
\[
\begin{aligned}
E={}&\eta\big[\epsilon\|\widehat d\|+(\|\widehat g\|+\epsilon)\delta\big]\\
&+\frac{L\eta^2}{2}(\|\widehat d\|+\delta)^2.
\end{aligned}\tag{A3}
\]

W szczególności \(\widehat a-E>0\) gwarantuje zmniejszenie \(V\).
Dla partii \(S\) i \(B\), przy tym samym stanie początkowym treningu,
warunek \(\widehat a_S-E_S>\widehat a_B+E_B\) gwarantuje większą
poprawę po partii \(S\). Porównanie samych dolnych granic nie wystarcza.

### A.4. Jawna wrażliwość gradientu na błąd cech

Dla liniowego odczytu \(W\), ustalonego oznaczenia \(y\) i straty
\(\ell(W;q,y)=\tfrac12\|Wq-y\|^2\) gradient wynosi
\(g_q=(Wq-y)q^\top\). Załóżmy
\(\|W\|_{\rm op}\le M\), \(\|q\|,\|\widetilde q\|\le R\),
\(\|y\|\le Y\). Wówczas

\[
g_q-g_{\widetilde q}
=W(q-\widetilde q)q^\top
+(W\widetilde q-y)(q-\widetilde q)^\top.
\]

Używając \(\|ab^\top\|_F=\|a\|\|b\|\), otrzymujemy

\[
\begin{aligned}
\|g_q-g_{\widetilde q}\|_F
&\le MR\|q-\widetilde q\|+(MR+Y)\|q-\widetilde q\|\\
&=(2MR+Y)\|q-\widetilde q\|.
\end{aligned}\tag{A4}
\]

Przy \(q=\sum_p w_pz_p\), \(\widetilde q=\sum_p w_p\widetilde z_p\)
i tych samych wagach \(w_p\), nierówność trójkąta daje

\[
\|g_q-g_{\widetilde q}\|_F
\le(2MR+Y)\sum_p|w_p|\,\|z_p-\widetilde z_p\|.\tag{A5}
\]

Dla kroku SGD tego odczytu prawa strona jest dopuszczalnym \(\delta\)
w (A3). Dla całego RF-DETR albo AdamW trzeba oddzielnie wykazać właściwą
wrażliwość. Zmiana oznaczenia, wag tokenów lub parametrów odczytu
wprowadza dodatkowe składniki błędu.

### A.5. Wersja z prawdopodobieństwem i granice twierdzenia

Jeśli ograniczenia błędów wszystkich potrzebnych cząstek, gradientów i
kierunków zachodzą jednocześnie na zdarzeniu \(\Omega_0\),
\(\Pr(\Omega_0)\ge1-\alpha\), dowód (A1)-(A5) obowiązuje punktowo na
\(\Omega_0\). Gwarancja poprawy ma zatem prawdopodobieństwo co najmniej
\(1-\alpha\), o ile jej dodatnia dolna granica jest spełniona na tym zdarzeniu.
Ograniczenia marginalne dla przypadkowej klatki nie zapewniają tego dla
klatki wybranej przez maksimum rankingu.

Dowód dotyczy ustalonej straty i jednego kroku. AP, przewaga po całym
treningu, generalizacja na nowe operacje i krótszy czas całej procedury
wymagają dodatkowej analizy. W RF-DETR trzeba także sprawdzić założenie
gładkości przy zmianie dopasowania obiektów i operacjach niegładkich.

W Lean sprawdzono wcześniej algebraiczny rdzeń (A3), konsekwencje dla
porównania partii oraz przeniesienie błędów cząstek przez mapę o założonej
wrażliwości. Nie sprawdzono tam całki (A1), jawnej stałej (A4), kalibracji
Kalmana ani spełnienia założeń przez rzeczywisty detektor.
"""

HEADER = r"""
\usepackage{xurl}
\usepackage{fancyhdr}
\usepackage{titlesec}
\definecolor{inkblue}{HTML}{173650}
\definecolor{muted}{HTML}{596875}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\sffamily\color{muted} Langevin, Kalman i dobór klatek}
\fancyhead[R]{\small\sffamily\color{muted} Wyprowadzenia matematyczne}
\fancyfoot[L]{\small\sffamily\color{muted} RF-DETR / cząstki latentu}
\fancyfoot[R]{\small\sffamily\thepage}
\setlength{\headheight}{16pt}
\renewcommand{\headrulewidth}{0.3pt}
\titleformat{\section}{\Large\sffamily\bfseries\color{inkblue}}{}{0pt}{}
\titleformat{\subsection}{\large\sffamily\bfseries\color{inkblue}}{}{0pt}{}
\titlespacing*{\section}{0pt}{2.0ex plus 1ex}{0.8ex}
\titlespacing*{\subsection}{0pt}{1.5ex plus .5ex}{0.5ex}
\setlength{\emergencystretch}{3em}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 1pt}
\allowdisplaybreaks[1]
\AtBeginDocument{\hypersetup{pdftitle={Reszta rozszerzonego Langevina, niepewność Kalmana i dobór klatek},
 pdfsubject={Wyprowadzenia i dowody warunkowe w opisie cząstek latentu},
 linkcolor=inkblue,urlcolor=inkblue}}
"""

COVER = r"""
\begin{titlepage}
\thispagestyle{empty}
\sffamily
{\color{muted}\large NOTA MATEMATYCZNA / DOKUMENT 45\par}
\vspace{24mm}
{\color{inkblue}\fontsize{29}{35}\selectfont\bfseries
Reszta rozszerzonego\par Langevina, niepewność\par Kalmana i dobór klatek\par}
\vspace{10mm}
{\Large Wyprowadzenia, dowody i założenia\par}
\vspace{12mm}
{\large Opis latentu przez osobne cząstki\par
RF-DETR / wykrywanie narzędzia chirurgicznego\par}
\vfill
\rule{\textwidth}{0.6pt}
\vspace{4mm}
{\normalsize
Pełna treść dokumentu źródłowego oraz dodatek z dowodem przejścia
od błędu cząstek do przedziału poprawy straty.\par
\vspace{3mm}
Dowody są warunkowe. Weryfikacja algebraiczna nie zastępuje sprawdzenia
założeń na rzeczywistym detektorze. Zakres dotychczasowej formalizacji
w Lean wskazano osobno.\par
\vspace{6mm}
23 września 2026\par}
\end{titlepage}
\setcounter{page}{1}
\renewcommand{\contentsname}{Spis treści}
\tableofcontents
\clearpage
"""

LUA = r'''
function Code(el)
  if #el.text > 26 or el.text:match("[/\\]") then
    local display = el.text:gsub(string.char(92), '/')
    return pandoc.RawInline('latex', '\\nolinkurl{' .. display .. '}')
  end
end
'''


def reflow(match):
    body = match.group(1).strip()
    # Explicit alignments in the appendix already have a reviewed layout.
    if r"\begin{" in body:
        return match.group(0)
    tags = re.findall(r"\\tag\{([^}]+)\}", body)
    body = re.sub(r"\\tag\{[^}]+\}", "", body).strip()
    # Break distinct expressions without scaling down mathematical type.
    if len(body) > 130 and (r"\qquad" in body or r"\quad" in body):
        body = re.sub(r"\\q(?:quad|uad)\s*", lambda _: r"\\" + "\n", body)
        body = r"\begin{gathered}" + "\n" + body + "\n" + r"\end{gathered}"
    tag = (r"\tag{" + tags[0] + "}") if tags else ""
    return "\\[\n" + body + tag + "\n\\]"


def main():
    here = Path(__file__).resolve()
    project = here.parents[2]
    source = project / "docs/Analiza_ponownej_publikacji/45_RESZTA_LANGEVINA_KALMAN_I_DOBOR_KLATEK.md"
    supporting = source.with_name("44_FORMALIZACJA_DOBORU_KLATEK_KOOPMAN_LANGEVIN.md")
    scratch = project / "tmp/pdfs/langevin_kalman"
    output = project / "output/pdf/45_Reszta_Langevina_Kalman_dowody.pdf"
    scratch.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    text = text.split("\n", 1)[1].lstrip()
    # Keep human-readable references in print; a Markdown source is not an external web link.
    text = text.replace("[formalizacji 44](44_FORMALIZACJA_DOBORU_KLATEK_KOOPMAN_LANGEVIN.md)",
                        "formalizacji 44; dowód przedziału poprawy zamieszczono także w dodatku A")
    text += "\n" + APPENDIX
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", reflow, text, flags=re.S)
    (scratch / "dokument.md").write_text(text, encoding="utf-8")
    (scratch / "naglowek.tex").write_text(HEADER, encoding="utf-8")
    (scratch / "okladka.tex").write_text(COVER, encoding="utf-8")
    (scratch / "paths.lua").write_text(LUA, encoding="utf-8")
    pandoc, xelatex = shutil.which("pandoc"), shutil.which("xelatex")
    if not pandoc or not xelatex:
        raise RuntimeError("Wymagane pandoc i xelatex w PATH")
    cmd = [pandoc, str(scratch / "dokument.md"),
           "--from=markdown+tex_math_single_backslash", "--to=latex", "--standalone",
           "--lua-filter="+str(scratch / "paths.lua"),
           "--include-in-header="+str(scratch / "naglowek.tex"),
           "--include-before-body="+str(scratch / "okladka.tex"),
           "-V", "documentclass=article", "-V", "fontsize=11pt",
           "-V", "papersize=a4", "-V", "geometry:margin=22mm",
           "-V", "mainfont=Cambria", "-V", "sansfont=Segoe UI",
           "-V", "monofont=Consolas", "-V", "mathfont=Cambria Math",
           "-V", "linestretch=1.08", "-V", "colorlinks=true",
           "-V", "toc-depth=2", "-o", str(scratch / "dokument.tex")]
    subprocess.run(cmd, check=True)
    for i in range(3):
        result = subprocess.run([xelatex, "--disable-installer", "-interaction=nonstopmode",
                                 "-halt-on-error", "-file-line-error", "dokument.tex"],
                                cwd=scratch, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        (scratch / f"przebieg_{i+1}.txt").write_text(result.stdout+result.stderr, encoding="utf-8")
        if result.returncode:
            print((result.stdout+result.stderr)[-7000:])
            raise RuntimeError(f"XeLaTeX: kod {result.returncode}, przebieg {i+1}")
    shutil.copyfile(scratch / "dokument.pdf", output)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(output=str(output), output_sha256=sha(output),
                    sources={str(p):sha(p) for p in (source, supporting, here)},
                    contents="Pełny dokument 45 + dodatek A odtwarzający T5, T6, T7 z dokumentu 44.",
                    note="Dodatek nie rozszerza zakresu weryfikacji Lean.")
    output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(output)
    log = (scratch / "dokument.log").read_text(encoding="utf-8", errors="replace")
    print("\n".join(line for line in log.splitlines()
                    if any(s in line for s in ("Overfull", "Missing character", "Output written"))))


if __name__ == "__main__":
    main()
