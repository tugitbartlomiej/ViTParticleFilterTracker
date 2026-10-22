"""Stage'uje w indeksie tylko zmianę treści (wersja z indeksu + podmiana fragmentów), bez zmiany końców linii w pliku roboczym."""
import subprocess

ROOT = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker'
ZMIANY = {
    'SmallViTAblation/app/app.py': [(
        "    ('zakladka_ruchu', 'Ruch zbioru', 'Ruch zbioru: cząsteczki i optymalizacja', 'zakladka_ruchu', 'ZakladkaRuchu', ()),\n"
        "    ('zakladka_laserow', 'Lasery i koloidy', 'Lasery i koloidy: oś czasu uczenia jednego zdjęcia — reflektory, studnie, drgania, '\n"
        "     'atencja 6 bloków, IoU i pewność (filmy 2, 12, 16)', 'zakladka_laserow', 'ZakladkaLaserow', ()),\n"
        ")\n",
        "    ('zakladka_ruchu', 'Ruch zbioru', 'Ruch zbioru: cząsteczki i optymalizacja', 'zakladka_ruchu', 'ZakladkaRuchu', ()),\n"
        ")\n"
        "# „Lasery i koloidy” to od 2026-10-22 osobna aplikacja: ../LaserLatentSpaceApp (Lasery i koloidy.exe)\n")],
    'SmallViTAblation/app/ustawienia.py': [(
        "'przestrzen_makro', 'opcje_atencji', 'zakladka_ruchu', 'zakladka_laserow', 'rzuty_okienka')",
        "'przestrzen_makro', 'opcje_atencji', 'zakladka_ruchu')")],
}
for sciezka, zm in ZMIANY.items():
    tresc = subprocess.run(['git', 'show', f':{sciezka}'], cwd=ROOT, capture_output=True, check=True).stdout.decode('utf-8')
    crlf = '\r\n' in tresc
    for a, b in zm:
        if crlf:
            a, b = a.replace('\n', '\r\n'), b.replace('\n', '\r\n')
        assert tresc.count(a) == 1, (sciezka, a[:60])
        tresc = tresc.replace(a, b)
    sha = subprocess.run(['git', 'hash-object', '-w', '--stdin'], cwd=ROOT, input=tresc.encode('utf-8'), capture_output=True,
                         check=True).stdout.decode().strip()
    subprocess.run(['git', 'update-index', '--cacheinfo', f'100644,{sha},{sciezka}'], cwd=ROOT, check=True)
    print(sciezka, 'crlf' if crlf else 'lf', sha[:8])
