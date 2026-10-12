from pathlib import Path
p=Path(__file__).with_name('train.py');s=p.read_text(encoding='utf-8')
lines=s.splitlines()
for i,line in enumerate(lines):
    if "'Fit obejmuje 90" in line:
        lines[i]="          'Fit obejmuje40 unikalnych przejsc cp50..90, czyli400 aktualizacji zrodla;31 sliding windows H10. Test10przejsc=100aktualizacji.',"
    if "'Encoder u" in line:
        lines[i]="          'Zakotwiczona prognoza z0+s*(D(A^h phi0)-D(phi0)) ma dodatkowy staly offset originu poza learned768. Encoder raz, bez przyszlych auxiliary i bez teacher forcing.',"
    if "'Wybrana epoka:" in line:
        lines[i]="          'Wybrana epoka: %d. lrG: %.8g. Calkowite kroki Adam: %d. Final refit: %d krokow.'%(chosen,selected_lr,total_steps,model.optimizer_steps),"
p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
