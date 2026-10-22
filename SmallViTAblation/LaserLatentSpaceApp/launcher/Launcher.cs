// Uruchamiacz aplikacji „Lasery i koloidy”: nie pakuje Pythona ani torcha, tylko startuje zainstalowany pythonw
// (z jego torchem i CUDA) ze skryptem run.py leżącym obok exe.
// Ścieżkę Pythona można nadpisać plikiem python.txt obok exe (jedna linia, pełna ścieżka do pythonw.exe).
// Budowanie (PowerShell, z katalogu aplikacji): & C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /nologo /codepage:65001
//   /target:winexe /win32icon:launcher\icon.ico /r:System.Windows.Forms.dll '/out:Lasery i koloidy.exe' launcher\Launcher.cs
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

static class Launcher
{
    const string TYTUL = "Lasery i koloidy";

    [STAThread]
    static int Main()
    {
        string katalog = AppDomain.CurrentDomain.BaseDirectory;
        string skrypt = Path.Combine(katalog, "run.py");
        if (!File.Exists(skrypt))
        {
            MessageBox.Show("Brak run.py obok exe:\n" + skrypt, TYTUL, MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        string python = @"W:\Python\3.11\pythonw.exe";
        string plik = Path.Combine(katalog, "python.txt");
        if (File.Exists(plik))
            python = File.ReadAllText(plik).Trim().Trim('"');
        if (!File.Exists(python))
            python = "pythonw.exe";   // z PATH
        var info = new ProcessStartInfo(python, "\"" + skrypt + "\"");
        info.WorkingDirectory = katalog;
        info.UseShellExecute = false;
        info.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        try
        {
            Process.Start(info);
        }
        catch (Exception e)
        {
            MessageBox.Show("Nie udało się uruchomić Pythona:\n" + python + "\n\n" + e.Message +
                            "\n\nWpisz pełną ścieżkę do pythonw.exe w pliku python.txt obok exe.",
                            TYTUL, MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        return 0;
    }
}
