// Uruchamiacz edytora latentu: nie pakuje Pythona ani torcha, tylko startuje zainstalowany pythonw
// (z jego torchem i CUDA) ze skryptem start.py leżącym obok exe.
// Ścieżkę Pythona można nadpisać plikiem python.txt obok exe (jedna linia, pełna ścieżka do pythonw.exe).
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

static class Uruchamiacz
{
    [STAThread]
    static int Main()
    {
        string katalog = AppDomain.CurrentDomain.BaseDirectory;
        string skrypt = Path.Combine(katalog, "start.py");
        if (!File.Exists(skrypt))
        {
            MessageBox.Show("Brak start.py obok exe:\n" + skrypt, "Edytor latentu", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        string python = @"F:\Python\3.13\pythonw.exe";
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
                            "Edytor latentu", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        return 0;
    }
}
