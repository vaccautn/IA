Option Explicit
Dim shell, files, folder, python, script
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
folder = files.GetParentFolderName(WScript.ScriptFullName)
python = files.BuildPath(folder, ".venv\Scripts\pythonw.exe")
script = files.BuildPath(folder, "scripts\video_gui.py")
If Not files.FileExists(python) Then
    MsgBox "Falta el entorno Python. Consultar docs/video-prototype.md para instalarlo.", 16, "VACCA"
Else
    shell.Run Chr(34) & python & Chr(34) & " " & Chr(34) & script & Chr(34), 1, False
End If
