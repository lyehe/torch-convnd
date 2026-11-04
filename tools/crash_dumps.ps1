<#
Keep a minidump of any python.exe crash on a Windows CI runner, and build the native
addon where a failure upload finds it, so that a crash inside a kernel can be located.

    pwsh tools/crash_dumps.ps1 .artifacts/crash-dumps

Windows Error Reporting writes the dumps. Processes that inherit SEM_NOGPFAULTERRORBOX
never reach it, so TORCH_CONVND_CRASH_DUMPS=1 makes the test session clear that flag.
#>
param([Parameter(Mandatory)][string]$Folder)
$ErrorActionPreference = 'Stop'

$dumps = New-Item -ItemType Directory -Force $Folder
$reporting = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting'
New-ItemProperty -Path $reporting -Name Disabled -Value 0 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $reporting -Name DontShowUI -Value 1 -PropertyType DWord -Force | Out-Null
$key = "$reporting\LocalDumps\python.exe"
New-Item -Path $key -Force | Out-Null
New-ItemProperty -Path $key -Name DumpFolder -Value $dumps.FullName -PropertyType ExpandString -Force | Out-Null
New-ItemProperty -Path $key -Name DumpType -Value 1 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $key -Name DumpCount -Value 10 -PropertyType DWord -Force | Out-Null
Start-Service WerSvc -ErrorAction SilentlyContinue
"TORCH_EXTENSIONS_DIR=$env:RUNNER_TEMP\torch-extensions" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV
'TORCH_CONVND_CRASH_DUMPS=1' | Out-File -Append -Encoding utf8 $env:GITHUB_ENV

# Crash a throwaway interpreter as the test session would, then remove its dump. ctypes
# turns faults inside its own calls into exceptions, so it only clears a list's type
# pointer, and repr() then dereferences it inside the interpreter.
& python -c @'
import ctypes
kernel32 = ctypes.windll.kernel32
print('Inherited error mode:', hex(kernel32.GetErrorMode()), flush=True)
kernel32.SetErrorMode(kernel32.GetErrorMode() & ~0x0002)
value = []
ctypes.memset(id(value) + 8, 0, 8)
repr(value)
'@
Write-Output "Test crash exit code: $LASTEXITCODE"
$written = $null
for ($attempt = 0; $attempt -lt 60 -and -not $written; $attempt++) {
    Start-Sleep -Milliseconds 500
    $written = Get-ChildItem $dumps -Filter *.dmp
}
if ($written) {
    $written | Remove-Item
    Write-Output "python.exe crash dumps go to $($dumps.FullName)"
} else {
    Write-Output '::warning::This runner wrote no dump for a test crash; crashes will have none'
}
exit 0
