<#
Keep a minidump of any python.exe crash on a Windows CI runner, and build the native
addon where a failure upload finds it, so that a crash inside a kernel can be located.

    pwsh tools/crash_dumps.ps1 .artifacts/crash-dumps
#>
param([Parameter(Mandatory)][string]$Folder)
$ErrorActionPreference = 'Stop'

$dumps = New-Item -ItemType Directory -Force $Folder
"TORCH_EXTENSIONS_DIR=$env:RUNNER_TEMP\torch-extensions" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV

# Crash a throwaway interpreter and wait for its dump, which is then removed. ctypes
# turns faults inside its own calls into exceptions, so it only clears a list's type
# pointer, and repr() then dereferences it inside the interpreter.
function Test-CrashDump {
    & python -c "import ctypes; value = []; ctypes.memset(id(value) + 8, 0, 8); repr(value)"
    Write-Host "Test crash exit code: $LASTEXITCODE"
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Milliseconds 500
        $written = Get-ChildItem $dumps -Filter *.dmp
        if ($written) {
            $written | Remove-Item
            return $true
        }
    }
    return $false
}

# First choice: Windows Error Reporting's local dumps, which need its service running.
$reporting = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting'
New-ItemProperty -Path $reporting -Name Disabled -Value 0 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $reporting -Name DontShowUI -Value 1 -PropertyType DWord -Force | Out-Null
$key = "$reporting\LocalDumps\python.exe"
New-Item -Path $key -Force | Out-Null
New-ItemProperty -Path $key -Name DumpFolder -Value $dumps.FullName -PropertyType ExpandString -Force | Out-Null
New-ItemProperty -Path $key -Name DumpType -Value 1 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $key -Name DumpCount -Value 10 -PropertyType DWord -Force | Out-Null
Start-Service WerSvc -ErrorAction SilentlyContinue
if (Test-CrashDump) {
    Write-Output "Windows Error Reporting writes crash dumps to $($dumps.FullName)"
    exit 0
}

# Otherwise Sysinternals ProcDump, installed as the postmortem debugger.
$tools = Join-Path $env:RUNNER_TEMP 'procdump'
Invoke-WebRequest https://download.sysinternals.com/files/Procdump.zip -OutFile "$tools.zip"
Expand-Archive "$tools.zip" $tools -Force
$binary = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'procdump64a.exe' } else { 'procdump64.exe' }
& (Join-Path $tools $binary) -accepteula -i $dumps.FullName | Out-Null
if (Test-CrashDump) {
    Write-Output "ProcDump writes crash dumps to $($dumps.FullName)"
} else {
    $service = Get-Service WerSvc -ErrorAction SilentlyContinue
    Write-Output "WerSvc: $($service.Status) ($($service.StartType))"
    Write-Output '::warning::This runner wrote no dump for a test crash; crashes will have none'
}
exit 0
