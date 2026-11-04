<#
Keep a minidump of any python.exe crash on a Windows CI runner, and build the native
addon where a failure upload finds it, so that a crash inside a kernel can be located.

    pwsh tools/crash_dumps.ps1 .artifacts/crash-dumps
#>
param([Parameter(Mandatory)][string]$Folder)
$ErrorActionPreference = 'Stop'

$dumps = New-Item -ItemType Directory -Force $Folder
$reporting = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting'
# Windows Error Reporting writes the dumps; make sure it is on and never shows UI.
New-ItemProperty -Path $reporting -Name Disabled -Value 0 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $reporting -Name DontShowUI -Value 1 -PropertyType DWord -Force | Out-Null
$key = "$reporting\LocalDumps\python.exe"
New-Item -Path $key -Force | Out-Null
New-ItemProperty -Path $key -Name DumpFolder -Value $dumps.FullName -PropertyType ExpandString -Force | Out-Null
New-ItemProperty -Path $key -Name DumpType -Value 1 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $key -Name DumpCount -Value 10 -PropertyType DWord -Force | Out-Null
"TORCH_EXTENSIONS_DIR=$env:RUNNER_TEMP\torch-extensions" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV

# Crash once on purpose, then remove the test dump. ctypes turns faults inside its own
# calls into exceptions, so it only clears a list's type pointer, and repr() then
# dereferences it inside the interpreter.
& python -c "import ctypes; value = []; ctypes.memset(id(value) + 8, 0, 8); repr(value)"
$written = $null
for ($attempt = 0; $attempt -lt 60 -and -not $written; $attempt++) {
    Start-Sleep -Milliseconds 500
    $written = Get-ChildItem $dumps -Filter *.dmp
}
if ($written) {
    $written | Remove-Item
    Write-Output "python.exe crash dumps go to $($dumps.FullName)"
} else {
    $service = Get-Service WerSvc -ErrorAction SilentlyContinue
    $policy = Get-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\Windows Error Reporting' -ErrorAction SilentlyContinue
    Write-Output "Test crash exit code: $LASTEXITCODE; WerSvc: $($service.Status) ($($service.StartType)); policy Disabled: $($policy.Disabled)"
    Write-Output '::warning::This runner wrote no dump for a test crash; crashes will have none'
}
exit 0
