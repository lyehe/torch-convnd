<#
Keep a minidump of any python.exe crash on a Windows CI runner, and build the native
addon where a failure upload finds it, so that a crash inside a kernel can be located.

    pwsh tools/crash_dumps.ps1 .artifacts/crash-dumps
#>
param([Parameter(Mandatory)][string]$Folder)
$ErrorActionPreference = 'Stop'

$dumps = New-Item -ItemType Directory -Force $Folder
$key = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\python.exe'
New-Item -Path $key -Force | Out-Null
New-ItemProperty -Path $key -Name DumpFolder -Value $dumps.FullName -PropertyType ExpandString -Force | Out-Null
New-ItemProperty -Path $key -Name DumpType -Value 1 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $key -Name DumpCount -Value 10 -PropertyType DWord -Force | Out-Null
"TORCH_EXTENSIONS_DIR=$env:RUNNER_TEMP\torch-extensions" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV

# Crash once on purpose: confirm this runner writes dumps, then remove the test dump.
& python -c "__import__('ctypes').string_at(0)"
$written = $null
for ($attempt = 0; $attempt -lt 20 -and -not $written; $attempt++) {
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
