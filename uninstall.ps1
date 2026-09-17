<#
.SYNOPSIS
    FreeJarvis - remove it.

.DESCRIPTION
    & "$env:LOCALAPPDATA\FreeJarvis\uninstall.ps1"

    Stops it, takes away the shortcuts, the PATH entry and the autostart
    registration, removes the private Python if the installer put one there,
    and deletes the install directory.

    Your settings, calibration and FreeClaw link are offered back to you first,
    copied to the desktop, because they represent real work: a calibration is
    a grid somebody clicked through and jarvis.json holds the password for
    their FreeClaw.

.PARAMETER InstallDir
    Where it is. Default: the directory this script is in.

.PARAMETER KeepSettings
    Copy settings, calibration and the FreeClaw link to the desktop first.
    Asked interactively when not given.

.PARAMETER Yes
    Do not ask anything.
#>
[CmdletBinding()]
param(
    [string]$InstallDir,
    [switch]$KeepSettings,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if (-not $InstallDir) { $InstallDir = $PSScriptRoot }
if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA "FreeJarvis" }

function Step($text) { Write-Host ""; Write-Host "  $text" -ForegroundColor Cyan }
function Info($text) { Write-Host "     $text" -ForegroundColor DarkGray }
function Ok($text)   { Write-Host "     $text" -ForegroundColor Green }
function Warn($text) { Write-Host "     ! $text" -ForegroundColor Yellow }
function Die($text)  { Write-Host ""; Write-Host "  x $text" -ForegroundColor Red; Write-Host ""; exit 1 }

$script:Out = @()
function Native($exe, [string[]]$arguments) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $script:Out = @(& $exe @arguments 2>&1 | ForEach-Object { "$_" })
    } finally {
        $ErrorActionPreference = $prev
    }
    return $LASTEXITCODE
}

Write-Host ""
Write-Host "  FreeJarvis" -ForegroundColor Cyan -NoNewline
Write-Host " - uninstall"
Write-Host ""

if (-not (Test-Path $InstallDir)) { Die "There is nothing at $InstallDir." }

# The guard that matters. An install is a checkout of the repo plus this
# marker, so without the check a developer running this from their working
# copy would delete the thing they were working on.
$marker = Join-Path $InstallDir ".freejarvis-install"
if (-not (Test-Path $marker)) {
    Write-Host "  $InstallDir does not look like an install." -ForegroundColor Yellow
    Write-Host "  It has no .freejarvis-install marker, which means it is" -ForegroundColor DarkGray
    Write-Host "  probably a source checkout. Refusing to delete it." -ForegroundColor DarkGray
    Write-Host ""
    exit 1
}

if (-not $Yes) {
    Write-Host "  This will remove $InstallDir and everything in it."
    $reply = Read-Host "  Type 'yes' to go ahead"
    if ($reply -ne "yes") { Write-Host ""; Write-Host "  Left alone."; Write-Host ""; exit 0 }
}

# -- 1. stop it -----------------------------------------------
$running = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -and
                            ($_.CommandLine -like "*gesture_control*" -or
                             $_.CommandLine -like "*launch.pyw*") })
if ($running.Count -gt 0) {
    Step "Stopping it"
    foreach ($proc in $running) {
        Native "taskkill.exe" @("/F", "/T", "/PID", "$($proc.ProcessId)") | Out-Null
    }
    Start-Sleep -Seconds 2
    Ok "stopped $($running.Count)"
}

# -- 2. keep what is worth keeping ----------------------------
$keepables = @("settings.json", "calibration.json", "jarvis.json", "session_log.csv") |
    ForEach-Object { Join-Path $InstallDir $_ } | Where-Object { Test-Path $_ }
if ($keepables.Count -gt 0) {
    $keep = $KeepSettings
    if (-not $keep -and -not $Yes) {
        Write-Host ""
        Write-Host "  Found $($keepables.Count) file(s) holding your own settings:" -ForegroundColor DarkGray
        $keepables | ForEach-Object { Info (Split-Path $_ -Leaf) }
        $reply = Read-Host "  Copy them to your desktop before deleting? [Y/n]"
        $keep = ($reply -eq "" -or $reply -match '^[Yy]')
    }
    if ($keep) {
        Step "Keeping your settings"
        $saveTo = Join-Path ([Environment]::GetFolderPath("Desktop")) `
                            ("FreeJarvis settings " + (Get-Date -Format "yyyy-MM-dd"))
        New-Item -ItemType Directory -Path $saveTo -Force | Out-Null
        foreach ($file in $keepables) {
            Copy-Item $file -Destination $saveTo -Force -ErrorAction SilentlyContinue
        }
        Ok "copied to $saveTo"
        Warn "jarvis.json holds your FreeClaw password in plain text"
    }
}

# -- 3. shortcuts, PATH, autostart ----------------------------
Step "Removing shortcuts and registrations"
foreach ($folder in @("Programs", "Desktop")) {
    $lnk = Join-Path ([Environment]::GetFolderPath($folder)) "FreeJarvis.lnk"
    if (Test-Path $lnk) { Remove-Item $lnk -Force -ErrorAction SilentlyContinue }
}

$runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
if (Get-ItemProperty -Path $runKey -Name "FreeJarvis" -ErrorAction SilentlyContinue) {
    Remove-ItemProperty -Path $runKey -Name "FreeJarvis" -ErrorAction SilentlyContinue
}

# Same care as putting it there: whole-entry comparison, unexpanded read, and
# only write when something actually has to change.
$binDir = Join-Path $InstallDir "bin"
$key = "HKCU:\Environment"
$current = (Get-ItemProperty -Path $key -Name Path -ErrorAction SilentlyContinue).Path
if ($current) {
    $entries = $current -split ';' | Where-Object { $_.Trim() -ne "" }
    $kept = $entries | Where-Object {
        $_.Trim().TrimEnd('\').ToLower() -ne $binDir.TrimEnd('\').ToLower() }
    if ($kept.Count -ne $entries.Count) {
        Set-ItemProperty -Path $key -Name Path -Value ($kept -join ';') -Type ExpandString
        Ok "taken off PATH"
    }
}
Ok "shortcuts and autostart removed"

# -- 4. the private Python ------------------------------------
# Only if the installer put one there. It is a per-user python.org install, so
# it has an Add/Remove Programs entry that deleting the folder would orphan --
# handing the installer back its own /uninstall is the only clean way out,
# which is why install.ps1 kept a copy of it.
$pyInstaller = Join-Path $InstallDir "python-installer.exe"
if (Test-Path $pyInstaller) {
    Step "Removing the private Python"
    $code = Native $pyInstaller @("/quiet", "/uninstall")
    if ($code -eq 0) { Ok "removed, and its entry in Add/Remove Programs with it" }
    else { Warn "the Python uninstaller returned $code - check Add/Remove Programs" }
    Start-Sleep -Seconds 2
}

# -- 5. the directory -----------------------------------------
Step "Deleting $InstallDir"
# Not from inside it: a shell whose working directory is the folder being
# deleted keeps a handle on it, and the delete half-fails.
Set-Location ([Environment]::GetFolderPath("Desktop"))
try {
    Remove-Item $InstallDir -Recurse -Force
    Ok "gone"
} catch {
    Warn "some of it could not be deleted: $($_.Exception.Message)"
    Warn "close anything using it and delete $InstallDir by hand."
}

Write-Host ""
Write-Host "  ----------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  FreeJarvis is uninstalled." -ForegroundColor Cyan
Write-Host ""
Write-Host "  Two things this did not touch, because they are not its to remove:" -ForegroundColor DarkGray
Write-Host "    - the speech models under %USERPROFILE%\.cache\huggingface" -ForegroundColor DarkGray
Write-Host "    - FreeClaw, if you installed one. It has its own uninstaller." -ForegroundColor DarkGray
Write-Host ""
