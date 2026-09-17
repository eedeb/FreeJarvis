<#
.SYNOPSIS
    FreeJarvis - Windows installer.

.DESCRIPTION
    irm https://raw.githubusercontent.com/eedeb/FreeJarvis/main/install.ps1 | iex

    Puts FreeJarvis in %LOCALAPPDATA%\FreeJarvis, gives it a Python of its own,
    installs what it needs, fetches the hand-tracking model, and makes a Start
    Menu shortcut. No administrator rights at any point.

    Nothing is hosted anywhere except this script. The source comes from the
    GitHub repo, the interpreter from python.org and the model from Google, so
    a release never has to be cut or an artifact uploaded for an install to
    work.

    Re-running it is the update path: the code is refreshed and your settings,
    calibration and FreeClaw link are never touched.

.PARAMETER InstallDir
    Where to install. Default: %LOCALAPPDATA%\FreeJarvis.

.PARAMETER Branch
    Branch to track. Default: main.

.PARAMETER NoStart
    Install but do not launch it afterwards.

.PARAMETER NoShortcut
    Skip the Start Menu and desktop shortcuts.

.PARAMETER NoPath
    Do not add the `freejarvis` command to PATH.

.PARAMETER Autostart
    Start FreeJarvis when you sign in.

.NOTES
    Piped through `iex` there is nowhere to put parameters, so each one also
    reads an environment variable: FREEJARVIS_DIR, FREEJARVIS_BRANCH,
    FREEJARVIS_NO_START, FREEJARVIS_NO_SHORTCUT, FREEJARVIS_NO_PATH,
    FREEJARVIS_AUTOSTART.

        $env:FREEJARVIS_AUTOSTART = 1
        irm https://raw.githubusercontent.com/eedeb/FreeJarvis/main/install.ps1 | iex
#>
[CmdletBinding()]
param(
    [string]$InstallDir,
    [string]$Branch,
    [switch]$NoStart,
    [switch]$NoShortcut,
    [switch]$NoPath,
    [switch]$Autostart
)

$ErrorActionPreference = "Stop"

# Invoke-WebRequest draws a progress bar by writing to the console on every
# chunk, which on Windows PowerShell makes a large download several times
# slower than it needs to be. This install downloads a few hundred megabytes.
$ProgressPreference = "SilentlyContinue"

$RepoUrl  = "https://github.com/eedeb/FreeJarvis"
$RepoZip  = "https://codeload.github.com/eedeb/FreeJarvis/zip/refs/heads"
# Fetched only when the machine has no suitable Python of its own. 3.12 rather
# than the newest: every wheel this app needs exists for it, which is not yet
# reliably true one version further on, and a missing wheel means pip tries to
# build MediaPipe from source and fails an hour later.
$PythonVersion = "3.12.10"
# What counts as suitable if one is already here. onnxruntime needs 3.11+.
$MinPython = [version]"3.11"
$MaxPython = [version]"3.14"      # exclusive

# -- parameters, or the environment when piped through iex ----
function Env-Or($value, $name) {
    if ($value) { return $value }
    $v = [Environment]::GetEnvironmentVariable($name)
    if ($v) { return $v }
    return $null
}
function Env-Flag($switch, $name) {
    if ($switch) { return $true }
    $v = [Environment]::GetEnvironmentVariable($name)
    return [bool]($v -and $v -ne "0" -and $v -ne "false")
}

$InstallDir = Env-Or $InstallDir "FREEJARVIS_DIR"
$Branch     = Env-Or $Branch     "FREEJARVIS_BRANCH"
$NoStart    = Env-Flag $NoStart    "FREEJARVIS_NO_START"
$NoShortcut = Env-Flag $NoShortcut "FREEJARVIS_NO_SHORTCUT"
$NoPath     = Env-Flag $NoPath     "FREEJARVIS_NO_PATH"
$Autostart  = Env-Flag $Autostart  "FREEJARVIS_AUTOSTART"

if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA "FreeJarvis" }
if (-not $Branch)     { $Branch = "main" }

# -- output ---------------------------------------------------
function Step($text) { Write-Host ""; Write-Host "  $text" -ForegroundColor Cyan }
function Info($text) { Write-Host "     $text" -ForegroundColor DarkGray }
function Ok($text)   { Write-Host "     $text" -ForegroundColor Green }
function Warn($text) { Write-Host "     ! $text" -ForegroundColor Yellow }
function Die($text)  { Write-Host ""; Write-Host "  x $text" -ForegroundColor Red; Write-Host ""; exit 1 }

# Every external command goes through this, for one specific reason: Windows
# PowerShell turns a native command's stderr into ErrorRecords when it is
# redirected, and with $ErrorActionPreference = 'Stop' the first line git
# writes to stderr - "From https://github.com/..." on a perfectly *successful*
# fetch - aborts the whole script. Relaxing the preference for the duration of
# the call and then checking the real exit code is what makes that survivable.
# Output is left in $script:Out for the callers that want it.
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
Write-Host " - a hand-tracking overlay with an assistant behind it"
Write-Host ""

# -- 1. preflight ---------------------------------------------
if ($PSVersionTable.PSVersion.Major -lt 5) {
    Die "PowerShell 5 or newer is required (found $($PSVersionTable.PSVersion))."
}
if (-not [Environment]::Is64BitOperatingSystem) {
    Die "FreeJarvis needs 64-bit Windows."
}
if ([Environment]::OSVersion.Version.Major -lt 10) {
    Die "Windows 10 or newer is required. The overlay uses per-pixel layered windows."
}
# Windows PowerShell defaults to SSL3/TLS1.0, which github.com and python.org
# both refuse.
try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

$IsUpgrade = Test-Path (Join-Path $InstallDir "gesture_control")

# -- 2. stop a running copy -----------------------------------
# Two copies fighting over one camera and one MCP port is the failure this
# avoids, and it is not hypothetical: a second copy binding port 8788 is how
# a stale tool list reached FreeClaw during development.
$running = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -and
                            ($_.CommandLine -like "*gesture_control*" -or
                             $_.CommandLine -like "*launch.pyw*") })
if ($running.Count -gt 0) {
    Step "Stopping the copy that is already running"
    foreach ($proc in $running) {
        Native "taskkill.exe" @("/F", "/T", "/PID", "$($proc.ProcessId)") | Out-Null
    }
    Start-Sleep -Seconds 2      # Windows releases file handles asynchronously
    Ok "stopped $($running.Count)"
}

# -- 3. source ------------------------------------------------
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
$git = (Get-Command git -ErrorAction SilentlyContinue).Source

if ($git) {
    Push-Location $InstallDir
    try {
        if (Test-Path (Join-Path $InstallDir ".git")) {
            Step "Updating from $RepoUrl"
            if ((Native $git @("fetch", "--depth", "1", "origin", $Branch)) -ne 0) {
                Die "git fetch failed - no network, or the repo moved."
            }
            # Hard reset is safe here and nowhere near user data: settings.json,
            # calibration.json and jarvis.json are all gitignored, so git has
            # never tracked them and cannot revert them.
            if ((Native $git @("reset", "--hard", "origin/$Branch")) -ne 0) {
                Die "git reset failed.`n     $($script:Out -join "`n     ")"
            }
            Ok "source updated"
        } else {
            Step "Cloning $RepoUrl"
            # init + fetch rather than `git clone`, because the directory may
            # already exist - clone refuses a non-empty target; this does not.
            Native $git @("init", "-q") | Out-Null
            Native $git @("remote", "add", "origin", $RepoUrl) | Out-Null
            if ((Native $git @("fetch", "--depth", "1", "origin", $Branch)) -ne 0) {
                # A repo whose default branch is master rather than main is
                # the one guess worth making before giving up: the failure is
                # otherwise "check your network" for something that is not the
                # network at all.
                $first = $script:Out
                if ($Branch -eq "main" -and
                    (Native $git @("fetch", "--depth", "1", "origin", "master")) -eq 0) {
                    $Branch = "master"
                    Info "no 'main' branch - using 'master'"
                } else {
                    Die "Couldn't fetch $RepoUrl - check your network, and that the repo is public.`n     $($first -join "`n     ")"
                }
            }
            if ((Native $git @("checkout", "-f", "-B", $Branch, "origin/$Branch")) -ne 0) {
                Die "git checkout failed.`n     $($script:Out -join "`n     ")"
            }
            Ok "source cloned"
        }
    } finally {
        Pop-Location
    }
} else {
    # No git, and FreeJarvis does not otherwise need it, so this is a plain
    # download rather than a reason to send someone away to install something.
    # The zip holds only tracked files, so extracting over an existing install
    # cannot touch settings or calibration - they were never in it.
    Step "Downloading $RepoUrl"
    Info "git isn't installed, so this is a zip download instead"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("freejarvis-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    try {
        $zip = Join-Path $tmp "source.zip"
        try {
            Invoke-WebRequest "$RepoZip/$Branch" -OutFile $zip -UseBasicParsing
        } catch {
            Die "Couldn't download the source: $($_.Exception.Message)"
        }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $tmp)
        $inner = Get-ChildItem -Path $tmp -Directory | Select-Object -First 1
        if (-not $inner) { Die "The downloaded zip was not what was expected." }
        Copy-Item -Path (Join-Path $inner.FullName "*") -Destination $InstallDir `
                  -Recurse -Force
        Ok "source downloaded"
    } finally {
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path (Join-Path $InstallDir "gesture_control\__main__.py"))) {
    Die "The source is not where it should be in $InstallDir."
}
$Version = "unknown"
$versionFile = Join-Path $InstallDir "VERSION"
if (Test-Path $versionFile) { $Version = (Get-Content $versionFile -Raw).Trim() }

# -- 4. an interpreter ----------------------------------------
# A private one, in the install directory, so nothing here depends on or
# disturbs a Python the user already has. The embeddable distribution that
# would be the obvious choice is not usable: it ships without tkinter, and
# tkinter is what draws the "Add FreeClaw" window and the calibration grid --
# so an install built on it would look fine until the moment somebody pressed
# Ctrl+Alt+J. This uses the full installer in per-user mode instead.
function Find-Python {
    $seen = @()
    # The launcher first: it knows about every Python on the machine, and
    # reports their real paths without any of them being on PATH.
    $launcher = (Get-Command py -ErrorAction SilentlyContinue).Source
    if ($launcher) {
        if ((Native $launcher @("-0p")) -eq 0) {
            foreach ($line in $script:Out) {
                if ($line -match '([A-Za-z]:\\[^\s].*python\.exe)') {
                    $seen += $Matches[1]
                }
            }
        }
    }
    $onPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($onPath) { $seen += $onPath }

    $best = $null; $bestVersion = $null
    foreach ($candidate in ($seen | Select-Object -Unique)) {
        if (-not (Test-Path $candidate)) { continue }
        # Version, bitness and tkinter in one go: all three have to hold, and
        # asking the interpreter is the only honest way to know any of them.
        if ((Native $candidate @("-c", "import sys,struct;import tkinter;print('%d.%d %d' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P')*8))")) -ne 0) {
            continue
        }
        $answer = ($script:Out | Select-Object -Last 1) -split ' '
        if ($answer.Count -lt 2 -or $answer[1] -ne "64") { continue }
        try { $found = [version]$answer[0] } catch { continue }
        if ($found -lt $MinPython -or $found -ge $MaxPython) { continue }
        # Prefer 3.12: the version every wheel here is certain to exist for.
        $score = if ($found.Minor -eq 12) { 100 } else { 50 - $found.Minor }
        if ($null -eq $bestVersion -or $score -gt $bestVersion) {
            $best = $candidate; $bestVersion = $score
        }
    }
    return $best
}

$PyDir = Join-Path $InstallDir "python"
$PrivatePy = Join-Path $PyDir "python.exe"
$Base = $null

if (Test-Path $PrivatePy) {
    $Base = $PrivatePy
    Info "using the Python already in $PyDir"
} else {
    Step "Looking for a Python to build on"
    $Base = Find-Python
    if ($Base) {
        Native $Base @("-c", "import sys;print(sys.version.split()[0])") | Out-Null
        Ok "found Python $($script:Out | Select-Object -Last 1) at $Base"
    } else {
        Info "no suitable Python 3.11-3.13 with tkinter, fetching one"
        Step "Fetching Python $PythonVersion"
        $exeName = "python-$PythonVersion-amd64.exe"
        $tmp = Join-Path ([IO.Path]::GetTempPath()) ("freejarvis-py-" + [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $tmp -Force | Out-Null
        $installer = Join-Path $tmp $exeName
        try {
            Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVersion/$exeName" `
                -OutFile $installer -UseBasicParsing
        } catch {
            Die "Couldn't download Python: $($_.Exception.Message)"
        }
        Info "installing it privately - this touches nothing outside $PyDir"
        # Per-user, no PATH, no file associations, no Start Menu entry: this
        # interpreter belongs to FreeJarvis and should be invisible otherwise.
        # Include_tcltk is the whole reason for taking this route.
        $code = Native $installer @(
            "/quiet", "InstallAllUsers=0", "TargetDir=$PyDir", "AssociateFiles=0",
            "PrependPath=0", "Shortcuts=0", "Include_launcher=0", "Include_test=0",
            "Include_doc=0", "Include_tcltk=1", "Include_pip=1")
        # The installer keeps a copy of itself so uninstall.ps1 can hand it
        # back its own /uninstall - there is no other clean way to remove a
        # per-user Python install and its Add/Remove Programs entry.
        if (Test-Path $installer) {
            Copy-Item $installer -Destination (Join-Path $InstallDir "python-installer.exe") -Force
        }
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
        if ($code -ne 0 -or -not (Test-Path $PrivatePy)) {
            Die "The Python installer did not finish (exit $code)."
        }
        $Base = $PrivatePy
        Ok "python $PythonVersion installed privately"
    }
}

# -- 5. the environment ---------------------------------------
# A venv either way, so that everything downstream - the shortcut, the shim,
# the uninstaller - has exactly one interpreter path to talk about, whether
# the base Python was found on the machine or fetched.
$VenvDir = Join-Path $InstallDir ".venv"
$PyExe = Join-Path $VenvDir "Scripts\python.exe"
$PywExe = Join-Path $VenvDir "Scripts\pythonw.exe"
if (-not (Test-Path $PyExe)) {
    Step "Making a private environment"
    if ((Native $Base @("-m", "venv", $VenvDir)) -ne 0 -or -not (Test-Path $PyExe)) {
        Die "Couldn't create the environment in $VenvDir.`n     $($script:Out -join "`n     ")"
    }
    Ok "environment ready"
}

# -- 6. dependencies ------------------------------------------
Step "Installing dependencies"
Info "this is the slow part - about 700 MB, several minutes on a first install"
Native $PyExe @("-m", "pip", "install", "--upgrade", "pip", "--quiet",
                "--disable-pip-version-check") | Out-Null
$code = Native $PyExe @("-m", "pip", "install", "--no-cache-dir",
                        "--disable-pip-version-check", "--no-warn-script-location",
                        "-r", (Join-Path $InstallDir "requirements.txt"))
if ($code -ne 0) {
    $script:Out | Where-Object { $_ -match 'ERROR|error:' } | Select-Object -Last 6 |
        ForEach-Object { Info $_ }
    Die "pip install failed - see the errors above."
}
Ok "dependencies installed"

# -- 7. the hand model ----------------------------------------
# Fetched now rather than on first run, so that a first launch is instant and
# a network problem surfaces here where there is a console to say so. Asking
# the app to do it keeps the URL in one place: it already knows, and a second
# copy here would be a second thing to keep current.
Step "Fetching the hand-tracking model"
# The install path arrives as an argument rather than interpolated into the
# source: it is user-controlled, and a quote or a backslash in it would
# otherwise be Python syntax rather than a path.
$code = Native $PyExe @("-c",
    "import sys; sys.path.insert(0, sys.argv[1]); from gesture_control.tracker import ensure_model; print(ensure_model())",
    $InstallDir)
if ($code -ne 0) {
    Warn "couldn't fetch it now - FreeJarvis will get it on first run"
} else {
    Ok "model ready"
}

# -- 8. the freejarvis command --------------------------------
# Its own directory, because that is what goes on PATH. Everything else under
# the install root would come with it, and putting a bare python.exe on a
# user's PATH is not something an app should do behind their back.
$binDir = Join-Path $InstallDir "bin"
New-Item -ItemType Directory -Path $binDir -Force | Out-Null
Set-Content -LiteralPath (Join-Path $binDir "freejarvis.cmd") -Encoding ascii -Value @(
    "@echo off",
    "rem Written by install.ps1. Runs FreeJarvis with a console, so that the",
    "rem startup notes and any error are visible.",
    "`"$PyExe`" -m gesture_control %*"
)

# The marker uninstall.ps1 keys off. An install looks exactly like a developer's
# checkout from the outside; this file is the only thing that tells them apart.
Set-Content -LiteralPath (Join-Path $InstallDir ".freejarvis-install") -Encoding ascii -Value @(
    "# Written by install.ps1. Its presence marks this directory as a",
    "# FreeJarvis install rather than a source checkout; uninstall.ps1 looks",
    "# for it before deleting anything.",
    "version=$Version",
    "installed=$(Get-Date -Format s)"
)

# -- 9. shortcuts, PATH, autostart ----------------------------
$launcher = Join-Path $InstallDir "windows\launch.pyw"
$icon = Join-Path $InstallDir "windows\freejarvis.ico"
if (-not (Test-Path $icon)) { $icon = $PywExe }

function New-Shortcut($path, $description) {
    $shell = New-Object -ComObject WScript.Shell
    $s = $shell.CreateShortcut($path)
    # pythonw, not python: the overlay is the interface, and a console window
    # sitting behind it for the whole session is not. windows/launch.pyw sends
    # everything that would have been printed to logs\freejarvis.log instead.
    $s.TargetPath = $PywExe
    $s.Arguments = '"' + $launcher + '"'
    $s.WorkingDirectory = $InstallDir
    $s.IconLocation = $icon
    $s.Description = $description
    $s.Save()
}

if (-not $NoShortcut) {
    New-Shortcut (Join-Path ([Environment]::GetFolderPath("Programs")) "FreeJarvis.lnk") `
                 "Hand-tracking overlay with a voice assistant"
    New-Shortcut (Join-Path ([Environment]::GetFolderPath("Desktop")) "FreeJarvis.lnk") `
                 "Hand-tracking overlay with a voice assistant"
    Ok "Start Menu and desktop shortcuts"
}

if (-not $NoPath) {
    # HKCU\Environment\Path is the user's own PATH and one careless write stops
    # every command on the machine resolving. So: read it unexpanded (it
    # routinely contains %USERPROFILE%), compare whole entries, and only write
    # when our bin directory is genuinely absent.
    $key = "HKCU:\Environment"
    $current = (Get-ItemProperty -Path $key -Name Path -ErrorAction SilentlyContinue).Path
    if ($null -eq $current) { $current = "" }
    $entries = $current -split ';' | Where-Object { $_.Trim() -ne "" }
    $normalised = $entries | ForEach-Object { $_.Trim().TrimEnd('\').ToLower() }
    if ($normalised -notcontains $binDir.TrimEnd('\').ToLower()) {
        Set-ItemProperty -Path $key -Name Path -Value ((@($entries) + $binDir) -join ';') `
                         -Type ExpandString
        Ok "added $binDir to PATH (open a new terminal to pick it up)"
    }
}

if ($Autostart) {
    Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" `
        -Name "FreeJarvis" -Value ('"' + $PywExe + '" "' + $launcher + '"')
    Ok "will start when you sign in"
}

# -- 10. start ------------------------------------------------
if (-not $NoStart) {
    Step "Starting FreeJarvis"
    Start-Process -FilePath $PywExe -ArgumentList ('"' + $launcher + '"') `
                  -WorkingDirectory $InstallDir
    Info "the overlay takes a few seconds to open the camera"
}

# -- done -----------------------------------------------------
Write-Host ""
Write-Host "  ----------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""
if ($IsUpgrade) {
    Write-Host "  Updated to FreeJarvis $Version." -ForegroundColor Cyan
    Write-Host "  Your settings, calibration and FreeClaw link were left alone." -ForegroundColor DarkGray
} else {
    Write-Host "  FreeJarvis $Version is installed." -ForegroundColor Cyan
}
Write-Host ""
Write-Host "  Start         " -NoNewline -ForegroundColor DarkGray
Write-Host "the FreeJarvis shortcut, or  freejarvis  in a terminal"
Write-Host "  Quit          " -NoNewline -ForegroundColor DarkGray
Write-Host "the QUIT button in the corner, or Ctrl+Alt+Q"
Write-Host "  Hand back     " -NoNewline -ForegroundColor DarkGray
Write-Host "Ctrl+Alt+M gives the mouse back to your mouse"
Write-Host "  Add a brain   " -NoNewline -ForegroundColor DarkGray
Write-Host "Ctrl+Alt+J, and point it at a FreeClaw"
Write-Host "  Log           " -NoNewline -ForegroundColor DarkGray
Write-Host "$InstallDir\logs\freejarvis.log"
Write-Host "  Update        " -NoNewline -ForegroundColor DarkGray
Write-Host "run this installer again"
Write-Host "  Uninstall     " -NoNewline -ForegroundColor DarkGray
Write-Host "& `"$InstallDir\uninstall.ps1`""
Write-Host ""
Write-Host "  It needs a webcam. Voice and the assistant are optional -" -ForegroundColor DarkGray
Write-Host "  without a FreeClaw it is still a hand-tracking mouse." -ForegroundColor DarkGray
Write-Host ""
