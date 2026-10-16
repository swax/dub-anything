<#
Build and stage the local Jellyfin integration. -Install copies the reviewed
build/configuration to Jellyfin. The worker runs as the signed-in Windows user
so Codex/Claude Code can use that user's existing login.
#>
[CmdletBinding()]
param(
    [string]$JellyfinInstallDir = 'C:\Program Files\Jellyfin\Server',
    [string]$JellyfinDataDir = 'C:\ProgramData\Jellyfin\Server',
    [string]$Python = '',
    [switch]$Install,
    [switch]$StartWorker,
    [switch]$EnableAutoUpdates
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Python) { $Python = Join-Path (Split-Path -Parent $taskRoot) 'VoiceStudio\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "VoiceStudio Python not found: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $JellyfinInstallDir 'MediaBrowser.Controller.dll'))) {
    throw "Jellyfin assemblies not found at $JellyfinInstallDir"
}
$taskProject = Join-Path $taskRoot 'integrations\jellyfin\Jellyfin.Plugin.DubAnything.csproj'
& dotnet build $taskProject --configuration Release --nologo "-p:JellyfinInstallDir=$JellyfinInstallDir"
if ($LASTEXITCODE) { throw 'Plugin build failed' }
Push-Location -LiteralPath $taskRoot
try {
    & $Python scripts/integrations service init
    if ($LASTEXITCODE) { throw 'Worker settings initialization failed' }
} finally { Pop-Location }
$taskServiceDir = Join-Path $taskRoot 'work\service'
$taskConfig = Get-Content -LiteralPath (Join-Path $taskServiceDir 'config.json') -Raw | ConvertFrom-Json
$taskStage = Join-Path $taskServiceDir 'plugin'
New-Item -ItemType Directory -Path $taskStage -Force | Out-Null
$taskAssembly = Join-Path $taskRoot 'integrations\jellyfin\bin\Release\net10.0\Jellyfin.Plugin.DubAnything.dll'
$taskPluginVersion = ([version][Diagnostics.FileVersionInfo]::GetVersionInfo($taskAssembly).FileVersion).ToString()
Copy-Item -LiteralPath $taskAssembly -Destination $taskStage -Force
$taskPluginConfig = Join-Path $taskStage 'Jellyfin.Plugin.DubAnything.xml'
$taskExistingConfig = Join-Path $JellyfinDataDir 'plugins\configurations\Jellyfin.Plugin.DubAnything.xml'
if (Test-Path -LiteralPath $taskExistingConfig) {
    # Keep the user's models, default director and review setting on upgrades.
    [xml]$taskXml = Get-Content -LiteralPath $taskExistingConfig -Raw
} else {
    [xml]$taskXml = @'
<?xml version="1.0" encoding="utf-8"?>
<PluginConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <ServiceUrl>http://127.0.0.1:8767</ServiceUrl>
  <ServiceToken></ServiceToken>
  <DefaultDirector>codex</DefaultDirector>
  <CodexModel></CodexModel>
  <ClaudeModel></ClaudeModel>
  <ReviewBeforePublish>true</ReviewBeforePublish>
  <RefreshedJobIds />
</PluginConfiguration>
'@
}
$taskXml.PluginConfiguration.ServiceUrl = "http://$($taskConfig.host):$($taskConfig.port)"
$taskXml.PluginConfiguration.ServiceToken = $taskConfig.token
$taskXml.Save($taskPluginConfig)
Write-Host "Build and private configuration staged in $taskStage"
if ($Install) {
    if (Test-Path -LiteralPath (Join-Path $taskServiceDir 'install.json')) {
        throw 'A coordinated installation is configured. Publish a matched release and let its updater install it when idle.'
    }
    # Automatic installs use the coordinator's atomic gate. A manual install
    # must also refuse a busy worker rather than replace its loaded integration.
    $taskDb = Join-Path $taskServiceDir 'queue.sqlite3'
    if (Test-Path -LiteralPath $taskDb) {
        & $Python -c "import sqlite3,json,sys; db=sqlite3.connect(sys.argv[1]); busy=any(json.loads(r[0])['state'] in ('queued','publish_queued','staging','language_setup','preparing','directing','rendering','publishing') for r in db.execute('SELECT document FROM jobs')); sys.exit(2 if busy else 0)" $taskDb
        if ($LASTEXITCODE) { throw 'Jobs are running or queued. Use -EnableAutoUpdates without -Install to defer the update until idle.' }
    }
    $taskDestination = Join-Path $JellyfinDataDir "plugins\Dub Anything_$taskPluginVersion"
    $taskConfigDestination = Split-Path -Parent $taskExistingConfig
    New-Item -ItemType Directory -Path $taskDestination -Force | Out-Null
    New-Item -ItemType Directory -Path $taskConfigDestination -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $taskStage 'Jellyfin.Plugin.DubAnything.dll') -Destination $taskDestination -Force
    if (Test-Path -LiteralPath $taskExistingConfig) {
        Copy-Item -LiteralPath $taskExistingConfig -Destination (Join-Path $taskStage 'previous-configuration.xml') -Force
    }
    Copy-Item -LiteralPath $taskPluginConfig -Destination $taskExistingConfig -Force
    Write-Host 'Plugin installed. Restart Jellyfin to load it.'
}
if ($EnableAutoUpdates) {
    $taskInstallFile = Join-Path $taskServiceDir 'install.json'
    if (Test-Path -LiteralPath $taskInstallFile) {
        $taskManaged = Get-Content -LiteralPath $taskInstallFile -Raw | ConvertFrom-Json
        $taskManaged.enabled = $true
    } else {
        $taskLoaded = Get-ChildItem -LiteralPath (Join-Path $JellyfinDataDir 'plugins') -Directory |
            Where-Object Name -like 'Dub Anything_*' | Sort-Object Name -Descending | Select-Object -First 1
        if (-not $taskLoaded) { throw 'Install the plugin once before enabling its coordinated updater' }
        $taskLoadedDll = Join-Path $taskLoaded.FullName 'Jellyfin.Plugin.DubAnything.dll'
        $taskLoadedVersion = ([version][Diagnostics.FileVersionInfo]::GetVersionInfo($taskLoadedDll).FileVersion).ToString()
        $taskManaged = [ordered]@{
            enabled = $true
            version = $taskLoadedVersion
            legacy = ($taskLoadedVersion -eq '0.1.0.0')
            source_root = $taskRoot
            active_runtime = $taskRoot
            python = $Python
            voicestudio = (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Python)))
            jellyfin_exe = (Join-Path $JellyfinInstallDir 'jellyfin.exe')
            jellyfin_data = $JellyfinDataDir
            jellyfin_url = 'http://127.0.0.1:8096'
            jellyfin_service = ''
            plugin_path = $taskLoaded.FullName
            check_seconds = 21600
        }
    }
    $taskManaged | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $taskInstallFile -Encoding utf8
    # A logon task runs under the same account as the director CLI logins.
    $taskUpdateLauncher = Join-Path $taskServiceDir 'start-updates.ps1'
    $taskUpdateText = @"
`$taskConfig = Get-Content -LiteralPath '$($taskInstallFile.Replace("'", "''"))' -Raw | ConvertFrom-Json
`$taskRuntime = `$taskConfig.active_runtime
if (-not (Test-Path -LiteralPath (Join-Path `$taskRuntime 'scripts\integrations\__main__.py')) -and
    -not (Test-Path -LiteralPath (Join-Path `$taskRuntime 'scripts\dub_updates.py'))) { `$taskRuntime = `$taskConfig.source_root }
if (Test-Path -LiteralPath (Join-Path `$taskRuntime 'scripts\integrations\__main__.py')) {
    & `$taskConfig.python -u (Join-Path `$taskRuntime 'scripts\integrations') updates watch --install '$($taskInstallFile.Replace("'", "''"))' *>> '$((Join-Path $taskServiceDir 'updates.log').Replace("'", "''"))'
} else {
    & `$taskConfig.python -u (Join-Path `$taskRuntime 'scripts\dub_updates.py') watch --install '$($taskInstallFile.Replace("'", "''"))' *>> '$((Join-Path $taskServiceDir 'updates.log').Replace("'", "''"))'
}
"@
    Set-Content -LiteralPath $taskUpdateLauncher -Value $taskUpdateText -Encoding utf8
    $taskUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $taskAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$taskUpdateLauncher`""
    $taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
    $taskPrincipal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
    $taskOptions = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $taskName = "Dub Anything Updates $($taskConfig.port)"
    Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTrigger -Principal $taskPrincipal -Settings $taskOptions -Force | Out-Null
    Start-ScheduledTask -TaskName $taskName
    Write-Host 'Coordinated automatic updates enabled. Running and queued jobs will finish before any installation.'
}
if ($StartWorker) {
    $taskWorkerRoot = $taskRoot
    $taskWorkerVoiceStudio = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Python))
    $taskManagedFile = Join-Path $taskServiceDir 'install.json'
    if (Test-Path -LiteralPath $taskManagedFile) {
        $taskManagedWorker = Get-Content -LiteralPath $taskManagedFile -Raw | ConvertFrom-Json
        $taskWorkerRoot = $taskManagedWorker.active_runtime
        $taskWorkerVoiceStudio = $taskManagedWorker.voicestudio
    }
    $taskAddress = "http://$($taskConfig.host):$($taskConfig.port)/health"
    try { $taskHealth = Invoke-RestMethod $taskAddress -TimeoutSec 2 } catch { $taskHealth = $null }
    if ($taskHealth.service -eq 'Dub Anything') {
        Write-Host 'Dubbing worker is already running.'
    } else {
        # A child of Win32_Process outlives the calling agent shell. ShowWindow=0
        # keeps the helper hidden; environment and auth remain this user's.
        $taskLauncher = Join-Path $taskServiceDir 'start-worker.ps1'
        $taskEscapedRoot = $taskWorkerRoot.Replace("'", "''")
        $taskEscapedPython = $Python.Replace("'", "''")
        $taskEscapedLog = (Join-Path $taskServiceDir 'service.log').Replace("'", "''")
        $taskLauncherText = @"
Set-Location -LiteralPath '$taskEscapedRoot'
`$env:VOICESTUDIO_DIR = '$($taskWorkerVoiceStudio.Replace("'", "''"))'
if (Test-Path -LiteralPath 'scripts/integrations/__main__.py') {
    & '$taskEscapedPython' -u scripts/integrations service serve *> '$taskEscapedLog'
} else {
    & '$taskEscapedPython' -u scripts/dub_service.py serve *> '$taskEscapedLog'
}
"@
        Set-Content -LiteralPath $taskLauncher -Value $taskLauncherText -Encoding utf8
        $taskStartup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
        $taskReply = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$taskLauncher`""
            CurrentDirectory = $taskRoot
            ProcessStartupInformation = $taskStartup
        }
        if ($taskReply.ReturnValue -ne 0) { throw "Worker launch failed: $($taskReply.ReturnValue)" }
        $taskHealthy = $false
        for ($taskAttempt = 0; $taskAttempt -lt 15; $taskAttempt++) {
            Start-Sleep -Milliseconds 500
            try {
                $taskHealth = Invoke-RestMethod $taskAddress -TimeoutSec 2
                if ($taskHealth.service -eq 'Dub Anything') { $taskHealthy = $true; break }
            } catch { }
        }
        if (-not $taskHealthy) { throw "Worker did not answer. See $taskServiceDir\service.log" }
        Write-Host "Worker is running at http://$($taskConfig.host):$($taskConfig.port)"
    }
}
