$ErrorActionPreference = "SilentlyContinue"
$root = "C:\Users\farsa\dev\GitHub\test_ali"
Set-Location $root
$key = ".codex_colab_ssh\id_ed25519"
$known = ".codex_colab_ssh\known_hosts"
$port = "55771"
$hostName = "bore.pub"
$remote = "root@${hostName}:/content/test_ali/MMRec/src/run_logs/RESULTS_LIVE.md"
while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path "colab_live_logs\autosync_results_status.log" -Value "$ts sync tick"
    scp -q -i $key -o StrictHostKeyChecking=no -o UserKnownHostsFile=$known -o ConnectTimeout=20 -P $port $remote "colab_live_logs\RESULTS_LIVE.md" 2>> "colab_live_logs\autosync_results_errors.log"
    Start-Sleep -Seconds 120
}
