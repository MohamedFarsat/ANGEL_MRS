$ErrorActionPreference = "SilentlyContinue"
$root = "C:\Users\farsa\dev\GitHub\test_ali"
Set-Location $root
$key = ".codex_colab_ssh\id_ed25519"
$known = ".codex_colab_ssh\known_hosts"
$port = "25032"
$hostName = "bore.pub"
$remoteSports = "root@${hostName}:/content/test_ali/MMRec/src/run_logs/mentor_only_sports.log"
$remoteClothing = "root@${hostName}:/content/test_ali/MMRec/src/run_logs/mentor_only_clothing.log"
while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path "colab_live_logs\autosync_status.log" -Value "$ts sync tick"
    scp -O -q -i $key -o StrictHostKeyChecking=no -o UserKnownHostsFile=$known -o ConnectTimeout=20 -P $port $remoteSports "colab_live_logs\mentor_only_sports.log" 2>> "colab_live_logs\autosync_errors.log"
    scp -O -q -i $key -o StrictHostKeyChecking=no -o UserKnownHostsFile=$known -o ConnectTimeout=20 -P $port $remoteClothing "colab_live_logs\mentor_only_clothing.log" 2>> "colab_live_logs\autosync_errors.log"
    Start-Sleep -Seconds 120
}
