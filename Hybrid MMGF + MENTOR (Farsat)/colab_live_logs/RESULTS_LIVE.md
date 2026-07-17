# MMRec Live Results

Last synced: 2026-07-10 09:09:50 UTC

## Runtime

```text
root          66       7  0 06:07 ?        00:00:14 [python3] <defunct>
root          67       7  0 06:07 ?        00:00:02 python3 /usr/local/bin/colab-fileshim.py
root         114       7  0 06:07 ?        00:00:08 /usr/bin/python3 /usr/local/bin/jupyter-server --debug --transport="ipc" --ip=172.28.0.12 --ServerApp.token= --port=9000 --FileContentsManager.root_dir=/ --FileContentsManager.allow_hidden=True --ServerApp.log_format="|%(levelname)s|%(message)s" --ServerApp.iopub_data_rate_limit=1e10 --MappingKernelManager.root_dir=/content
root         424     114  0 06:08 ?        00:00:18 /usr/bin/python3 -m colab_kernel_launcher -f /root/.local/share/jupyter/runtime/kernel-af54e26b-6b24-4dfb-ab99-ec125513e605.json
root        2324       1  0 06:15 ?        00:00:00 bash -c tail -n +0 -F "/root/.config/Google/DriveFS/Logs/drive_fs.txt" | python3 /opt/google/drive/drive-filter.py > "/root/.config/Google/DriveFS/Logs/timeouts.txt" 
root        2327    2324  0 06:15 ?        00:00:00 python3 /opt/google/drive/drive-filter.py
root        2334       1  0 06:15 ?        00:00:01 python3 /content/sync_mmrec_results.py

Tesla T4, 0 %, 0 MiB, 15360 MiB

```

## mentor_only_sports.log

- Latest validation epoch: **82**
- Latest validation: R@10=0.0739, R@20=0.1087, N@10=0.0397, N@20=0.0486
- Best observed validation R@20: epoch 61 with R@10=0.0728, R@20=0.1098, N@10=0.0389, N@20=0.0483
- Final best validation checkpoint: sports, score=0.1098, R@10=0.0728, R@20=0.1098, N@10=0.0389, N@20=0.0483
- **FINAL TEST sports**: R@10=0.0740, R@20=0.1122, N@10=0.0398, N@20=0.0497
