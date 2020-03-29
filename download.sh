mkdir dataset
cd dataset
wget https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv -O telemetry.csv
wget https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_errors.csv -O errors.csv
wget https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_maint.csv -O maintenance.csv
wget https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_failures.csv -O failures.csv
wget https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_machines.csv -O machines.csv
