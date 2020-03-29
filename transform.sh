#docker run kubeflow-test python main.py \
#  --telemetry="https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv" \
#  --maintenance="https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_maint.csv" \
#  --errors="https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_errors.csv" \
#  --machines="https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_machines.csv" \
#  --failures="https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_failures.csv" \
#  --out="model"
#
LOC=./dataset
python transform.py \
  --telemetry="$LOC/telemetry.csv" \
  --maintenance="$LOC/maintenance.csv" \
  --errors="$LOC/errors.csv" \
  --machines="$LOC/machines.csv" \
  --failures="$LOC/failures.csv" \
  --out="$LOC/dataset.csv" \
  --from_date="2015-06-15 01:00:00" \
  --to_date="2015-10-15 01:00:00" \
