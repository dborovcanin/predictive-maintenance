docker run kubeflow-test python main.py \
  --telemetry="https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv" \
  --maintenance="https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_maint.csv" \
  --errors="https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_errors.csv" \
  --machines="https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_machines.csv" \
  --failures="https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_failures.csv" \
  --out="model"

