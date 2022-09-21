#!/bin/sh
MY_PROJECT="My First Project"
MY_BUCKET="detiondx9006"
MY_REGION="us"

python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=c4/en \
  --data_dir=gs://$MY_BUCKET/tensorflow_datasets 
  --beam_pipeline_options="project=$MY_PROJECT,job_name=c4,staging_location=gs://$MY_BUCKET/binaries,temp_location=gs://$MY_BUCKET/temp,runner=DataflowRunner,requirements_file=/tmp/beam_requirements.txt,experiments=shuffle_mode=service,region=$MY_REGION"