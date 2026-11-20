#!/bin/bash
# STAC Sentinel-2 download (s2_download/02_download_s2_stac.py): runs NP parts in parallel inside one job, parts
# PART0 .. PART0+NP-1 of S2_N_PARTS, each with THREADS = 8 read threads. The job fails (exit 1) if any part fails,
# and the parts that failed are listed at the end of the job log; each part has its own log
# jobs/logs/<job name>-<job id>_p<part>.log.
#
# Memory: a part needs up to ~6 GB (a month with ~40 scenes of a 1024 x 1024 px block); 16 parts peaked at 95 of
# 96 GB, so keep NP <= 12 with --mem=96G, or give ~7 GB per part.
#
# Usage (from ~/irrimapping-geofm), e.g. all 1,307 Utah blocks in 32 parts, 4 jobs of 8 parts:
#   for P0 in 0 8 16 24; do sbatch --job-name=s2_ut_p$P0 -c 64 --mem=56G --time=12:00:00 \
#       --export=ALL,EXPERIMENT_REGION=utah,S2_N_PARTS=32,PART0=$P0,NP=8 s2_download/s2_dl.sbatch; done
#   only some parts again (finished tiles are skipped), e.g. parts 5, 6 and 21:
#       sbatch ... --export=ALL,EXPERIMENT_REGION=utah,S2_N_PARTS=32,PARTS="5 6 21" s2_download/s2_dl.sbatch
#SBATCH --job-name=s2_dl
#SBATCH --partition=itc-cpu,main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/s3677133/irrimapping-geofm/jobs/logs/%x-%j.log
export https_proxy=http://proxy.utwente.nl:3128 http_proxy=http://proxy.utwente.nl:3128
export HTTPS_PROXY=$https_proxy HTTP_PROXY=$http_proxy
export GDAL_HTTP_PROXY=proxy.utwente.nl:3128
source /software/miniconda3/25.7/etc/profile.d/conda.sh
conda activate s2dl   # own environment: stackstac needs pandas < 3 (condaml has pandas 3)
cd ~/irrimapping-geofm
NP=${NP:-1}; PART0=${PART0:-0}
PARTS=${PARTS:-$(seq -s " " $PART0 $((PART0 + NP - 1)))}
echo "$(date) host=$(hostname) region=$EXPERIMENT_REGION parts $PARTS of ${S2_N_PARTS:-1} test_blocks=${S2_TEST_BLOCKS:-0}"

PID=()                # process id per part number
for P in $PARTS; do
  S2_PART=$P python -u -W ignore s2_download/02_download_s2_stac.py > jobs/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_p${P}.log 2>&1 &
  PID[$P]=$!
done

FAILED=""
for P in $PARTS; do
  wait ${PID[$P]}; RC=$?
  if [ $RC -ne 0 ]; then
    FAILED="$FAILED $P"
    echo "$(date) part $P FAILED (exit $RC): see jobs/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_p${P}.log"
  fi
done
if [ -n "$FAILED" ]; then
  echo "$(date) parts failed:$FAILED"
  exit 1
fi
echo "$(date) all parts finished"
