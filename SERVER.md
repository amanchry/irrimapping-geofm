# Running the project on the UT EEMCS HPC cluster

How this project is set up and run on the University of Twente EEMCS HPC cluster (Slurm). 

| | |
| --- | --- |
| Account | `s3677133` (UT AD account, always lowercase) |
| Head nodes | `hpc-head1.ewi.utwente.nl`, `hpc-head2.ewi.utwente.nl` (login, edit, submit only) |
| Project on the server | `~/irrimapping-geofm` (= `/home/s3677133/irrimapping-geofm`) |
| Conda environment | `condaml` |
| Partitions | GPU: `itc-gpu,main-gpu` · CPU: `itc-cpu,main-cpu` |
| GPUs | NVIDIA L40 / L40S (46 GB) and A100 (40 GB); driver 615, CUDA 12/13 |
| Job scripts and logs | `jobs/` in the project (`jobs/*.sbatch`, `jobs/logs/`); not in git (§3.1) |
| Home storage | 1 TB (`/home/s3677133`); also `/projects/…`, `/datasets/…`; `/local` = node-local scratch |
| Docs (VPN only) | [wiki](https://hpc.wiki.utwente.nl/eemcs-hpc) · [Slurm](https://hpc.wiki.utwente.nl/slurm:start) · [partitions](https://hpc.wiki.utwente.nl/eemcs-hpc:specifics#partitions) · [status page](http://hpc-status.ewi.utwente.nl/slurm/) |

**Rules from HPC staff:** never log in to compute nodes directly; never run heavy work on the head node; run all
computation through `sbatch` (preferred) or `srun`.

---

> **Path change (12 Nov 2026, local copy only):** outputs are now per region: `outputs/morocco/{dataset, study,
> analysis}` and `outputs/utah/dataset`. The server copy below still has the old layout (`outputs/dataset`,
> `outputs/study`, `outputs/analysis`); move it the same way before the next code sync, or the scripts will not
> find their inputs.

## 1. Connecting from off campus

### 1.1 eduVPN

The cluster, the wiki and the status page are reachable only from the campus network or over eduVPN.

1. Install the **eduVPN** app (Mac App Store / eduvpn.org).
2. Choose **Institute Access → University of Twente** and log in with the UT account.
   **Not** "Secure Internet": that profile gives general internet access only, and SSH to the cluster times out.
3. Connect. Check:

```bash
nc -vz -G 5 hpc-head1.ewi.utwente.nl 22      # must print "succeeded"
```

| Symptom | Cause |
| --- | --- |
| `Operation timed out` | eduVPN off, or the wrong profile (Secure Internet) |
| `Permission denied` | username not lowercase, wrong password, or key not installed |

### 1.2 SSH key (one-time)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_uthpc          # Enter twice: no passphrase
```

Add to `~/.ssh/config` on the Mac:

```
Host ut-hpc
    HostName hpc-head1.ewi.utwente.nl
    User s3677133
    IdentityFile ~/.ssh/id_ed25519_uthpc
    IdentitiesOnly yes
```

Copy the key to the cluster (asks for the UT password one last time), then log in without a password:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_uthpc.pub ut-hpc
ssh ut-hpc
```

VS Code: *Remote Explorer → SSH → ut-hpc* uses the same config entry (editing only; runs still go through Slurm).

---

## 2. Setting up the code repository

### 2.1 Clone

```bash
ssh ut-hpc
cd ~
git clone https://github.com/amanchry/irrimapping-geofm.git irrimapping-geofm
```

Update later with `cd ~/irrimapping-geofm && git pull`. (An older copy from 2025-10-14 is kept in
`~/irrimapping-geofm_old_oct14`; the old `geofm` environment belongs to it.)

### 2.2 Conda environment `condaml`

Conda is provided centrally: `/software/miniconda3/25.7` (loaded by `~/.bashrc`, or
`module load miniconda3/25.7`). Two pitfalls had to be worked around:

| Problem | Fix |
| --- | --- |
| The system `.condarc` adds Anaconda's `defaults` channels, which stop with `CondaToSNonInteractiveError` (Terms of Service) | Redirect `defaults` to conda-forge for the command: `export CONDA_DEFAULT_CHANNELS=https://conda.anaconda.org/conda-forge` (no config change) |
| The head node has no GPU, so conda-forge installs the **CPU** build of PyTorch | `export CONDA_OVERRIDE_CUDA=12.8` and request the CUDA builds explicitly: `"pytorch=*=cuda*" "torchvision=*=cuda*"` |

Full setup (as run on 2026-11-06; scripts kept in `~/setup_condaml.sh` and `~/fix_torch_cuda.sh`):

```bash
source /software/miniconda3/25.7/etc/profile.d/conda.sh
export CONDA_DEFAULT_CHANNELS=https://conda.anaconda.org/conda-forge
export CONDA_OVERRIDE_CUDA=12.8
cd ~/irrimapping-geofm

# environment.yml + "nodefaults" (server-side copy; the repo file stays unchanged)
sed "s/^  - conda-forge$/  - conda-forge\n  - nodefaults/" environment.yml > ~/condaml_env.yml
conda env create -f ~/condaml_env.yml

conda activate condaml
conda install -y "pytorch=*=cuda*" "torchvision=*=cuda*"     # GPU build of PyTorch
pip install --no-deps git+https://github.com/Clay-foundation/model.git   # Clay
pip install mmengine                                                     # AgriFM
```

Run setup on the head node in the background (`nohup … > log 2>&1 &`), since it takes 10–20 min.

Result: PyTorch 2.13.0, CUDA build 12.9, GPU visible on the L40 nodes.

Check on a GPU node (a few seconds):

```bash
conda activate condaml
srun --partition=itc-gpu,main-gpu --gres=gpu:1 --mem=8G --time=00:05:00 \
     python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2.3 Data (not in git)

Copied from the Mac with `rsync` (run **on the Mac**, from the project folder; re-running resumes and skips files
already copied):

| Folder | Size | How |
| --- | --- | --- |
| `data/` (field survey, admin, worldcover) | 0.3 GB | copied |
| `external/AgriFM` | 8 MB | copied |
| `outputs/dataset/01_labels`, `02_folds`, `outputs/study`, `outputs/analysis` | 0.1 GB | copied |
| `outputs/dataset/03_tiles_128px/agyear` (tile store, without `views/`) | 11 GB | copied (see below) |
| `outputs/dataset/03_tiles_128px/agyear/views` | 12 GB | **not copied**: rebuilt on the server with `ml_dataset_prep/04_make_views.py` (reads only the tile store, no Earth Engine) |
| `checkpoints/` (model weights) | 7.8 GB | **not copied**: downloaded automatically on the first run of each embedding script |
| `secrets/` (Earth Engine key) | — | **not copied**: only needed to rebuild tiles with `03_build_tiles.py` |

```bash
# small inputs
rsync -az --relative --exclude='.DS_Store' data external outputs/dataset/01_labels outputs/dataset/02_folds \
      outputs/study outputs/analysis ut-hpc:irrimapping-geofm/

# tiles: GeoTIFFs are already compressed -> no -z
caffeinate -i rsync -a --relative --exclude=.DS_Store --exclude=views --stats \
      outputs/dataset/03_tiles_128px ut-hpc:irrimapping-geofm/
```

Then build the views on the server as a CPU job (§3.2, `jobs/make_views.sbatch`; 16 min, 12 GB):

```bash
cd ~/irrimapping-geofm && sbatch jobs/make_views.sbatch
```

Notes: zipping first gives nothing for the GeoTIFFs (< 1 % smaller) and needs extra disk. The upload rate over
eduVPN is about 5 MB/s (≈ 40 min for 11 GB). `caffeinate` keeps the Mac awake; eduVPN must stay connected.

Results back to the Mac (on the Mac):

```bash
rsync -az ut-hpc:irrimapping-geofm/experiments/E02_frozen_pca/results/ experiments/E02_frozen_pca/results/
```

---

## 3. Running jobs

All commands in this section run **on the server** (`ssh ut-hpc` first); `sbatch`, `squeue` etc. do not exist on
the Mac.

### 3.1 The `jobs/` folder

Job scripts and their logs live in the project, in `jobs/`, which is kept out of git:

| Where | How it is ignored |
| --- | --- |
| Mac | `jobs/` line in `.gitignore` |
| Server | `jobs/` line in `.git/info/exclude` (local to the clone, so `git pull` never conflicts) |

```
jobs/
├── make_views.sbatch        CPU · ml_dataset_prep/04_make_views.py
├── embed_clay.sbatch        GPU · models_run/02a_embed_clay.py
├── embed_prithvi.sbatch     GPU · models_run/02b_embed_prithvi.py
├── embed_terramind.sbatch   GPU · models_run/02c_embed_terramind.py
├── embed_agrifm.sbatch      GPU · models_run/02d_embed_agrifm.py
└── logs/                    <job-name>-<job-id>.log, written by Slurm
```

The server copy is the one that runs; the Mac copy is a reference. Keep them in sync with (on the Mac):

```bash
rsync -a --exclude logs ut-hpc:irrimapping-geofm/jobs/ jobs/      # server -> Mac
rsync -a --exclude logs jobs/ ut-hpc:irrimapping-geofm/jobs/      # Mac -> server
```

### 3.2 Job script (GPU)

`jobs/embed_clay.sbatch`; the other embedding scripts differ only in `--job-name` and the `python` line:

```bash
#!/bin/bash
#SBATCH --job-name=embed_clay
#SBATCH --partition=itc-gpu,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00                  # default 1 day if omitted; no maximum
#SBATCH --output=/home/s3677133/irrimapping-geofm/jobs/logs/%x-%j.log

# compute nodes reach the internet (weights, Earth Engine, pip) only through the UT proxy
export http_proxy=http://proxy.utwente.nl:3128 https_proxy=http://proxy.utwente.nl:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy

source /software/miniconda3/25.7/etc/profile.d/conda.sh
conda activate condaml
cd ~/irrimapping-geofm                    # scripts are run from the project folder

nvidia-smi --query-gpu=name,memory.total --format=csv
python -u models_run/02a_embed_clay.py    # -u: unbuffered, so the log fills while it runs
```

CPU job (`jobs/make_views.sbatch`): `--partition=itc-cpu,main-cpu`, no `--gres` line, no proxy needed,
`--cpus-per-task=2 --mem=16G` (the dataset scripts use one core).

New step: copy a script, change `--job-name` and the `python` line. Settings stay at the top of each project
script (no command-line arguments), so edit the script (e.g. `EXPERIMENT = "E02_frozen_pca"`,
`QUICK_TEST: true` in the config) and then submit. `jobs/logs/` must exist before submitting (Slurm does not create
it): `mkdir -p jobs/logs`.

### 3.3 Submitting

```bash
cd ~/irrimapping-geofm
sbatch jobs/embed_clay.sbatch                  # one job
for m in clay prithvi terramind agrifm; do sbatch jobs/embed_$m.sbatch; done   # all four in parallel
```

The four embedding scripts are independent (own weights in `checkpoints/<model>/`, own parts folder and output
file; Prithvi and TerraMind only read the same `chips_p16`), so they run in parallel, one GPU each. Run times on
2026-11-06 (8,781 parcels, first run including the weights download):

| Step | Job | Node / GPU | Wall time | Output |
| --- | --- | --- | --- | --- |
| Views | `make_views` | CPU (hpc-node32) | 16 min | `outputs/dataset/03_tiles_128px/agyear/views/` (12 GB) |
| Clay | `embed_clay` | L40S | 4 min (2.6 min embedding) | `outputs/study/02_embeddings/clay_agyear_tiles.npz` |
| Prithvi | `embed_prithvi` | A100 | 3.7 min | `…/prithvi_agyear_tiles.npz` |
| TerraMind | `embed_terramind` | A100 | 6.3 min (5.4 min embedding) | `…/terramind_agyear_tiles.npz` |
| AgriFM | `embed_agrifm` | GPU (ctit088) | 3.9 min | `…/agrifm_agyear_tiles.npz` |

Embedding scripts resume from `outputs/study/02_embeddings/_parts_<model>_agyear_tiles/`: a job stopped by its time
limit can be resubmitted. To recompute from scratch, move that folder aside first. (The partial parts made on the
Mac are kept as `_parts_<model>_agyear_tiles_from_mac`.)

### 3.4 Monitoring

| Command | What |
| --- | --- |
| `squeue -u s3677133` | own jobs (`PD` = waiting, `R` = running) |
| `tail -f jobs/logs/embed_clay-<jobid>.log` | live output |
| `scancel <jobid>` | cancel a job |
| `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` | state, run time and memory after the job |
| `sinfo -s` | partitions and free nodes |
| [hpc-status.ewi.utwente.nl/slurm](http://hpc-status.ewi.utwente.nl/slurm/) | cluster load (VPN) |

### 3.5 Short interactive tests

For a quick test only (not for long runs):

```bash
srun --partition=itc-gpu,main-gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
conda activate condaml && cd ~/irrimapping-geofm
```

### 3.6 JupyterLab on a GPU node

`~/myjupyter/jupyterlab.sbatch` starts JupyterLab on a GPU node (6 h, 8 CPUs, 64 GB). It still activates the old
`geofm` environment; change it to `conda activate condaml` for this project.

```bash
jup        # alias: sbatch ~/myjupyter/jupyterlab.sbatch
jupurl     # alias: prints the URL from the newest log, e.g. http://hpc-node29.eemcs.utwente.nl:8xxx/lab
```

Open the URL in the browser while on eduVPN; `scancel` the job when done so the GPU is freed.
