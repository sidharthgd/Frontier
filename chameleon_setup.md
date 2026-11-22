# Chameleon setup for Frontier RL training

Lease: `my-reservation` at **CHI@TACC**, 1 × `compute_zen3` node (CPU only, no GPU).
It ends **2026-11-23 02:55 UTC**. The node's disk is wiped when the lease ends, so copy results back before then (step 6).

- [Part A: launching and setting up the server](#part-a-launching-and-setting-up-the-server) (lease owner)
- [Part B: getting a partner's SSH key authorized](#part-b-getting-a-partners-ssh-key-authorized) (partner + lease owner)

---

## Part A: launching and setting up the server

### 0. Laptop prerequisites (once)

Run these from the Frontier repo root with the venv active.

```bash
source .venv/bin/activate
# openstacksdk 4.23.0 ships without its config JSON files; 4.22.0 works
pip install "openstacksdk==4.22.0"
export OS_CLOUD=tacc        # clouds.yaml lists 7 sites; pick CHI@TACC
```

### 1. Register your SSH key and open port 22 (once per site)

```bash
openstack keypair create --public-key ~/.ssh/id_ed25519.pub mykey
```

### 2. Launch the instance on the reservation

```bash
openstack server create frontier-train \
  --image CC-Ubuntu24.04 \
  --flavor baremetal \
  --network sharednet1 \
  --key-name mykey \
  --hint reservation=8529fe6d-52c3-4f1d-91da-0b6b5e282c86

# Bare metal takes ~10–15 min to boot. Repeat until the status is ACTIVE:
openstack server show frontier-train -c status -f value
```

`--key-name` is required. Without it nobody can log in, and you would have to delete and relaunch the instance.

### 3. Attach a floating IP

```bash
IP=$(openstack floating ip create public -f value -c floating_ip_address)
openstack server add floating ip frontier-train "$IP"
echo "$IP"          # share this with your partner
ssh cc@"$IP"        # uses ~/.ssh/id_ed25519
```

### 4. Copy the repo, data and predictor cache from the laptop

The workload CSVs and `cache/` are gitignored, so rsync the working tree directly instead of cloning from GitHub. Run on the **laptop**, from the repo root:

```bash
# Pin the exact package versions (the cache .pkl files need the same scikit-learn).
# Drop the editable frontier line and torch; both are installed separately on the node.
pip freeze | grep -vE '^-e |^torch==' > requirements-lock.txt

rsync -avz --progress \
  --exclude .venv --exclude '__pycache__' --exclude .pytest_cache --exclude rl_runs \
  ./ cc@"$IP":~/Frontier/
```

That transfers about 6 GB, most of it `cache/`. Without `cache/` the simulator still runs, but it retrains the predictors on the first run, which is slow.

### 5. Build the environment on the node

Run on the **node** (`ssh cc@$IP`):

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential zsh tmux htop rsync

cd ~/Frontier
python3 -m venv .venv                       # Ubuntu 24.04 ships Python 3.12
.venv/bin/pip install --upgrade pip
# CPU-only torch: the node has no GPU, and this skips the multi-GB CUDA wheels
.venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps

# Sanity checks
export PYTHONPATH=$PWD
.venv/bin/python -c "import frontier, torch, sklearn; print('ok', torch.__version__, sklearn.__version__)"
nproc                                       # use this to raise num_workers in configs/rl/*.yaml
```

### 6. Run training and copy the results back

The `scripts/rl/*.sh` launchers call `caffeinate`, which only exists on macOS. On Linux, run inside `tmux` so the job survives SSH disconnects:

```bash
# On the node
tmux new -s train
cd ~/Frontier && mkdir -p rl_runs
zsh scripts/rl/run_frontier_concurrency.sh 2>&1 | tee rl_runs/frontier_concurrency.log
# Detach: Ctrl-b then d    Reattach later: tmux attach -t train
```

```bash
# On the laptop: copy results back regularly, and before 2026-11-23 02:55 UTC
rsync -avz cc@"$IP":~/Frontier/rl_runs/ ./rl_runs/
```

---

## Part B: getting a partner's SSH key authorized

Your partner doesn't need a Chameleon account or OpenStack access to SSH in. They need an SSH key, and the lease owner adds the key's public half to the node.

### B1. Partner: create a key (skip if you already have `~/.ssh/id_ed25519.pub`)

```bash
ssh-keygen -t ed25519 -C "frontier-chameleon"
# Press Enter to accept the default path; setting a passphrase is recommended
```

### B2. Partner: send the PUBLIC key to the lease owner

```bash
cat ~/.ssh/id_ed25519.pub
```

Paste the single line it prints (it starts with `ssh-ed25519`) into a message to the lease owner. Only send the `.pub` file. Never share `~/.ssh/id_ed25519` (no `.pub`), which is the private key.

### B3. Lease owner: add the partner's key on the node

```bash
ssh cc@"$IP"
# Paste the partner's line between the quotes:
echo 'ssh-ed25519 AAAA...partner-key... frontier-chameleon' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### B4. Partner: connect

```bash
ssh cc@<FLOATING_IP>
tmux attach -t train     # view the running training job
```

Optional: add a shortcut to `~/.ssh/config`:

```
Host frontier-chameleon
    HostName <FLOATING_IP>
    User cc
    IdentityFile ~/.ssh/id_ed25519
```

Then connect with `ssh frontier-chameleon`, and copy results with `rsync -avz frontier-chameleon:~/Frontier/rl_runs/ ./rl_runs/`.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `defaults.json: No such file or directory` | Broken openstacksdk 4.23.0. Run `pip install "openstacksdk==4.22.0"` |
| `Missing value auth-url` | No cloud selected. Run `export OS_CLOUD=tacc` or pass `--os-cloud tacc` |
| `ssh` hangs / times out | Port 22 is blocked. Check that `allow-ssh` is attached: `openstack server show frontier-train -c security_groups` |
| `Permission denied (publickey)` | Wrong key, or the key isn't in `~/.ssh/authorized_keys` on the node. Try `ssh -i ~/.ssh/id_ed25519 cc@$IP` |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` after a relaunch | The IP was reused by a new instance. Run `ssh-keygen -R <FLOATING_IP>` |
| Server stuck in `ERROR` | `openstack server delete frontier-train`, then repeat step 2 |
