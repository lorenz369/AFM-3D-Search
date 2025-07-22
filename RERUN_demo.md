## Recorded rerun stuff:

## Set up server:
### Install uv if necessary:
curl -LsSf https://astral.sh/uv/install.sh | sh

### Build env
source .bashrc

uv venv .rr_env -p 3.11

source .rr_env/bin/activate

uv pip install -r environments/rerun_requirements.txt

### Start different interactive demos

#### TUM Room
python run_interactive.py --config-name tum_room server.port=9000
python run_interactive.py --config-name tum_room rerun_save_enabled=true scripted_run.enabled=true
ssh -L 9000:localhost:9000 runpod_google
ssh -L 9000:localhost:9000 atcremers

#### Bude Tim
python run_interactive.py --config-name bude_tim server.port=9001
python run_interactive.py --config-name bude_tim rerun_save_enabled=true scripted_run.enabled=true
ssh -L 9001:localhost:9001 runpod_google
ssh -L 9001:localhost:9001 atcremers

#### Bude (Tim alt)
python run_interactive.py --config-name bude server.port=9002
python run_interactive.py --config-name bude rerun_save_enabled=true scripted_run.enabled=true
ssh -L 9002:localhost:9002 runpod_google
ssh -L 9002:localhost:9002 atcremers

#### Bude Marco
python run_interactive.py --config-name bude_marco server.port=9003
python run_interactive.py --config-name bude_marco rerun_save_enabled=true scripted_run.enabled=true
ssh -L 9003:localhost:9003 runpod_google
ssh -L 9003:localhost:9003 atcremers

#### Kueche
python run_interactive.py --config-name kueche server.port=9004
python run_interactive.py --config-name kueche rerun_save_enabled=true scripted_run.enabled=true
ssh -L 9004:localhost:9004 runpod_google
ssh -L 9004:localhost:9004 atcremers