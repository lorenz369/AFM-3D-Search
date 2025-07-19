# git clone https://github.com/lorenz369/AFM-3D-Search.git --branch refactor/rerun

curl -LsSf https://astral.sh/uv/install.sh | sh

source .bashrc

cd AFM-3D-Search

uv venv .rr_env -p 3.11

source .rr_env/bin/activate

uv pip install -r environments/rerun_requirements.txt
