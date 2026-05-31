python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$env:CONFIG_ENV="alpha_0"
python run.py