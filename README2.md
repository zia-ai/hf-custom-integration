Started with converters
Local tests


Create a custom integration generate certifacte - all in README.md

workspace_example is Matt
sworkspace_generic to clone
--model necessary to run other things - without that just start with import and export.


Expose IP address ngrok.

# make the venv
python -m venv venv

# 
source venv/bin/activate

python -m pip install poetry

# export your dependencies in the requirements.txt format using poetry
poetry export --without-hashes -f requirements.txt -o requirements.txt

# create your venv like you did on your example (you may want to upgrade pip/wheel/setuptools first)
python3 -m venv venv && . venv/bin/activate

# then install the dependencies
pip install --no-cache-dir --no-deps -r requirements.txt

# then you install your own project
pip install .


Think we should think more about 
- venv setup with poetry lock management
Folder structure
- put each integration in one
Testing pytest setup - each test should have the integration name in test_lexv2