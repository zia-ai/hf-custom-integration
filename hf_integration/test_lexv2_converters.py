# Work In Progress
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)

import hf_integration.lexv2_converters
import humanfirst
import json

def test_lexv2_to_hf_conversation():
    """Converts Lex zip to hf JSON"""
    converter = hf_integration.lexv2_converters.lexv2_to_hf_converter()
    print(converter.lexv2_to_hf_process("lost_at_zoo_virginia-DRAFT-1IARL6UWPR-LexJson.zip","./data"))

def test_hf_to_lexv2_conversation():
    """Converts Lex zip to hf JSON"""
    test_file = "./data/lost_at_zoo_virginia.json"
    with open(test_file,mode='r',encoding='utf8') as file_in:
        hf_workspace = humanfirst.objects.HFWorkspace.from_json(json.load(file_in),delimiter="/")
    converter = hf_integration.lexv2_converters.hf_to_lexv2_converter()
    print(converter.hf_to_lexv2_process(
        zip_file_full_name_inc_ext="lost_at_zoo_virginia-DRAFT-1IARL6UWPR-LexJson.zip",
        hf_workspace=hf_workspace,
        tempdir="./data"))