# Work In Progress
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)

# standard imports
import os
import json
import shutil

# 3rd party imports
import pytest

# custom imports
import hf_integration.lexv2_apis

TEMPDIR = 'data'
TEST_DATA_DIR = 'test_data/lexv2'
TEST_BOT_FILE = "lost_at_zoo_virginia-DRAFT-1IARL6UWPR-LexJson.zip"

@pytest.fixture
def lexv2():
    """Sets up a copy of the api for test"""

    # declare client
    lexv2 = hf_integration.lexv2_apis.lexv2_apis()
    yield lexv2  # Provide the data to the test
    # Teardown: Clean up resources (if any) after the test

@pytest.fixture
def botzip():
    """Sets up a copy of the test bot_zip"""

    fq_test_file = os.path.join(TEMPDIR,TEST_BOT_FILE)
    fq_source_test_file = os.path.join(TEST_DATA_DIR,TEST_BOT_FILE)

    # check if need to clearup
    if os.path.isfile(fq_test_file):
        os.remove(fq_test_file)
    
    # copy file
    shutil.copy(fq_source_test_file,fq_test_file)
    
    botzip = True       
    yield botzip  # Provide the data to the test
    
    # check if need to clearup
    if os.path.isfile(fq_test_file):
        os.remove(fq_test_file)
   
def test_lexv2_get_iam_user(lexv2: hf_integration.lexv2_apis.lexv2_apis):
    """Tests getting the IAM user"""
    print(lexv2.get_iam_user())   

def test_lexv2_create_bot(lexv2: hf_integration.lexv2_apis.lexv2_apis):
    """Tests creating a basic skeleton of a bot"""
    print(json.dumps(lexv2.create_bot(bot_name = "Testing1234")))

def test_lexv2_delete_bot(lexv2: hf_integration.lexv2_apis.lexv2_apis):
    """Tests deleting a bot"""
    print(json.dumps(lexv2.delete_bot(bot_id="3PGNCR3OIK")))

def test_lexv2_list_bots(lexv2: hf_integration.lexv2_apis.lexv2_apis):
    """Tests instantiation and listing bots"""
    
    # test paging (assumes there are least two in the environment to test)
    # TODO: setup creating the bots
    bot_list1 = lexv2.list_bots(max_results=1)
    bot_list2 = lexv2.list_bots()
    assert bot_list1 == bot_list2
    
    # check json serialises
    print(json.dumps(bot_list1,indent=2))

def test_lexv2_import_bot(lexv2, botzip):
    print(json.dumps(lexv2.import_bot(zip_file_name=TEST_BOT_FILE,
                                      TEMPDIR=TEMPDIR)))
    
    
    



