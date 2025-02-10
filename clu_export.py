"""
python clu_export.py

A way of testing locally conversion of a passed JSON into CLU format

"""
# ******************************************************************************************************************120

# standard imports
import json

# 3rd party imports
import click

# custom imports
import hf_integration.clu_converters

@click.command()
@click.option('-f', '--filename', type=str, required=True, help='Input HF JSON')
@click.option('-m', '--merge_filename', type=str, required=True, help='Input CLU to merge with')
def main(filename: str, merge_filename:str) -> None: # pylint: disable=unused-argument
    """Main Function"""
    
    # read the json
    with open(filename,mode="r",encoding="utf8") as file_in:
        hf_dict = json.load(file_in)
            
    with open(merge_filename,mode="r",encoding="utf8") as file_in_merge:
        clu_dict = json.load(file_in_merge)
    
    converter = hf_integration.clu_converters.hf_to_clu_converter()
    hf_dict = converter.hf_to_clu_process(hf_json=hf_dict,clu_json=clu_dict,delimiter="-",language="nl")
    
    # write the file
    output_filename = filename.replace(".json","_clu_output.json")
    assert output_filename != filename
    with open(output_filename,mode="w",encoding="utf8") as file_out:
        json.dump(hf_dict,file_out,indent=4)
    
    

if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
