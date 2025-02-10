"""
python clu_import.py

A way of testing locally conversion of a passed JSON into HF format

"""
# ******************************************************************************************************************120

# standard imports
import json

# 3rd party imports
import click

# custom imports
import hf_integration.clu_converters

@click.command()
@click.option('-f', '--filename', type=str, required=True, help='Input CLU JSON')
def main(filename: str) -> None: # pylint: disable=unused-argument
    """Main Function"""
    
    # read the json
    with open(filename,mode="r",encoding="utf8") as file_in:
        clu_dict = json.load(file_in)
            
    converter = hf_integration.clu_converters.clu_to_hf_converter()
    hf_dict = converter.clu_to_hf_process(clu_json=clu_dict,delimiter="-",language="nl")
    
    # write the file
    output_filename = filename.replace(".json","_hf_output.json")
    assert output_filename != filename
    with open(output_filename,mode="w",encoding="utf8") as file_out:
        json.dump(hf_dict,file_out,indent=4)
    
    

if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
