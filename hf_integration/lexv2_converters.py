"""
python lexv2_converters.py

Convert Lex JSON into HF and  HF to Lex JSON

"""
# *********************************************************************************************************************

# standard imports
import zipfile
import os

# 3rd party imports

# custom imports
import humanfirst

class lexv2_to_hf_converter:
    def lexv2_to_hf_process(
            self,
            lex_zip_location: str,
            tempdir: str,) # WTF will this be
    
        zip = zipfile.ZipFile(lex_zip_location, 'r')
        zip.extractall(tempdir)
        zip.close()
        # Perform operations on the ZIP file
        
        # get a HFWorkspace object to populate
        hf_workspace = humanfirst.objects.HFWorkspace()

        os.listdir(tempdir)

        return {}
  
class hf_to_lexv2_converter:
    def hf_to_lexv2_process(self,
                        hf_json: dict,
                        lex_zip: dict,
                        language: str = "en-us",
                        delimiter: str = "-",
                        skip: bool = False) -> None:
        """Process HF to CLU conversion"""


class lexv2_converter(lexv2_to_hf_converter, hf_to_lexv2_converter):
    """Handles HF to Lexv2 and CLU to Lexv2"""