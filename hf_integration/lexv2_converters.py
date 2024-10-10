"""
python lexv2_converters.py

Convert Lex JSON into HF and  HF to Lex JSON

https://docs.aws.amazon.com/lexv2/latest/dg/import-export-format.html

"""
# *********************************************************************************************************************

# standard imports
import zipfile
import os
import shutil
import json
import datetime

# 3rd party imports

# custom imports
import humanfirst

# https://docs.aws.amazon.com/lexv2/latest/dg/how-languages.html
LEXV2_LOCALES =  {
    "ar_AE": "Gulf Arabic (United Arab Emirates)",
    "ca_ES": "Catalan (Spain)",
    "de_AT": "German (Austria)",
    "de_DE": "German (Germany)",
    "en_AU": "English (Australia)",
    "en_GB": "English (UK)",
    "en_IN": "English (India)",
    "en_US": "English (US)",
    "en_ZA": "English (South Africa)",
    "es_419": "Spanish (Latin America)",
    "es_ES": "Spanish (Spain)",
    "es_US": "Spanish (US)",
    "fi_FI": "Finnish (Finland)",
    "fr_CA": "French (Canada)",
    "fr_FR": "French (France)",
    "hi_IN": "Hindi (India)",
    "it_IT": "Italian (Italy)",
    "ja_JP": "Japanese (Japan)",
    "ko_KR": "Korean (Korea)",
    "nl_NL": "Dutch (The Netherlands)",
    "no_NO": "Norwegian (Norway)",
    "pl_PL": "Polish (Poland)",
    "pt_BR": "Portuguese (Brazil)",
    "pt_PT": "Portuguese (Portugal)",
    "sv_SE": "Swedish (Sweden)",
    "zh_CN": "Mandarin (PRC)",
    "zh_HK": "Cantonese (Hong Kong)"
}

class lexv2_to_hf_converter:
    def lexv2_to_hf_process(
            self,
            zip_file_full_name_inc_ext: str,
            tempdir: str,
            delimiter: str = "/"):
        """Assumes a zip in the tempdir to workwith
        Converts the zip into hf_format"""
    
        # extract to workdir
        workdir = _extract_zip_to_tempdirdir(tempdir,zip_file_full_name_inc_ext)

        # check manifest
        bot_name = _check_manifest(workdir)
        bot_root = os.path.join(workdir,bot_name)

        # get a HFWorkspace object to populate
        hf_workspace = humanfirst.objects.HFWorkspace()

        # load_time
        load_time = datetime.datetime.now()

        list_locales = os.listdir(os.path.join(bot_root,"BotLocales"))
        print(f'Locales: {list_locales}')

        # cycle through creating tags for each locale and then looking for intents
        for locale in list_locales:

            # create a tag to be able to create back from HumanFirst
            locale_tag = hf_workspace.tag(tag=locale)
            
            # List the files
            intents_path = os.path.join(bot_root,"BotLocales",locale,"Intents")
            list_intents = os.listdir(intents_path)

            # Check with have the BotLocale.json file
            # This has things like the intent detection accuracy in it.
            assert os.path.isfile(os.path.join(bot_root,"BotLocales",locale,"BotLocale.json"))

            # Cycle through the intents
            for intent in list_intents:
                # read the json
                with open(os.path.join(intents_path,intent,"Intent.json")) as intent_in:
                    dict_intent = json.load(intent_in)
                hier_intent = list(str(dict_intent["name"]).split(delimiter))
                
                # map name, id - no metadata - no tag - that phrase level (assuming each has same name?)
                hf_intent = hf_workspace.intent(name_or_hier=hier_intent,id=dict_intent["identifier"])

                # add examples if they exist
                i = 0
                if "sampleUtterances" in dict_intent and dict_intent["sampleUtterances"] is not None:
                    for sample in dict_intent["sampleUtterances"]:
                        example = hf_workspace.example(
                            text=sample["utterance"],
                            id = humanfirst.objects.hash_string(sample["utterance"],prefix="example"),
                            created_at = load_time,
                            intents = [hf_intent],
                            tags=[locale_tag]
                        )
                        hf_workspace.add_example(example)
                        i = i + 1
                print(f"Finished {locale} {intent} samples: {i}")
        
        # output
        output_filename = os.path.join(tempdir,f'{bot_name}.json')
        with open(output_filename,'w',encoding='utf8') as file_out:
            hf_workspace.write_json(file_out)

class hf_to_lexv2_converter:
    def hf_to_lexv2_process(self,
                        zip_file_full_name_inc_ext: dict,
                        hf_workspace: humanfirst.objects.HFWorkspace,
                        tempdir: str = "en-us",
                        delimiter: str = "/") -> None:
        """Process HF to CLU conversion
        Takes a Zip and merges the intents and phrases into it."""

        # extract to workdir
        workdir = _extract_zip_to_tempdirdir(tempdir,zip_file_full_name_inc_ext)

        # check manifest
        bot_name = _check_manifest(workdir)

        # get tag_names
        locale_tag_names = []
        for tag_id in hf_workspace.tags.keys():
            tag = hf_workspace.tags[tag_id]
            assert isinstance(tag,humanfirst.objects.HFTag)
            if tag.name in LEXV2_LOCALES:
                locale_tag_names.append(tag.name)
        print(locale_tag_names)

        # get locales
        list_locales = os.listdir(os.path.join(workdir,bot_name,"BotLocales"))

        # check we have tags for all the locales
        for locale in list_locales:
            assert locale in locale_tag_names

        # check we have locales for all the tags
        for tag in locale_tag_names:
            assert tag in list_locales

        # going to make an assumption that intents in HumanFirst with no training - don't update LexV2
        # this means that parents if correctly stored without training shouldn't get created.
        # if they do this mistraining can be corrected.

        # gives a FQN index ID: FQN - lets us match up to Lexv2
        dict_id_fqn = hf_workspace.get_intent_index(delimiter=delimiter)

        # for each locale and each intent check whether we have training.
        for locale in list_locales:
            for intent_id in dict_id_fqn:
                sample_utterances = _get_intent_locale_examples(hf_workspace=hf_workspace, 
                                                                intent_name=dict_id_fqn[intent_id],
                                                                locale="en_GB")
                if len(sample_utterances) > 0:
                    print("blah")


        # this whole thing needs better thinking about with the fully qualified intent name
        





        # have exactly the number of locales as tag_locales
        # will cycle through each locale updating only the data tagged with that locale into the intents
        # for tag in locale_tag_names:


def _search_for_intent_id(hf_workspace: humanfirst.objects.HFWorkspace, intent_name_flattened_in_lex: str):
    """Take a flattened name and compare it against the fully qualified name of every intent to search for it"""
    for intent_name in hf_workspace.intents:
        intent_id = hf_workspace.intents[intent_name].id
        fqn_intent_name = hf_workspace.get_fully_qualified_intent_name(intent_id)
        if fqn_intent_name == intent_name_flattened_in_lex:
            return intent_id

def _get_intent_locale_examples(hf_workspace: humanfirst.objects.HFWorkspace, intent_name: str, locale: str) -> list:
    """Filter out the training for the intent and the locale
    Returns empty list if no training"""
    
    # work out intent ID 
    intent_id = _search_for_intent_id(hf_workspace=hf_workspace,intent_name_flattened_in_lex="injured_at_the_zoo")

    # Filter out training
    sample_utterances = []
    for example_id in hf_workspace.examples:
        example = hf_workspace.examples[example_id]
        for intent in example.intents:
            if intent.intent_id == intent_id:
                for tag in example.tags:
                    if tag.name == locale:
                        dict_utterance = {
                            "utterance": example.text
                        }
                        sample_utterances.append(dict_utterance)
                        break
    return(sample_utterances)
    
    


def _create_intent(self, locale: str, intent_name: str):
    return ""

def _update_intent(self, locale: str, intent_name: str):
    return ""

def _delete_intent(self, locale: str, intent_name: str):
    return ""

def _extract_zip_to_tempdirdir(tempdir: str, zip_file_full_name_inc_ext: str) -> str:
    """extract the zip file to the temporary directory named same as zip so can tell only one bot"""
    
    # extract zip
    zip = zipfile.ZipFile(os.path.join(tempdir, zip_file_full_name_inc_ext), 'r')
    zip_file_name = zip_file_full_name_inc_ext.replace('.zip','')

    # it is of this shape        
    # <workdir>/<botname>/BotLocales/<lang_code>/intents
    # lost_at_zoo_virginia-DRAFT-1IARL6UWPR-LexJson/lost_at_zoo_virginia/BotLocales/en_GB/intents
    # create a tag for each Locale then merge and split on that.

    # check if workdir exists and delete if it does
    workdir = os.path.join(tempdir,zip_file_name)
    if os.path.isdir(workdir):
        shutil.rmtree(workdir, ignore_errors=True)
    zip.extractall(workdir)
    zip.close()

    return workdir

def _check_manifest(workdir: str) -> str:
    """"check we got a manifest and one dir and establish bot_name
    Check we have a Bot.json,"""

    # Check we have a dir and a file with the Manifest in 
    list_root_files = os.listdir(workdir)
    assert len(list_root_files) == 2
    list_root_files.remove("Manifest.json")

    # work out bot root and cehck it
    bot_root = os.path.join(workdir,list_root_files[0])
    assert os.path.isdir(bot_root)

    # check bot root has the Bot.json
    list_bot_root_files = os.listdir(bot_root)
    assert len(list_bot_root_files) == 2
    list_bot_root_files.remove("Bot.json")
    lexv2_bot_dict = _get_bot_json(bot_root)
    
    # Check the name matches the directory
    assert lexv2_bot_dict["name"] == list_root_files[0]
    bot_name = list_root_files[0]
    return bot_name

def _get_bot_json(bot_root: str) -> dict:
    """Extract the bot json from an expanded zipfile"""
    with open(os.path.join(bot_root,"Bot.json")) as bot_json_in:
        return json.load(bot_json_in)


class lexv2_converter(lexv2_to_hf_converter, hf_to_lexv2_converter):

    """Handles HF to Lexv2 and CLU to Lexv2"""