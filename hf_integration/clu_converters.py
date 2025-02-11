"""
python clu_clu_to_hf_converter.py

Convert CLU JSON into HF and  HF to CLU JSON

"""
# *********************************************************************************************************************
# TODO: standardise if and when deepcopy required
# on the return - on the call - not both

# standard imports
import datetime
import warnings
import copy
import logging.config
import os
from datetime import datetime
import json

# 3rd party imports
import pandas
from pythonjsonlogger import jsonlogger

# custom imports
import humanfirst

# CLU name constants
TRAIN="Train"
TEST="Test"

# Sys entity mappers 
SYSTEM_ENTITY_MAPPER = {
    "DateTime": "SYS_DATE_TIME",
    "Quantity.Number": "SYS_NUMBER"
}
SYSTEM_ENTITY_REVERSER = {
    "SYS_DATE_TIME":"DateTime",
    "SYS_NUMBER":"Quantity.Number"
}


# locate where we are
here = os.path.abspath(os.path.dirname(__file__))

path_to_log_config_file = os.path.join(here,'config','logging.conf')

# Get the current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create the log file name with the current datetime
log_filename = f"log_{current_datetime}.log"

# Decide whether to save logs in a file or not
log_file_enable = os.environ.get("CI_LOG_FILE_ENABLE")

log_handler_list = []

if log_file_enable == "TRUE":
    log_handler_list.append('rotatingFileHandler')
elif log_file_enable == "FALSE" or log_file_enable is None:
    pass
else:
    raise RuntimeError("Incorrect CI_LOG_FILE_ENABLE value. Should be - 'TRUE', 'FALSE' or ''")

log_defaults = {}

# get log directory if going to save the logs
if log_file_enable == "TRUE":
    log_dir = os.path.join(here,"logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    path_to_save_log = os.path.join(log_dir,log_filename)
else:
    # avoid logging to a file
    path_to_save_log = '/dev/null'  # On Linux/MacOS, this discards logs (Windows: NUL) pylint:disable=invalid-name
log_defaults['CI_LOG_FILE_PATH'] = path_to_save_log


# Decide whether to print the logs in the console or not
log_console_enable = os.environ.get("CI_LOG_CONSOLE_ENABLE")

if log_console_enable == "TRUE":
    log_handler_list.append('consoleHandler')
elif log_console_enable == "FALSE" or log_console_enable is None:
    pass
else:
    raise RuntimeError("Incorrect CI_LOG_CONSOLE_ENABLE value. Should be - 'TRUE', 'FALSE' or ''")


if log_console_enable == "TRUE" and log_file_enable == "TRUE":
    raise RuntimeError("Custom integration supports either console logging or file logging but not both")
    # this is because of unable to override SSL errors logging configurations and able to only have them in either console or log file 


if log_handler_list:
    log_defaults['CI_LOG_HANDLER'] = ",".join(log_handler_list)
else:
    log_defaults['CI_LOG_HANDLER'] = "nullHandler"


# Set log levels
log_level = os.environ.get("CI_LOG_LEVEL")
if log_level is not None:
    # set log level
    if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise RuntimeError("Incorrect log level. Should be - 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")

    log_defaults['CI_LOG_LEVEL'] = log_level
else:
    log_defaults['CI_LOG_LEVEL'] = 'INFO' # default level


# Load logging configuration
logging.config.fileConfig(
    path_to_log_config_file,
    defaults=log_defaults
)

# Add JSON formatter to the handlers
def add_json_formatter_to_handlers():
    json_formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(json_formatter)

# Apply JSON formatter
add_json_formatter_to_handlers()

# create logger
logger = logging.getLogger('custom_integration.clu_converters')

def utf8span(s: str, pos: int):
    """
    Compute the utf8 character index of the [pos] position within the string s
    """

    return len(s[:pos].encode('utf-8'))


def reverse_utf8span(s: str, byte_index: int) -> int:
    """
    Compute the string character index of the [byte_index] position within the UTF-8 encoded string s
    """
    encoded = s.encode('utf-8')
    truncated_encoded = encoded[:byte_index]
    return len(truncated_encoded.decode('utf-8', errors='ignore'))

def at_replacer(hf_entity_object: dict) -> dict:
    """Replaces @ in the name with AT__"""
    if str(hf_entity_object["name"]).startswith("@"):
        hf_entity_object["name"] = str(hf_entity_object["name"]).replace("@","AT__")
    return hf_entity_object
        
def at_reverser(clu_entity_object: dict) -> dict:
    """Replaces AT__ in the name with @"""
    if str(clu_entity_object["category"]).startswith("AT__"):
        clu_entity_object["category"] = str(clu_entity_object["category"]).replace("AT__","@")
    return clu_entity_object


class clu_to_hf_converter:
    def clu_to_hf_process(
            self,
            clu_json: dict,
            delimiter: str,
            language: str = "en-us") -> None:

        # TODO: note potential clashes with utf16 and utf8 in future depending on PVA

        # get a HFWorkspace object to populate
        hf_workspace = humanfirst.objects.HFWorkspace()

        # so assumptions are:
        # create everything into new workspace - if want to to do a merge do it
        # import from hf into temp workspace in HF gui
        # start merge from temp workspace to target workspace (this is how merge works under the hood)
        # TODO: entities - not matching todo in HFWorkspace!
        # Tags will come in with Train and Test

        # examples section
        df_clu_utterances = pandas.json_normalize(clu_json["assets"]["utterances"])
        df_clu_intents = pandas.json_normalize(clu_json["assets"]["intents"])

        # make tags
        tags = df_clu_utterances["dataset"].unique().astype(list)
        # make Train and Test consistent colours
        color_mapper = {
            "Train": "#C3E2C2", # a pastel green for Train
            "Test": "#7ec4e6", # a pastel blue for Test ame color as test-regresion in Academy Ex04
        }
        for tag in tags:
            if pandas.isna(tag):
                continue
            try:
                color = color_mapper[tag]
            except KeyError:
                color = humanfirst.objects.generate_random_color()
            hf_workspace.tag(tag=tag,color=color)

        # make intents
        df_clu_intents["category"].apply(self.clu_to_hf_intent_mapper,args=[hf_workspace,delimiter])

        # make utterances
        created_at = datetime.now().isoformat()
        df_clu_utterances.apply(self.clu_to_hf_utterance_mapper,axis=1,args=[hf_workspace, created_at, delimiter])

        # go to JSON to do entities as not in HFWorkspace

        hf_json = hf_workspace.get_hf_json()

        clu_entities = clu_json["assets"]["entities"]

        # make entities
        hf_json["entities"] = []
        entity = {}
        for clu_entity_object in clu_entities:

            assert isinstance(clu_entity_object,dict)

            # is it a regex?
            if "regex" in clu_entity_object:
                if "list" in clu_entity_object:
                    logger.warning("Both regex and list in entity - this cannot be supported")
                    logger.warning(clu_entity_object)
                # make a check here that there is no list type
                entity = self.clu_to_hf_regex_entity_mapper(clu_entity_object,language=language)
            
            # is it prebuilt
            elif "prebuilts" in clu_entity_object:
                entity = self.clu_to_hf_prebuilt_entity_mapper(clu_entity_object,language=language)
        
            # is it a list
            elif "list" in clu_entity_object:
                entity = self.clu_to_hf_list_entity_mapper(clu_entity_object,language=language)
                           
            # is it learned
            elif "requiredComponents" in clu_entity_object and "learned" in clu_entity_object["requiredComponents"]:
                logger.debug("Learned will be built later")
                continue
            
            # is it a one word entity - convert it to list
            elif len(clu_entity_object.keys()) == 2 and "category" in clu_entity_object and "compositionSetting" in clu_entity_object:
                entity = self.clu_to_hf_one_word_entity_mapper(clu_entity_object,language=language)
                
            else:
                logger.warning("None of the expected entity types encountered")
                logger.warning(clu_entity_object)
                continue
            
            # add it to the json    
            hf_json["entities"].append(entity)

        
        # Entity annotation section
        error_annotated_text = []
        error_full_text = []
        error_full_intent_name = []
        
        # For every utterance in the inbound CLU file
        for i,_ in enumerate(clu_json["assets"]["utterances"]):
            # see if it has entities annotating the utterance
            if "entities" in clu_json["assets"]["utterances"][i]:
                # if it does and it's not blank
                if clu_json["assets"]["utterances"][i]["entities"] != []:
                    # Fill out the entities
                    hf_json["examples"][i]["entities"] = []
                    # For each CLU annotation build the HF annotation
                    for clu_annotation in clu_json["assets"]["utterances"][i]["entities"]:
                                                
                        # Work out based on the offset and length the span and synonym
                        start_char_index = clu_annotation["offset"]
                        end_char_index = clu_annotation["offset"] + clu_annotation["length"]
                        start_byte_index = utf8span(clu_json["assets"]["utterances"][i]["text"], start_char_index)
                        end_byte_index = utf8span(clu_json["assets"]["utterances"][i]["text"], end_char_index)
                        synonym = clu_json["assets"]["utterances"][i]["text"][start_char_index:end_char_index]
                                               
                        # Check here we have the entity keys and values created in humanfirst by a list entity
                        # if not create it
                        hf_json["entities"] = self.check_and_update_list_entity(hf_json["entities"],clu_annotation["category"],synonym,language)
                        
                        
                        hf_annotation = {
                            "name": clu_annotation["category"],
                            "text": synonym,
                            "span": {
                                "from_character": start_byte_index,
                                "to_character": end_byte_index
                            },
                            "value": self.find_key_value_for_synonym(hf_json["entities"],clu_annotation["category"],synonym)
                        }

                        hf_json["examples"][i]["entities"].append(hf_annotation)
        if len(error_annotated_text) != 0:
            raise RuntimeError(f"Error annotatations: {error_annotated_text}\nIn corresponding utterance list : {error_full_text}\nIn corresponding intent list {error_full_intent_name}\nDon't exists in any entities")
        return hf_json

    def check_and_update_list_entity(self, hf_entities: list, category: str, synonym: str, language: str) -> dict:
        """Checks if a clu entity category exists in the hf_entities
        If it doesn't it creates it.
        Then checks if the annotation"""
        
        isonow = datetime.now().isoformat()
        
        # Search to see if the matching list entity exists.
        found_entity_index = None
        i=0
        for hf_entity in hf_entities:
            if hf_entity["name"] == category:
                found_entity_index = i
                break
            i = i + 1
        
        # create it if it doesnt and return 
        if found_entity_index is None: 
            hf_entity = {
                "id": humanfirst.objects.hash_string(category,"entity"),
                "name": category,
                "values": [
                    {
                        "id": humanfirst.objects.hash_string(category + synonym,"entval"),
                        "key_value": "learned_annotation",
                        "synonyms": [
                            {
                                "value": synonym
                            }
                        ]
                    }
                ],
                "created_at": isonow,
                "updated_at": isonow
            }
            hf_entities.append(hf_entity)
            logger.debug(f"Created entity {category} as a list with key_value: learned_annotation with synonym {synonym}")
            return hf_entities

        # otherwise it must have been found and we have the index
        # check if the synonym is in the entity and return if it is 
        for hf_value in hf_entities[found_entity_index]["values"]:
            for hf_synonym in hf_value["synonyms"]:
                if hf_synonym["value"] == synonym:
                    logger.debug(f'Synonym: {synonym} already exists within key_value: {hf_value["key_value"]} for entity {category}')
                    return hf_entities
        
        # otherwise we check whether we already have learned_annotation to add to
        for hf_value in hf_entities[found_entity_index]["values"]:
            if hf_value["key_value"] == "learned_annotation":
                hf_value["synonyms"].append({
                    "value": synonym
                })
                logger.debug(f"Synonym: {synonym} added to key_value: learned_entities for entity {category}")
                return hf_entities
        
        # otherwise we need to create a new learned_annotation
        hf_value = {
            "id": humanfirst.objects.hash_string(category + synonym,"entval"),
            "key_value": "learned_annotation",
            "synonyms": [
                {
                    "value": synonym
                }
            ]
        }
        hf_entities[found_entity_index]["values"].append(hf_value)
        logger.debug(f"Synonym: {synonym} added to created key_value: learned_entities for entity {category}")
        return hf_entities
    
    
    def find_key_value_for_synonym(self, hf_entities: dict, category: str, synonym: str) -> str:
        """Finds the key_value for a synonym in a list of entities and returns
        an empty string if it can't find it."""
        
        # Search to see if the matching list entity exists.
        for hf_entity in hf_entities:
            if hf_entity["name"] == category:
                for hf_value in hf_entity["values"]:
                    for hf_synonym in hf_value["synonyms"]:
                        if hf_synonym["value"] == synonym:
                            return hf_value["key_value"]
        return ""
                

    def clu_to_hf_list_entity_mapper(self, clu_entity_object: dict, language: str) -> dict:
        """Builds a HF entity object for any clu lists"""
        
        try:
            # hf_entity using name to generate hash id
            isonow = datetime.now().isoformat()
            hf_entity =  {
                "id": humanfirst.objects.hash_string(clu_entity_object["category"],"entity"),
                "name": clu_entity_object["category"],
                "values": [],
                "created_at": isonow,
                "updated_at": isonow
            }
            
            # replace @
            hf_entity = at_replacer(hf_entity)

            # add key values
            for clu_sublist_object in clu_entity_object["list"]["sublists"]:
                hf_key_value_object = {
                    "id": humanfirst.objects.hash_string(clu_entity_object["category"] + clu_sublist_object["listKey"],"entval"),
                    "key_value": clu_sublist_object["listKey"],
                    "synonyms": []
                }
                # add synonyms
                for clu_synonyms_object in clu_sublist_object["synonyms"]:
                    found_language = False
                    if clu_synonyms_object["language"] == language:
                        found_language = True
                        for clu_synonym in clu_synonyms_object["values"]:
                            hf_synonym = {
                                "value": clu_synonym
                            }
                            hf_key_value_object["synonyms"].append(copy.deepcopy(hf_synonym))
                    if not found_language:
                        raise RuntimeError(f'Could not find language synonyms for {language}')
                    hf_entity["values"].append(copy.deepcopy(hf_key_value_object))
        except Exception as e:
            print(json.dumps(clu_entity_object,indent=2))
            raise
            # Need some sort of debug here

        return copy.deepcopy(hf_entity)


    def clu_to_hf_intent_mapper(self, intent_name: str, hf_workspace: humanfirst.objects.HFWorkspace, delimiter: str) -> None:
        """Builds the parent and child structures for an intent name"""
        # clu doesn't have separate IDs (current understanding)
        if delimiter != "":
            intent_hierarchy = intent_name.split(delimiter)
        else:
            intent_hierarchy = intent_name
        hf_workspace.intent(intent_hierarchy)

    def clu_to_hf_utterance_mapper(self, 
                                   row: pandas.Series,
                                   hf_workspace: humanfirst.objects.HFWorkspace,
                                   created_at: datetime,
                                   delimiter: str) -> None:
        """Builds HF example"""
        fully_qualified_intent_name = str(row["intent"])

        if delimiter != "":
            intent_hierarchy = fully_qualified_intent_name.split(delimiter)
        else:
            intent_hierarchy = fully_qualified_intent_name

        try:
            tag_name = row["dataset"]
            if pandas.isna(tag_name):
                tag_name = "Train"
        except KeyError:
            tag_name = "Train"
        hf_workspace.example(
            row["text"],
            intents=[hf_workspace.intent(intent_hierarchy)],
            created_at=created_at,
            tags=[{"id": hf_workspace.tag(tag_name).id }]
        )

    def clu_to_hf_one_word_entity_mapper(self, clu_entity_object: dict, language: str) -> dict:
        """Builds a HF entity object list object out of entities in
        CLU with just a name value."""
        
        try:
            # hf_entity using name to generate hash id
            isonow = datetime.now().isoformat()
            hf_entity =  {
                "id": humanfirst.objects.hash_string(clu_entity_object["category"],"entity"),
                "name": clu_entity_object["category"],
                "values": [
                    {
                        "id": humanfirst.objects.hash_string(clu_entity_object["category"],"entval"),
                        "key_value": clu_entity_object["category"],
                        "synonyms": [
                            {
                                "value": clu_entity_object["category"]
                            }
                        ]
                    }
                ],
                "created_at": isonow,
                "updated_at": isonow
            }
            
            # replace @
            hf_entity = at_replacer(hf_entity)

        except Exception as e:
            print(json.dumps(clu_entity_object,indent=2))
            raise
            # Need some sort of debug here

        return copy.deepcopy(hf_entity)

    def clu_to_hf_prebuilt_entity_mapper(self, clu_entity_object: dict, language: str) -> dict:
        """Builds a HF system object from a CLU prebuilt entity
        for a limited number of DF compatible types
        Currently only Datetime and Quantity.Number supported"""

        # These are the 2025-02-10 types of system entities humanfirst supports creating.
        # based on dialog flow with the assumed CLU types next to in brackets
        # sys.date
        # sys.date-time (Datetime)
        # sys.email
        # sys.number (Quantity.Number)
        # sys.phone-number
        # sys.time
        # sys.url
        # sys.zip-code
        
        # This is the CLU page
        # https://learn.microsoft.com/en-us/azure/ai-services/language-service/conversational-language-understanding/prebuilt-component-reference

        # Format - will hide the name in the id for returning to CLU             
        # {
        #     "id": "entity-HXTCKVLE45HKHEV3Z7IALSUL",
        #     "name": "sys.date-time",
        #     "system_type": "SYS_DATE_TIME",
        #     "settings": {},
        #     "created_at": "2025-02-10T15:41:06Z",
        #     "updated_at": "2025-02-10T15:41:06Z"
        # },
        # {
        #     "id": "entity-EU565KK7IRBPJCKNOEX4XMDE",
        #     "name": "sys.number",
        #     "system_type": "SYS_NUMBER",
        #     "settings": {},
        #     "created_at": "2025-02-10T15:41:37Z",
        #     "updated_at": "2025-02-10T15:41:37Z"
        # }
        
        # CLU FORMAT
        #  {
        #         "category": "builtin.number",
        #         "compositionSetting": "combineComponents",
        #         "prebuilts": [
        #             {
        #                 "category": "Quantity.Number"
        #             }
        #         ]
        #     },

           
        try:
            # hf_entity skeleton regex using name to generate hash id
            isonow = datetime.now().isoformat()
            hf_entity =  {
                "id": humanfirst.objects.hash_string(clu_entity_object["category"],"entity"),
                "name": clu_entity_object["category"],
                "system_type":SYSTEM_ENTITY_MAPPER[clu_entity_object["prebuilts"][0]["category"]],
                "settings": {},
                "created_at": isonow,
                "updated_at": isonow
            }
            
            # replace @
            hf_entity = at_replacer(hf_entity)           
            
        except Exception as e:
            print(json.dumps(clu_entity_object,indent=2))
            raise
            # Need some sort of debug here

        return copy.deepcopy(hf_entity)


    def clu_to_hf_regex_entity_mapper(self, clu_entity_object: dict, language: str) -> dict:
        """Builds a HF regex object from a CLU regex entity"""
        
        try:            
            # hf_entity skeleton regex using name to generate hash id
            isonow = datetime.now().isoformat()
            hf_entity =  {
                "id": humanfirst.objects.hash_string(clu_entity_object["category"],"entity"),
                "name": clu_entity_object["category"],
                "values": [],
                "is_regex": True,
                "settings": {},
                "created_at": isonow,
                "updated_at": isonow
            }
            
            # replace @
            hf_entity = at_replacer(hf_entity)
            
            # go through each CLU expression and create a humanfirst value object for it.
            # language is not preserved - assumed to come back in on reconversion
            values = []
            for expression in clu_entity_object["regex"]["expressions"]:
                hf_value_object = {
                    "id": f'entval-{expression["regexKey"]}',
                    "key_value": expression["regexPattern"],
                    "synonyms": [
                        {
                            "value": expression["regexPattern"]
                        }
                    ]
                }
                values.append(hf_value_object)
            hf_entity["values"] = values
            
        except Exception as e:
            print(json.dumps(clu_entity_object,indent=2))
            raise
            # Need some sort of debug here

        return copy.deepcopy(hf_entity)

        
    def clu_to_hf_regex_entity_mapper(self, clu_entity_object: dict, language: str) -> dict:
        """Builds a HF regex object from a CLU regex entity"""
        
        try:            
            # hf_entity skeleton regex using name to generate hash id
            isonow = datetime.now().isoformat()
            hf_entity =  {
                "id": humanfirst.objects.hash_string(clu_entity_object["category"],"entity"),
                "name": clu_entity_object["category"],
                "values": [],
                "is_regex": True,
                "settings": {},
                "created_at": isonow,
                "updated_at": isonow
            }
            
            # replace @
            hf_entity = at_replacer(hf_entity)
            
            # go through each CLU expression and create a humanfirst value object for it.
            # language is not preserved - assumed to come back in on reconversion
            values = []
            for expression in clu_entity_object["regex"]["expressions"]:
                hf_value_object = {
                    "id": f'entval-{expression["regexKey"]}',
                    "key_value": expression["regexPattern"],
                    "synonyms": [
                        {
                            "value": expression["regexPattern"]
                        }
                    ]
                }
                values.append(hf_value_object)
            hf_entity["values"] = values
            
        except Exception as e:
            print(json.dumps(clu_entity_object,indent=2))
            raise
            # Need some sort of debug here

        return copy.deepcopy(hf_entity)

class hf_to_clu_converter:
    def hf_to_clu_process(self,
                        hf_json: dict,
                        clu_json: dict,
                        delimiter: str,
                        language: str = "en-us",
                        skip: bool = False) -> None:
        """Process HF to CLU conversion"""

        # TODO: note potential clashes with utf16 and utf8 in future depending on PVA

        # get a HFWorkspace object to get fully qualified intent names
        # logger.info("delimiter blah blah")
        logger.info(f"Delimiter {delimiter}")

        if delimiter != "":
            hf_workspace = humanfirst.objects.HFWorkspace.from_json(hf_json,delimiter=delimiter)
        else:
            hf_workspace = humanfirst.objects.HFWorkspace.from_json(hf_json,delimiter=None)

        # get the tag for Test dataset
        test_tag_id = None
        found = False
        if "tags" in hf_json:
            for tag in hf_json["tags"]:
                if tag["name"] == "Test":
                    found = True
                    test_tag_id = tag["id"]
                    break
        
        if found:
            logger.info(f'Found test_tag_id: {test_tag_id}\n')
        else:
            logger.info('No test_tag_id found.\n')
            
        # entities - have to do ahead of utterances as the utterances will need to refer back to them.
        clu_json["assets"]["entities"] = []
        if "entities" in hf_json:
            for hf_entity in hf_json["entities"]:
                # if goes here of different entity types
                
                # check if regex
                if "is_regex" in hf_entity:
                    clu_json["assets"]["entities"].append(self.hf_to_clu_regex_entity_mapper(hf_entity,language))             
                # check if system
                elif "system_type" in hf_entity:
                    if hf_entity["system_type"] in ["SYS_DATE_TIME","SYS_NUMBER"]:
                        clu_json["assets"]["entities"].append(self.hf_to_clu_prebuilt_entity_mapper(hf_entity,language))             
                    else:
                        logger.warning("SystemType not supported")    
                # else we are going to assume list (learned) will come from annotations
                else:
                    if not "values" in hf_entity:
                        logger.warning("No values in entity")
                        continue
                    else:
                        clu_json["assets"]["entities"].append(self.hf_to_clu_list_entity_mapper(hf_entity,language))

        # examples section
        df_examples = pandas.json_normalize(hf_json["examples"])
        logger.info(df_examples)
        df_examples["clu_utterance"] = df_examples.apply(self.hf_to_clu_utterance_mapper,
                                                        args=[language,hf_workspace,test_tag_id,skip],
                                                        axis=1)
        clu_json["assets"]["utterances"] = df_examples["clu_utterance"].to_list()
        
        # go through and update all the clu_entities with requiredComponent = ['learned']
        # where they have any annotations
        # find every utterance
        for clu_utterance in clu_json["assets"]["utterances"]:
            # where it has entity annotations find each one
            for clu_entity_annotation in clu_utterance["entities"]:
                # search for that same entity name within the entities
                for clu_entity in clu_json["assets"]["entities"]:
                    # Check if learned is already set and if not set it.
                    if clu_entity_annotation["category"] == clu_entity["category"]:
                        if not "requiredComponents" in clu_entity:
                            clu_entity["requiredComponents"] = ["learned"]
                        break      

        # find any intents that were in utterances
        # this avoids creating any parents, but also doesn't create empty children
        # TODO: Empty intents should be passed if the signal from the request deems it to be
        clu_intent_names = set()
        for clu_utterance in clu_json["assets"]["utterances"]:
            clu_intent_names.add(clu_utterance["intent"])
        
        # logger.info(clu_intent_names)
        # set to list
        clu_intents = []
        for intent_name in clu_intent_names:
            clu_intents.append(self.hf_to_clu_intent_mapper(intent_name))
        
        # logger.info(clu_intents)
        #
        clu_json["assets"]["intents"] = clu_intents

        return clu_json

    def hf_to_clu_intent_mapper(self, intent_name: str) -> dict:
        """Returns a clu_intent as a dict with the category set to
        the passed name"""
        # clu doesn't have separate IDs (current understanding)
        return {
            "category": intent_name
        }

    def hf_to_clu_prebuilt_entity_mapper(self, hf_entity: dict, language: str) -> dict:
        """converts hf system entity format to clu prebuilt entity format"""
        # {
        #     "category": "builtin.number",
        #     "compositionSetting": "combineComponents",
        #     "prebuilts": [
        #         {
        #             "category": "Quantity.Number"
        #         }
        #     ]
        # },

        try:
            # build entity object
            clu_entity_object = {
                "category": hf_entity["name"],
                "compositionSetting": "combineComponents",
                "prebuilts": [
                    {
                        "category": SYSTEM_ENTITY_REVERSER[hf_entity["system_type"]]
                    }
                    
                ]
            }
            
            # put back @
            clu_entity_object = at_reverser(clu_entity_object)

        except Exception as e:
            print(json.dumps(hf_entity,indent=2))
            raise
            # Need some sort of debug here  

        # return copy of entity
        return copy.deepcopy(clu_entity_object)

    def hf_to_clu_regex_entity_mapper(self, hf_entity: dict, language: str) -> dict:
        """converts hf regex entity format to clu regex entity format"""
        # {
        #     "category": "telefoonnummer",
        #     "compositionSetting": "combineComponents",
        #     "regex": {
        #         "expressions": [
        #             {
        #                 "regexKey": "tel_mobiel_vast",
        #                 "language": "nl",
        #                 "regexPattern": "\\b(0[1-9][0-9]{1,2})[ -]?[0-9]{3}[ -]?[0-9]{4}|(06)[ -]?[0-9]{2}[ -]?[0-9]{3}[ -]?[0-9]{3}\\b"
        #             },
        #             {
        #                 "regexKey": "mobiel",
        #                 "language": "nl",
        #                 "regexPattern": "\\b(((\\\\+31|0|0031)6){1}[1-9]{1}[0-9]{7})\\b"
        #             },
        #         ]
        #     }
        # },

        try:
            # build entity object
            clu_entity_object = {
                "category": hf_entity["name"],
                "compositionSetting": "combineComponents",
                "regex": {
                    "expressions": []
                }
            }
            
            # put back @
            clu_entity_object = at_reverser(clu_entity_object)

            # fill list with key values
            for hf_key_value_object in hf_entity["values"]:
                clu_expression = {
                    "regexKey": str(hf_key_value_object["id"]).replace("entval-",""), # remove entval from the ID to recreate the key
                    "language": language,
                    "regexPattern": hf_key_value_object["synonyms"][0]["value"]
                }
                clu_entity_object["regex"]["expressions"].append(copy.deepcopy(clu_expression))

        except Exception as e:
            print(json.dumps(hf_entity,indent=2))
            raise
            # Need some sort of debug here  

        # return copy of entity
        return copy.deepcopy(clu_entity_object)

    def hf_to_clu_list_entity_mapper(self, hf_entity: dict, language: str) -> dict:
        """converts hf entity format to clu list entity format"""
        # known_entity_key_types = ["prebuilts","list","requiredComponents"]
        # script_supported_types = ["list"]

        # check type and skip if unknown
        # known_entity = False
        # for entity_type in known_entity_key_types:
        #     if entity_type in entity:
        #         known_entity = True
        # if not known_entity:
        #     warnings.warn(f'Unknown entity type keys are: {entity.keys()}')
        #     continue:

        try:
            # build entity object
            clu_entity_object = {
                "category": hf_entity["name"],
                "compositionSetting": "combineComponents",
                "list": {
                    "sublists": []
                }
            }
            
            # put back @
            clu_entity_object = at_reverser(clu_entity_object)

            # fill list with key values
            for hf_key_value_object in hf_entity["values"]:
                clu_sublist_object = {
                    "listKey": hf_key_value_object["key_value"],
                    "synonyms": [
                        {
                            "language": language,
                            "values": []
                        }
                    ]
                }

                # fill values with values
                if "synonyms" in hf_key_value_object:
                    for synonym in hf_key_value_object["synonyms"]:
                        clu_sublist_object["synonyms"][0]["values"].append(synonym["value"])
                else:
                    # Every entity value should have atleast a synonym
                    clu_sublist_object["synonyms"][0]["values"].append(hf_key_value_object["key_value"])

                # insert sublist into entity
                clu_entity_object["list"]["sublists"].append(copy.deepcopy(clu_sublist_object))
        except Exception as e:
            print(json.dumps(hf_entity,indent=2))
            raise
            # Need some sort of debug here  

        # return copy of entity
        return copy.deepcopy(clu_entity_object)


    def hf_to_clu_utterance_mapper(self, row: pandas.Series,
                        language: str,
                        hf_workspace: humanfirst.objects.HFWorkspace,
                        test_tag_id: str,
                        skip: bool) -> dict:
        """Maps CLU utterance to HF utterance format"""

        # Check fit the data is labelled Train/Test - all with no labels will be Train
        dataset = TRAIN
        if "tags" in row:
            if isinstance(row["tags"],list):
                for tag in row["tags"]:
                    if tag["id"] == test_tag_id:
                        dataset = TEST
                        break
            elif pandas.isna(row["tags"]):
                pass
            else:
                warnings.warn(f'Found utterance with tags not list or Na: {row}')

        intent_name = hf_workspace.get_fully_qualified_intent_name(row["intents"][0]["intent_id"])
        # logger.info(row["intents"][0]["intent_id"])
        # logger.info(intent_name)
        if len(intent_name) > 50:
            if not skip:
                raise RuntimeError(f'intent name length of {len(intent_name)} exceeds 50 chars.  {intent_name}')

        clu_entities = []
        if "entities" in row:
            if row["entities"] != [] and isinstance(row["entities"], list):
                for hf_annotation in row["entities"]:
                    start_char_index = reverse_utf8span(row["text"], hf_annotation["span"]["from_character"])
                    end_char_index = reverse_utf8span(row["text"], hf_annotation["span"]["to_character"])
                    clu_annotation = {
                            "category": hf_annotation["name"],
                            "offset": start_char_index,
                            "length": end_char_index - start_char_index
                        }

                    clu_entities.append(clu_annotation)
        
            # So all the annotations are here but we need to add back in the requiredComponent for learned
            # So if it has annotations we assume it has learned and that is required only assumption possible
            # Can't do it here in the code as we need to add the information to the clu_entity object
            # not the utterance annotation.

        return {
            "text": row["text"],
            "language": language,
            "intent": hf_workspace.get_fully_qualified_intent_name(row["intents"][0]["intent_id"]),
            "entities": clu_entities,
            "dataset": dataset
        }

class clu_converter(clu_to_hf_converter, hf_to_clu_converter):
    """Handles HF to CLU and CLU to HF"""