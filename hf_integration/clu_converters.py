"""
python clu_clu_to_hf_converter.py

Convert CLU JSON into HF and  HF to CLU JSON

"""
# *********************************************************************************************************************

# standard imports
import datetime
import warnings
import copy

# 3rd party imports
import pandas

# custom imports
import humanfirst

# CLU name constants
TRAIN="Train"
TEST="Test"


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


class clu_to_hf_converter:
    def clu_to_hf_process(
            self,
            clu_json: dict,
            delimiter: str = "-",
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
        created_at = datetime.datetime.now().isoformat()
        df_clu_utterances.apply(self.clu_to_hf_utterance_mapper,axis=1,args=[hf_workspace, created_at, delimiter])

        # go to JSON to do entities as not in HFWorkspace

        hf_json = hf_workspace.get_hf_json()

        clu_entities = clu_json["assets"]["entities"]

        # make entities
        hf_json["entities"] = []
        simple_entity = {}
        for clu_entity_object in clu_entities:

            assert isinstance(clu_entity_object,dict)
            known_entity_key_types = ["prebuilts","list","requiredComponents"]
            script_supported_types = ["list"]

            # check type and skip if unknown
            known_entity = False
            for entity_type in known_entity_key_types:
                if entity_type in clu_entity_object:
                    known_entity = True
                    if entity_type in script_supported_types:
                        entity = self.clu_to_hf_entity_mapper(clu_entity_object,language=language)
                        hf_json["entities"].append(entity)
                        simple_entity[entity["name"]] = {}
                        for value in entity["values"]:
                            simple_entity[entity["name"]][value["key_value"]] = []
                            for synonym in value["synonyms"]:
                                simple_entity[entity["name"]][value["key_value"]].append(synonym["value"])

            if not known_entity:
                warnings.warn(f'Unknown entity type keys are: {clu_entity_object.keys()}')
                continue

        for i,_ in enumerate(clu_json["assets"]["utterances"]):
            if "entities" in clu_json["assets"]["utterances"][i]:
                if clu_json["assets"]["utterances"][i]["entities"] != []:
                    hf_json["examples"][i]["entities"] = []
                    for clu_annotation in clu_json["assets"]["utterances"][i]["entities"]:
                        start_char_index = clu_annotation["offset"]
                        end_char_index = clu_annotation["offset"] + clu_annotation["length"]
                        start_byte_index = utf8span(clu_json["assets"]["utterances"][i]["text"], start_char_index)
                        end_byte_index = utf8span(clu_json["assets"]["utterances"][i]["text"], end_char_index)
                        synonym = clu_json["assets"]["utterances"][i]["text"][start_char_index:end_char_index]
                        for entity_key,entity_value in simple_entity[clu_annotation["category"]].items():
                            for j,_ in enumerate(entity_value):
                                entity_value[j] = entity_value[j].lower()
                            if synonym.lower() in entity_value:
                                hf_entity_key = entity_key
                                break
                        else:
                            raise RuntimeError(f"'{synonym}' is not present in entity: '{simple_entity[clu_annotation['category']]}'")

                        hf_annotation = {
                            "name": clu_annotation["category"],
                            "text": synonym,
                            "span": {
                                "from_character": start_byte_index,
                                "to_character": end_byte_index
                            },
                            "value": hf_entity_key
                        }

                        hf_json["examples"][i]["entities"].append(hf_annotation)

        return hf_json


    def clu_to_hf_entity_mapper(self, clu_entity_object: dict, language: str) -> dict:
        """Builds a HF entity object for any clu lists"""

        # hf_entity using name to generate hash id
        isonow = datetime.datetime.now().isoformat()
        hf_entity =  {
            "id": humanfirst.objects.hash_string(clu_entity_object["category"],"entity"),
            "name": clu_entity_object["category"],
            "values": [],
            "created_at": isonow,
            "updated_at": isonow
        }

        # add key values
        for clu_sublist_object in clu_entity_object["list"]["sublists"]:
            hf_key_value_object = {
                "id": humanfirst.objects.hash_string(clu_sublist_object["listKey"],"entval"),
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

        return copy.deepcopy(hf_entity)


    def clu_to_hf_intent_mapper(self, intent_name: str, hf_workspace: humanfirst.objects.HFWorkspace, delimiter: str) -> None:
        """Builds the parent and child structures for an intent name"""
        # clu doesn't have separate IDs (current understanding)
        intent_hierarchy = intent_name.split(delimiter)
        hf_workspace.intent(intent_hierarchy)

    def clu_to_hf_utterance_mapper(self, 
                                   row: pandas.Series,
                                   hf_workspace: humanfirst.objects.HFWorkspace,
                                   created_at: datetime.datetime,
                                   delimiter: str) -> None:
        """Builds HF example"""
        fully_qualified_intent_name = str(row["intent"])
        intent_hierarchy = fully_qualified_intent_name.split(delimiter)
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

class hf_to_clu_converter:
    def hf_to_clu_process(self,
                        hf_json: dict,
                        clu_json: dict,
                        language: str = "en-us",
                        delimiter: str = "-",
                        skip: bool = False) -> None:
        """Process HF to CLU conversion"""

        # TODO: note potential clashes with utf16 and utf8 in future depending on PVA

        # get a HFWorkspace object to get fully qualified intent names
        # print("delimiter blah blah")
        print(f"Delimiter {delimiter}")
        hf_workspace = humanfirst.objects.HFWorkspace.from_json(hf_json,delimiter)

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
            print(f'Found test_tag_id: {test_tag_id}\n')
        else:
            print('No test_tag_id found.\n')

        # examples section
        df_examples = pandas.json_normalize(hf_json["examples"])
        print(df_examples)
        df_examples["clu_utterance"] = df_examples.apply(self.hf_to_clu_utterance_mapper,
                                                        args=[language,hf_workspace,test_tag_id,skip],
                                                        axis=1)
        clu_json["assets"]["utterances"] = df_examples["clu_utterance"].to_list()

        # find any intents that were in utterances
        # this avoids creating any parents, but also doesn't create empty children
        # TODO: Empty intents should be passed if the signal from the request deems it to be
        clu_intent_names = set()
        for clu_utterance in clu_json["assets"]["utterances"]:
            clu_intent_names.add(clu_utterance["intent"])
        
        # print(clu_intent_names)
        # set to list
        clu_intents = []
        for intent_name in clu_intent_names:
            clu_intents.append(self.hf_to_clu_intent_mapper(intent_name))
        
        # print(clu_intents)
        #
        clu_json["assets"]["intents"] = clu_intents

        # entities
        clu_json["assets"]["entities"] = []
        if "entities" in hf_json:
            for hf_entity in hf_json["entities"]:
                    clu_json["assets"]["entities"].append(self.hf_to_clu_entity_mapper(hf_entity,language))

        return clu_json

    def hf_to_clu_intent_mapper(self, intent_name: str) -> dict:
        """Returns a clu_intent as a dict with the category set to
        the passed name"""
        # clu doesn't have separate IDs (current understanding)
        return {
            "category": intent_name
        }

    def hf_to_clu_entity_mapper(self, hf_entity: dict, language: str) -> dict:
        """converts hf entity format to clu entity format"""
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

        # build entity object
        clu_entity_object = {
            "category": hf_entity["name"],
            "compositionSetting": "combineComponents",
            "list": {
                "sublists": []
            }
        }

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
                        # print("Found")
                        dataset = TEST
                        break
            elif pandas.isna(row["tags"]):
                pass
            else:
                warnings.warn(f'Found utterance with tags not list or Na: {row}')

        intent_name = hf_workspace.get_fully_qualified_intent_name(row["intents"][0]["intent_id"])
        # print(row["intents"][0]["intent_id"])
        # print(intent_name)
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

        return {
            "text": row["text"],
            "language": language,
            "intent": hf_workspace.get_fully_qualified_intent_name(row["intents"][0]["intent_id"]),
            "entities": clu_entities,
            "dataset": dataset
        }

class clu_converter(clu_to_hf_converter, hf_to_clu_converter):
    """Handles HF to CLU and CLU to HF"""