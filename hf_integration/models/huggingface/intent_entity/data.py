import numpy as np
from bidict import bidict

LABEL_IGNORE = -100
LABEL_NO_ENTITY = 0

class Workspace:
    def __init__(self, data) -> None:
        self.data = data
        self.intents = {i['id']: i for i in self.data.get('intents', [])}
        self.entities = {i['id']: i for i in self.data.get('entities', [])}
        self.entity_values = {v['id']: (e,v) for e in self.data.get('entities', []) for v in e['values']}
        self.entity_values_entities = {v['id']: e['id'] for e, v in self.entity_values.values()}

        def map_ids(d):
            return bidict({i: d for i, d in enumerate(d.keys())})

        self.intent_ids = map_ids(self.intents)

        expanded_entities = ["O"]
        for entity_id in self.entity_values.keys():
            expanded_entities.extend([f'B-{entity_id}', f'I-{entity_id}'])

        self.entity_ids = bidict({i: k for i, k in enumerate(expanded_entities)})


    def labels_for_examples(self, tokenizer, examples):
        ## XXX: TODO: max_length=256 might be too small for some datasets, we would need some heuristics around the
        ## max vram usage, and dynamically adjust the batch size.
        tokenized = tokenizer([e['text'] for e in examples], return_tensors='tf', padding=True, max_length=256, pad_to_multiple_of=8)
        
        labels = np.zeros((len(tokenized.encodings), len(tokenized[0].offsets)))
        intent_labels = np.zeros((len(tokenized.encodings),))
        
        for i, (tokenized_example, example) in enumerate(zip(tokenized.encodings, examples)):
            
            intent_id = example['intents'][0]['intent_id']
            intent_labels[i] = self.intent_ids.inv[intent_id]
            
            for j, offset in enumerate(tokenized_example.offsets):
                tok_len = offset[1] - offset[0]
                if tok_len == 0:
                    labels[i, j] = LABEL_IGNORE
                else:
                    labels[i, j] = LABEL_NO_ENTITY
                    
            found = False
            for ent in example.get('entities', []):
                found = False
                # Find which spans overlap with this entity
                for j, offset in enumerate(tokenized_example.offsets):
                    prefix = "I-" if found else "B-"
                    if overlaps(offset, (ent['span']['from_character'], ent['span']['to_character'])):
                        if ent.get('value_id', None) is None:
                            ## Disregard invalid entity annotations
                            continue

                        label_id = self.entity_ids.inv.get(prefix + ent['value_id'])
                        if label_id is not None:
                            labels[i,j] = label_id
                        found = True
        
        tokenized['entity_labels'] = labels
        tokenized['intent_labels'] = intent_labels
    
        return tokenized


def overlaps(a, b):
    return not (a[1] < b[0] or b[1] < a[0])

