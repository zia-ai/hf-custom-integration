import inspect
import tensorflow as tf
import transformers as xf
import numpy as np
from .token_classification import TokenClassificationPipeline, AggregationStrategy

from typing import Dict, Any, List, Optional, Tuple, Union
from .data import Workspace
from ....humanfirst.protobuf.external_nlu.v1alpha1 import service_pb2

from transformers.modeling_tf_utils import (
    TFModelInputType,
    TFTokenClassificationLoss,
    get_initializer,
    unpack_inputs,
)


def create_model_class(BaseClass):
    """
    Create a custom model class based on the given pretrained model base class.

    Adds a dual intent & entity objective on top of the encoder.
    """

    class TFCustomModel(BaseClass):
        def __init__(self, config, *inputs, **kwargs):
            super().__init__(config, *inputs, **kwargs)

            self.num_intents = config.task_specific_params['intents']['num_intents']
            self.num_entities = config.task_specific_params['entities']['num_entities']

            if hasattr(config, 'classifier_dropout'):
                dropout_rate = (
                    config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
                )
            else:
                dropout_rate = 0.1

            initializer = get_initializer(config.initializer_range) if hasattr(
                config, 'initializer_range') else None

            self.entity_dropout = tf.keras.layers.Dropout(
                rate=dropout_rate, name="entity_classifier_dropout")
            self.entity_classifier = tf.keras.layers.Dense(
                units=self.num_entities,
                kernel_initializer=initializer,
                name="entity_classifier",
            )

            self.intent_dropout = tf.keras.layers.Dropout(
                rate=dropout_rate, name="intent_classifier_dropout")
            self.intent_classifier = tf.keras.layers.Dense(
                units=self.num_intents,
                kernel_initializer=initializer,
                name="intent_classifier",
                activation="softmax",
            )

        @unpack_inputs
        def call(
                self,
                input_ids: Optional[TFModelInputType] = None,
                attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
                token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
                position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
                head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
                inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                intent_labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
                entity_labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
                training: Optional[bool] = False,
        ) -> Union[Dict[str, Any], Tuple[tf.Tensor]]:
            r"""
            labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
            """

            kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'head_mask': head_mask,
                'inputs_embeds': inputs_embeds,
                'output_attentions': output_attentions,
                'output_hidden_states': output_hidden_states,
                'return_dict': True,
                'training': training,
            }
            
            sig = inspect.signature(super().call)
            for k in list(kwargs.keys()):
                if k not in sig.parameters:
                    del kwargs[k]

            outputs = super().call(**kwargs)

            # Entites are predicted on a per-token basis
            sequence_output = outputs['last_hidden_state']
            sequence_output = self.entity_dropout(
                inputs=sequence_output, training=training)
            entity_logits = self.entity_classifier(inputs=sequence_output)
            entity_loss = None if entity_labels is None else TFTokenClassificationLoss.hf_compute_loss(self,
                                                                                                       labels=entity_labels,
                                                                                                       logits=entity_logits)

            # Intents are predicted on a per-example basis
            pooled_output = outputs.get('pooler_output')
            if pooled_output is None:
                # If the model doesn't provide pooling, average all the hidden states
                pooled_output = tf.reduce_sum(sequence_output, axis=1)

            pooled_output = self.intent_dropout(
                inputs=pooled_output, training=training)
            intent_logits = self.intent_classifier(inputs=pooled_output)
            intent_loss = None if intent_labels is None else tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE)(intent_labels, intent_logits)

            # Add both losses for joint training
            loss = (tf.reduce_sum(intent_loss) + tf.reduce_sum(
                entity_loss)) if intent_loss is not None and entity_loss is not None else None

            if not return_dict:
                output = (intent_logits, entity_logits) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return {
                'loss': loss,
                'intent_logits': intent_logits,
                'entity_logits': entity_logits,

                'hidden_states': outputs.hidden_states,
                'attentions': outputs.attentions,
            }
        def serving_output(self, output):
            return {
                'intent_logits': output['intent_logits'],
                'entity_logits': output['entity_logits'],
            }

    return TFCustomModel


HF_CONFIGS = {
    'bert': ('bert-base-uncased', xf.TFBertModel),
    'mt5': ('google/mt5-base', xf.TFMT5EncoderModel),
    'xlm-roberta': ('jplu/tf-xlm-roberta-base', xf.TFXLMRobertaModel)
}


class Trainer:

    def __init__(self, dir_name=None, output_dir_name=None, workspace_path=None, workspace_data=None):
        self.dir_name = dir_name
        import json
        import os
        if workspace_data is not None:
            self.workspace = Workspace(json.loads(workspace_data))
        else:
            if workspace_path is None:
                workspace_path = os.path.join(dir_name, 'workspace.json')
            with open(workspace_path, 'r') as f:
                self.workspace = Workspace(json.load(f))

        # Check if training_args.json exists
        if dir_name is not None and os.path.exists(os.path.join(dir_name, 'training_args.json')):
            with open(os.path.join(dir_name, 'training_args.json'), 'r') as f:
                self.training_args = xf.TFTrainingArguments(
                    output_dir=output_dir_name, **json.load(f))
        else:
            self.training_args = xf.TFTrainingArguments(
                output_dir=output_dir_name,
                seed=42,
                num_train_epochs=10,
            )

        if dir_name is not None and os.path.exists(os.path.join(dir_name, 'config.json')):
            with open(os.path.join(dir_name, 'config.json'), 'r') as f:
                config = json.load(f)
                self.use_model = config.get('use_model', 'bert')
        else:
            self.use_model = 'bert'

        # TODO: Pretty sure there is a map from the model name to the base class to use as barebone model
        self.base_model, self.BaseModelClass = HF_CONFIGS[self.use_model]
        self.ModelClass = create_model_class(self.BaseModelClass)

        self.output_dir_name = output_dir_name
        self._total_train_batch_size = self.training_args.train_batch_size

        self.model = None
        self.tokenizer = None

        print(f'{len(self.workspace.data["examples"])} examples')
        print(f'{len(self.workspace.intents)} intents')
        print(f'{len(self.workspace.entities)} entities')
        print(f'{len(self.workspace.entity_values)} entity values')
        
        if len(self.workspace.data["examples"]) == 0:
            raise RuntimeError('There were no training examples in this workspace')

    def _load_pretrained(self):
        self.tokenizer = xf.AutoTokenizer.from_pretrained(
            self.base_model, use_fast=True)
        config = self._load_config()
        self.model = self.ModelClass.from_pretrained(
            self.base_model, config=config)

    def _load_config(self):
        config = xf.AutoConfig.from_pretrained(self.base_model)
        config.task_specific_params = {
            'intents': {
                'num_intents': len(self.workspace.intent_ids),
                'id2label': dict(self.workspace.intent_ids),
            },

            'entities': {
                'num_entities': len(self.workspace.entity_ids),
                'entity_values': dict(self.workspace.entity_values_entities),
                'id2label': dict(self.workspace.entity_ids),
            },
            'meta': {
                'use_model': self.use_model,
            }
        }
        return config

    def _train_dataset(self):
        ds = self.workspace.labels_for_examples(
            self.tokenizer, self.workspace.data['examples'])
        n = len(ds.encodings)

        tds = tf.data.Dataset.from_tensor_slices(dict(ds)).repeat()
        totally_labels = tf.data.Dataset.from_tensors(
            tf.convert_to_tensor(0)).repeat()
        train_dataset = tf.data.Dataset.zip((tds, totally_labels)).shuffle(
            n).padded_batch(self._total_train_batch_size)

        return train_dataset, n

    def _fit(self, train_batches_per_epoch, train_dataset):
        optimizer, _ = xf.create_optimizer(
            init_lr=self.training_args.learning_rate,
            num_train_steps=int(
                self.training_args.num_train_epochs * train_batches_per_epoch),
            num_warmup_steps=self.training_args.warmup_steps,
            adam_beta1=self.training_args.adam_beta1,
            adam_beta2=self.training_args.adam_beta2,
            adam_epsilon=self.training_args.adam_epsilon,
            weight_decay_rate=self.training_args.weight_decay,
        )

        # The convention in hf's transformers lib is to output the loss as y_pred, and to pass dummy losses/labels
        # whenever the framework requires one.
        def dummy_loss(_, y_pred):
            return tf.reduce_sum(y_pred)

        self.model.compile(loss={"loss": dummy_loss}, optimizer=optimizer)
        self.model.fit(
            train_dataset,
            epochs=int(self.training_args.num_train_epochs),
            steps_per_epoch=train_batches_per_epoch
        )

    def train(self):
        if self.model is None:
            self._load_pretrained()

        train_dataset, n = self._train_dataset()

        train_batches_per_epoch = n // self._total_train_batch_size

        self._fit(train_batches_per_epoch, train_dataset)

        if self.output_dir_name is not None:
            self.model.save_pretrained(self.output_dir_name)
            self.tokenizer.save_pretrained(self.output_dir_name)

        return IntentEntityPipeline(self.model, self.tokenizer)


def load(dir_name):
    # Load the config file from the model's saved output
    import os
    import json
    with open(os.path.join(dir_name, 'config.json'), 'r') as f:
        config = json.load(f)

    meta_config = config['task_specific_params']['meta']
    _, BaseModelClass = HF_CONFIGS[meta_config['use_model']]

    from model import create_model_class
    ModelClass = create_model_class(BaseModelClass)

    tokenizer = xf.AutoTokenizer.from_pretrained(dir_name, use_fast=True)
    model = ModelClass.from_pretrained(dir_name)

    return IntentEntityPipeline(model, tokenizer)


class IntentEntityPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config = self.model.config

        self.intent_ids = {int(
            k): v for k, v in self.config.task_specific_params['intents']['id2label'].items()}
        self.entity_ids = {int(
            k): v for k, v in self.config.task_specific_params['entities']['id2label'].items()}
        self.entity_values = {
            k: v for k, v in self.config.task_specific_params['entities']['entity_values'].items()}

        # Monkey-patch a few things to re-use huggingface's built-in entity pipeline
        self.config.id2label = {k: self._mangle_entity_name(
            v) for k, v in self.entity_ids.items()}
        self.config.label2id = {v: k for k, v in self.entity_ids.items()}
        self.entity_pipeline = TokenClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer, framework='tf')

    def _mangle_entity_name(self, name) -> str:
        prefix = name[0:2]
        name = name[2:].replace('-', '_')
        return prefix + name

    def _unmangle_entity_name(self, name) -> str:
        prefix = name[0:2]
        name = name[2:].replace('_', '-')
        return prefix + name

    def _intent_matches(self, logits) -> List[List[service_pb2.IntentMatch]]:
        # Sort through the last dimension and reverse it
        ordering = logits.argsort(axis=-1)[:, ::-1]

        matches = []
        for phrase_index, orders in enumerate(ordering):
            phrase_matches = []
            for intent_index in orders:
                score = logits[phrase_index][intent_index]
                # Skip intents below 1% confidence
                if score < 0.01:
                    break
                intent_id = self.intent_ids[intent_index]
                phrase_matches.append(service_pb2.IntentMatch(
                    id=intent_id, name=intent_id, score=score))

            matches.append(phrase_matches)

        return matches

    def _entity_matches(self, inputs, tokenized, logits, offset_mapping, special_tokens_mask) -> List[
            List[service_pb2.EntityMatch]]:

        # Check if the tokenizer's convert_tokens_to_string function is broken
        out = self.tokenizer.convert_tokens_to_string(["hi", "there"])
        if isinstance(out, list):
            from functools import partial
            self.tokenizer.convert_tokens_to_string = partial(
                xf.BertTokenizer.convert_tokens_to_string, None)

        matches = []

        input_ids = tokenized['input_ids'].numpy()
        logits = logits.numpy()
        offset_mapping = offset_mapping.numpy()
        special_tokens_mask = special_tokens_mask.numpy()

        for phrase, input_ids, logits, off, st in zip(inputs, input_ids, logits, offset_mapping,
                                                      special_tokens_mask):
            phrase_matches = []
            entities = self.entity_pipeline.postprocess({
                'logits': logits[None, ],
                'input_ids': input_ids[None, ],
                'sentence': phrase,
                'special_tokens_mask': st[None, ],
                'offset_mapping': off[None, ],
            }, aggregation_strategy=AggregationStrategy.MAX)

            for m in entities:
                entity_value_id = self._unmangle_entity_name(m['entity_group'])
                entity_id = self.entity_values[entity_value_id]
                entity = service_pb2.EntityReference(entity_id=entity_id, entity_value_id=entity_value_id, text=m['word'])
                span = service_pb2.Span(start=m['start'], end=m['end'])
                phrase_matches.append(service_pb2.EntityMatch(entity=entity, score=m['score'], span=span))

            matches.append(phrase_matches)

        return matches

    def __call__(self, inputs: List[str]) -> List[service_pb2.Predictions]:
        """
        Runs the intent & entity pipeline on all strings simultaneously
        """

        tokenized = self.tokenizer(inputs,
                                   return_tensors='tf',
                                   return_offsets_mapping=True,
                                   return_special_tokens_mask=True,
                                   padding=True,
                                   # TODO: max_length=256 may be too small for some datasets
                                   truncation=True,
                                   max_length=256,
                                   pad_to_multiple_of=8)

        offset_mapping = tokenized.pop('offset_mapping')
        special_tokens_mask = tokenized.pop('special_tokens_mask')

        outputs = self.model(**tokenized)

        all_intent_matches = self._intent_matches(
            outputs['intent_logits'].numpy())
        all_entity_matches = self._entity_matches(inputs,
                                                  tokenized,
                                                  outputs['entity_logits'],
                                                  offset_mapping,
                                                  special_tokens_mask)

        predictions = []
        for intent_matches, entity_matches in zip(all_intent_matches, all_entity_matches):
            predictions.append(service_pb2.Predictions(
                matches=intent_matches, entity_matches=entity_matches))

        return predictions
