import unittest, os

from .humanfirst.protobuf.external_integration.v1alpha1 import discovery_pb2, discovery_pb2_grpc, models_pb2, models_pb2_grpc
from .humanfirst.protobuf.external_nlu.v1alpha1 import service_pb2_grpc, service_pb2
from .humanfirst.protobuf.playbook.data.config.v1alpha1 import config_pb2

from .model_service import ModelService

class TestIntegration(unittest.TestCase):
    def setUp(self) -> None:
        # Make sure our test is deterministic
        import tensorflow as tf
        tf.random.set_seed(42)

    def test_train(self):
        """
        Trains a dummy model and make it return predictions from the loaded model
        """

        current_dir = os.path.dirname(os.path.abspath(__file__))
        fixtures_path = os.path.join(current_dir, 'test_fixtures')

        # Read the file as binary since the service is expecting `bytes` from the request message
        with open(os.path.join(fixtures_path, 'workspace.json'), 'rb') as f:
            workspace_data = f.read()

        svc = ModelService()

        # Make sure the service expects HF JSON as its data format
        train_params = svc.GetTrainParameters(models_pb2.GetTrainParametersRequest(), None)
        self.assertEqual(train_params.data_format, config_pb2.IntentsDataFormat.INTENTS_FORMAT_HF_JSON)

        train_response = svc.TrainModel(models_pb2.TrainModelRequest(
            data_format=config_pb2.IntentsDataFormat.INTENTS_FORMAT_HF_JSON,
            format_options=config_pb2.IntentsDataOptions(),
            data=workspace_data), None)
        
        # The model id should be a non-empty string
        self.assertNotEqual(train_response.model.id, "")
        

        # Classify a test example
        example = service_pb2.Example(contents="hi there good morning new york")
        classify_response = svc.Classify(models_pb2.ClassifyRequest(model_id=train_response.model.id, examples=[example]), None)

        self.assertEqual(classify_response.predictions[0].matches[0].name, 'intent-GYEKPUFIJ5MEVINPZ6TNZXJS')
        morning_entity = classify_response.predictions[0].entity_matches[0]
        self.assertEqual(morning_entity.entity.entity_id, 'entity-TNOSTSLZ6BC4XEOFBP5GF4ZH')
        self.assertEqual(morning_entity.entity.entity_value_id, 'entval-BF67VRX5UVA5DJTMCEUX4R2J')
        self.assertEqual(morning_entity.entity.text, 'morning')
        self.assertEqual(morning_entity.span.start, 14)
        self.assertEqual(morning_entity.span.end, 21)
