# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from external_integration.v1alpha1 import discovery_pb2 as external__integration_dot_v1alpha1_dot_discovery__pb2


class DiscoveryStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetCapabilities = channel.unary_unary(
                '/zia.ai.external_integration.v1alpha1.Discovery/GetCapabilities',
                request_serializer=external__integration_dot_v1alpha1_dot_discovery__pb2.GetCapabilitiesRequest.SerializeToString,
                response_deserializer=external__integration_dot_v1alpha1_dot_discovery__pb2.GetCapabilitiesResponse.FromString,
                )


class DiscoveryServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetCapabilities(self, request, context):
        """Gets the services supported by this integration
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DiscoveryServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetCapabilities': grpc.unary_unary_rpc_method_handler(
                    servicer.GetCapabilities,
                    request_deserializer=external__integration_dot_v1alpha1_dot_discovery__pb2.GetCapabilitiesRequest.FromString,
                    response_serializer=external__integration_dot_v1alpha1_dot_discovery__pb2.GetCapabilitiesResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'zia.ai.external_integration.v1alpha1.Discovery', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Discovery(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetCapabilities(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/zia.ai.external_integration.v1alpha1.Discovery/GetCapabilities',
            external__integration_dot_v1alpha1_dot_discovery__pb2.GetCapabilitiesRequest.SerializeToString,
            external__integration_dot_v1alpha1_dot_discovery__pb2.GetCapabilitiesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)