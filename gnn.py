import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from typing import List, Tuple, Dict, Sequence, Any
import inspect


class AdjacencyList:
    """represent the topology of a graph"""
    def __init__(self, node_num: int, adj_list: List[Tuple], use_cuda: bool = False):
        self.node_num = node_num
        self.edge_num = len(adj_list)
        self.use_cuda = use_cuda
        data = (torch.cuda if use_cuda else torch).LongTensor(self.edge_num, 2).zero_()
        for row_id, (node_s, node_t) in enumerate(adj_list):
            data[row_id, 0] = node_s
            data[row_id, 1] = node_t

        self.data = Variable(data)

    def __getitem__(self, item):
        return self.data[item]


class GatedGraphNeuralNetwork(nn.Module):
    def __init__(self, hidden_size, num_edge_types, layer_timesteps,
                 residual_connections, state_to_message_dropout=0.3,
                 rnn_dropout=0.3,
                 use_cuda=False):
        local_vars = locals()
        arg_spec = inspect.getfullargspec(self.__init__)
        for arg_name in filter(lambda x: x != 'self', arg_spec.args):
            print('%s: %s' % (arg_name, local_vars[arg_name]))
            self.__dict__[arg_name] = local_vars[arg_name]

        super(GatedGraphNeuralNetwork, self).__init__()

        # Prepare linear transformations from node states to messages, for each layer and each edge type
        # Prepare rnn cells for each layer
        self.state_to_message_linears = []
        self.rnn_cells = []
        for layer_idx in range(len(self.layer_timesteps)):
            state_to_msg_linears_cur_layer = []
            # Initiate a linear transformation for each edge type
            for edge_type_j in range(self.num_edge_types):
                # TODO: glorot_init?
                state_to_msg_linear_layer_i_type_j = nn.Linear(self.hidden_size, self.hidden_size)
                setattr(self,
                        'state_to_message_linear_layer%d_type%d' % (layer_idx, edge_type_j),
                        state_to_msg_linear_layer_i_type_j)

                state_to_msg_linears_cur_layer.append(state_to_msg_linear_layer_i_type_j)
            self.state_to_message_linears.append(state_to_msg_linears_cur_layer)

            layer_residual_connections = self.residual_connections.get(layer_idx, [])
            rnn_cell_layer_i = nn.GRUCell(self.hidden_size * (1 + len(layer_residual_connections)), self.hidden_size)
            setattr(self, 'rnn_cell_layer%d' % layer_idx, rnn_cell_layer_i)
            self.rnn_cells.append(rnn_cell_layer_i)

        self.state_to_message_dropout_layer = nn.Dropout(self.state_to_message_dropout)
        self.rnn_dropout_layer = nn.Dropout(self.rnn_dropout)

    def forward(self,
                initial_node_representation: Variable,
                adjacency_lists: List[AdjacencyList]) -> Variable:
        return self.compute_final_node_representations(initial_node_representation, adjacency_lists)

    def compute_final_node_representations(self,
                                           initial_node_representation: Variable,
                                           adjacency_lists: List[AdjacencyList]) -> Variable:
        # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer = [initial_node_representation]

        for layer_idx, num_timesteps in enumerate(self.layer_timesteps):
            # Used shape abbreviations:
            #   V ~ number of nodes
            #   D ~ state dimension
            #   E ~ number of edges of current type
            #   M ~ number of messages (sum of all E)

            # Extract residual messages, if any:
            layer_residual_connections = self.residual_connections.get(layer_idx, [])
            # List[(V, D)]
            layer_residual_states: List[torch.FloatTensor] = [node_states_per_layer[residual_layer_idx]
                                                              for residual_layer_idx in layer_residual_connections]

            # Record new states for this layer. Initialised to last state, but will be updated below:
            node_states_for_this_layer = node_states_per_layer[-1]
            # For each message propagation step
            for t in range(num_timesteps):
                messages: List[torch.FloatTensor] = []  # list of tensors of messages of shape [E, D]
                message_source_states: List[torch.FloatTensor] = []  # list of tensors of edge source states of shape [E, D]

                # Collect incoming messages per edge type
                for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
                    # shape [E]
                    edge_sources = adjacency_list_for_edge_type[:, 0]
                    # shape [E, D]
                    edge_source_states = node_states_for_this_layer[edge_sources]

                    f_state_to_message = self.state_to_message_linears[layer_idx][edge_type_idx]
                    # Shape [E, D]
                    all_messages_for_edge_type = self.state_to_message_dropout_layer(f_state_to_message(edge_source_states))

                    messages.append(all_messages_for_edge_type)
                    message_source_states.append(edge_source_states)

                # shape [M, D]
                messages: torch.FloatTensor = torch.cat(messages, dim=0)

                # Sum up messages that go to the same target node
                # shape [V, D]
                incoming_messages = self.get_incoming_message_sparse_matrix(adjacency_lists) @ messages

                # shape [V, D * (1 + num of residual connections)]
                incoming_information = torch.cat(layer_residual_states + [incoming_messages], dim=-1)

                # pass updated vertex features into RNN cell
                # Shape [V, D]
                updated_node_states = self.rnn_cells[layer_idx](incoming_information, node_states_for_this_layer)
                updated_node_states = self.rnn_dropout_layer(updated_node_states)
                node_states_for_this_layer = updated_node_states

            node_states_per_layer.append(node_states_for_this_layer)

        node_states_for_last_layer = node_states_per_layer[-1]
        return node_states_for_last_layer

    def get_incoming_message_sparse_matrix(self, adjacency_lists: List[AdjacencyList]) -> Variable:
        typed_edge_nums = [adj_list.edge_num for adj_list in adjacency_lists]
        all_messages_num = sum(typed_edge_nums)
        node_num = adjacency_lists[0].node_num

        T = torch.cuda if self.use_cuda else torch
        x = T.FloatTensor(node_num, all_messages_num).zero_()

        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
            # shape [E]
            target_nodes = adjacency_list_for_edge_type[:, 1]
            for msg_idx, tgt_node in enumerate(target_nodes.data):
                msg_offset = sum(typed_edge_nums[:edge_type_idx]) + msg_idx
                x[tgt_node, msg_offset] = 1

        return Variable(x)


def main():
    gnn = GatedGraphNeuralNetwork(hidden_size=64, num_edge_types=2,
                                  layer_timesteps=[3, 5, 7, 2], residual_connections={2: [0], 3: [0, 1]})

    adj_list_type1 = AdjacencyList(node_num=4, adj_list=[(0, 2), (2, 1), (1, 3)])
    adj_list_type2 = AdjacencyList(node_num=4, adj_list=[(0, 0), (0, 1)])

    node_representations = gnn.compute_final_node_representations(initial_node_representation=Variable(torch.randn(4, 64)),
                                                                  adjacency_lists=[adj_list_type1, adj_list_type2])

    print(node_representations)


if __name__ == '__main__':
    main()
