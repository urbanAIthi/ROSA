import torch
import torch.nn as nn
from einops import rearrange


class MaskedSequenceTransformer(nn.Module):
    """
    A transformer-based model for spatiotemporal prediction of agent features in traffic scenarios
    using attention mechanisms that can optionally be restricted to certain agents or time steps.

    The model processes masked sequences of traffic data, allowing for attention masking based
    on agent identity and presence in the sequence. It supports optional positional and agent
    embeddings and produces predictions for each agent at the next time step.

    Args:
        sequence_len (int): Number of historical time steps in each input sample.
        max_agents (int): Maximum number of agents per time step in the input.
        full_attention (bool): Whether to use full attention across all agents and time steps. Defaults to False.
        agent_only_attention (bool): Whether to restrict attention to only the same agents across time steps. Defaults to False.
        embed_dim (int): Dimension of the embedding vector.
        num_heads (int): Number of attention heads used in the transformer encoder. Enables multi-head attention.
        dropout (float): Dropout probability applied within the transformer layers for regularization. Prevents overfitting.
        num_layers (int): Number of stacked transformer encoder layers.
        use_pos_embed (bool): Whether to use learnable positional embeddings for each time step. Defaults to True.
        use_agent_embed (bool): Whether to use learnable agent embeddings for each agent ID. Defaults to False.
    """

    def __init__(self, sequence_len: int, max_agents: int, full_attention: bool = False,
                 agent_only_attention: bool = False,
                 embed_dim: int = 64, num_heads: int = 8, dropout: float = 0.1, num_layers: int = 4,
                 use_pos_embed: bool = True, use_agent_embed: bool = False) -> None:

        super(MaskedSequenceTransformer, self).__init__()

        self.sequence_len = sequence_len
        self.max_agents = max_agents
        self.full_attention = full_attention
        self.agent_only_attention = agent_only_attention
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_pos_embed = use_pos_embed
        self.use_agent_embed = use_agent_embed

        # Define a transformer encoder with multiple layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Linear embedding layer to project raw input to embed_dim
        self.raw_traffic_embedder = nn.Linear(13, self.embed_dim)

        # Positional and agent embeddings
        self.pos_embedding = nn.Embedding(self.sequence_len, self.embed_dim)
        self.agent_embedding = nn.Embedding(self.max_agents, self.embed_dim)

        # Linear output layer to predict the features
        self.prediction_head = nn.Linear(self.embed_dim, 13)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MaskedSequenceTransformer.

        Processes a batch of masked agent trajectories across multiple time steps, applying
        embeddings, attention masking, and transformer encoding to predict agent states
        at the next time step.

        Args:
            input_sequence (torch.Tensor): A 4D tensor of shape (batch_size, sequence_len, max_agents, num_features). The tensor represents agent data over time.
                Zero-padded agents are automatically masked.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, max_agents, num_features)
                representing the predicted features (e.g., positions, speeds) for each
                agent at the next time step.
        """

        batch_size, sequence_len, max_agents, num_features = input_sequence.shape
        assert sequence_len == self.sequence_len
        assert max_agents == self.max_agents
        assert num_features == 13

        #Create a mask for zero inputs to avoid attending to zero inputs
        zero_mask = (input_sequence.sum(dim=-1) == 0)  # Shape: (batch_size, sequence_len, max_agents)
        sequence_zero_mask = zero_mask.all(dim=-1)  # Shape: (batch_size, sequence_len)

        # Also ignore entire time steps where all agents are missing
        expanded_sequence_zero_mask = sequence_zero_mask.unsqueeze(-1).expand(batch_size, self.sequence_len, self.max_agents)  # Shape: (batch_size, sequence_len, max_agents)
        expanded_sequence_zero_mask = expanded_sequence_zero_mask.reshape(batch_size, -1)  # Shape: (batch_size, sequence_len * max_agents)

        # Padding mask used for src_key_padding_mask in transformer
        padding_mask = zero_mask.view(batch_size, -1)  # Shape: (batch_size, sequence_len * max_agents)

        # Combine both masks
        padding_mask = padding_mask | expanded_sequence_zero_mask

        # Embed raw input features
        input_sequence = self.raw_traffic_embedder(input_sequence)  # Shape: (batch_size, sequence_len, max_agents, embed_dim)

        # Flatten the sequence for the transformer: combine time and agent dimensions
        flattened_sequence = rearrange(input_sequence, 'b s v e -> b (s v) e')  # Shape: (batch_size, sequence_len * max_agents, embed_dim)

        # Create time step indices for each position in the flattened sequence
        time_step_indices = torch.arange(self.sequence_len).unsqueeze(1).repeat(1, self.max_agents).view(-1).to(input_sequence.device)
        time_step_indices = time_step_indices.unsqueeze(0).repeat(batch_size, 1)  # Shape: (batch_size, sequence_len * max_agents)

        # Create agent indices for each position in the flattened sequence
        agent_indices = torch.arange(self.max_agents).unsqueeze(0).repeat(self.sequence_len, 1).view(-1).to(input_sequence.device)
        agent_indices = agent_indices.unsqueeze(0).repeat(batch_size, 1)  # Shape: (batch_size, sequence_len * max_agents)

        # Create attention masks
        time_step_diff = time_step_indices.unsqueeze(2) != time_step_indices.unsqueeze(1)
        agent_diff = agent_indices.unsqueeze(2) != agent_indices.unsqueeze(1)

        if self.agent_only_attention:
            attention_mask = agent_diff  # Shape: (batch_size, sequence_len * max_agents, sequence_len * max_agents)
        else:
            # Mask positions where both time steps and agent indices are different
            attention_mask = time_step_diff & agent_diff  # Shape: (batch_size, sequence_len * max_agents, sequence_len * max_agents)

        # Incorporate the zero mask
        # Prevent attention to padded positions
        zero_mask_flat = zero_mask.view(batch_size, -1)  # Shape: (batch_size, sequence_len * max_agents)
        zero_mask_expanded = zero_mask_flat.unsqueeze(1) | zero_mask_flat.unsqueeze(2)

        if self.full_attention:
            attention_mask = zero_mask_expanded
        else:
            attention_mask = attention_mask | zero_mask_expanded  # Combine with existing attention mask

        # Convert boolean attention mask for the transformer (needs to be float with -inf where masked)
        attention_mask = attention_mask.float()
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float(0.0))

        # Expand attention mask for multiple heads
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = attention_mask.view(batch_size * self.num_heads, sequence_len * max_agents, sequence_len * max_agents)

        # Add positional and agent embeddings (if enabled)
        if self.use_pos_embed:
            pos_embed = self.pos_embedding(time_step_indices)  # Shape: (batch_size, sequence_len * max_agents, embed_dim)
        else:
            pos_embed = torch.zeros(batch_size, sequence_len * max_agents, self.embed_dim).to(input_sequence.device)

        if self.use_agent_embed:
            agent_embed = self.agent_embedding(agent_indices)  # Shape: (batch_size, sequence_len * max_agents, embed_dim)
        else:
            agent_embed = torch.zeros(batch_size, sequence_len * max_agents, self.embed_dim).to(input_sequence.device)

        # Add positional and agent information to the embeddings
        flattened_sequence = flattened_sequence + pos_embed + agent_embed

        # Transpose for transformer input: (sequence_length, batch_size, embed_dim)
        transformer_input = flattened_sequence.transpose(0, 1)

        # Apply transformer with attention and padding masks
        transformer_output = self.transformer_encoder(transformer_input, mask=attention_mask, src_key_padding_mask=padding_mask)

        # Transpose back to original format: (batch_size, sequence_len * max_agents, embed_dim)
        transformer_output = transformer_output.transpose(0, 1)
        transformer_output = transformer_output.view(batch_size, sequence_len, max_agents, self.embed_dim)

        # Use output only from the last time step
        transformer_output = transformer_output[:, -1, :, :]  # Shape: (batch_size, max_agents, embed_dim)

        # Predict original features
        # Apply prediction head
        predictions = self.prediction_head(transformer_output)  # Shape: (batch_size, max_agents, num_features)

        return predictions


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")