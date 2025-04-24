import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLPPoolMLP(nn.Module):

    @staticmethod
    def sinusoidal_positional_encoding(seq_len, d_model, device='cuda'):
        """
        Create sinusoidal positional encodings for a given sequence length and model dimensionality.
        
        Args:
            seq_len (int): The length of the sequence (number of tokens).
            d_model (int): The dimensionality of the model (hidden size).
            device (str): The device on which to create the tensor (e.g., 'cpu' or 'cuda').
        
        Returns:
            torch.Tensor: A tensor of shape (seq_len, d_model) containing the positional encodings.
        """
        # Create a matrix of shape (seq_len, d_model) to store positional encodings
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * 
                            -(math.log(1000.) / d_model))
        
        # Create the positional encodings
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in the encoding
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in the encoding
        
        return pe



    def __init__(self, input_size, encoder_hidden_sizes, decoder_hidden_sizes, dropout_prob, pool='max', pos_enc=None):
        super(MLPPoolMLP, self).__init__()

        pos_enc_dim = pos_enc if pos_enc is not None else 0

        self.encoder_layers = []
        self.encoder_layers.append(nn.Linear(input_size + pos_enc_dim, encoder_hidden_sizes[0]))
        self.encoder_layers.append(nn.ReLU())
        self.encoder_layers.append(nn.Dropout(dropout_prob))
        for i in range(1, len(encoder_hidden_sizes)):
            self.encoder_layers.append(nn.Linear(encoder_hidden_sizes[i-1], encoder_hidden_sizes[i]))
            if i != len(encoder_hidden_sizes) - 1:
                self.encoder_layers.append(nn.ReLU())
            self.encoder_layers.append(nn.Dropout(dropout_prob))
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.decoder_layers = []
        if len(decoder_hidden_sizes) > 0:
            self.decoder_layers.append(nn.Linear(encoder_hidden_sizes[-1], decoder_hidden_sizes[0]))
            self.decoder_layers.append(nn.ReLU())
            self.decoder_layers.append(nn.Dropout(dropout_prob))
            for i in range(1, len(decoder_hidden_sizes)):
                self.decoder_layers.append(nn.Linear(decoder_hidden_sizes[i-1], decoder_hidden_sizes[i]))
                self.decoder_layers.append(nn.ReLU())
                self.decoder_layers.append(nn.Dropout(dropout_prob))
            self.decoder_layers.append(nn.Linear(decoder_hidden_sizes[-1], 1))
            self.decoder = nn.Sequential(*self.decoder_layers)
        elif encoder_hidden_sizes[-1] > 1:
            self.decoder = nn.Linear(encoder_hidden_sizes[-1], 1)
        else:
            self.decoder = None
        

        self.pool = pool

        self.pos_enc = pos_enc

    def _pool(self, x):
        if self.pool == 'max':
            return torch.max(x, dim=1).values
        elif self.pool == 'mean':
            return torch.mean(x, dim=1)
        elif self.pool == 'max_plus_min':
            return torch.max(x, dim=1).values + torch.min(x, dim=1).values

    
    def forward(self, traj):
        traj = traj[:, ::20, :]
        B, L, OS = traj.shape # Batch, Length, obs size

        if self.pos_enc is not None:
            if isinstance(self.pos_enc, int):
                self.pos_enc = self.sinusoidal_positional_encoding(L, self.pos_enc)
            traj = torch.cat((traj, self.pos_enc.unsqueeze(0).repeat(B, 1, 1)), dim=2)

        x = self.encoder(traj)
        x = self._pool(x)
        if self.decoder is not None:
            x = self.decoder(x)
        x = torch.sigmoid(x)

        return x.squeeze(1)
    


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, max_seq_len=300, dropout_prob=0.3, dim_feedforward=18, output_dim=1):
        super(TransformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)  #TODO: maybe MLP instead of just embedding?
        
        self.positional_encoding = self.create_positional_encoding(max_seq_len, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout_prob, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(model_dim, output_dim)
    
    def create_positional_encoding(self, seq_len, model_dim):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        positional_encoding = torch.zeros(seq_len, model_dim)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):

        batch_size, seq_len, _ = x.size()

        x = self.embedding(x) 

        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)  # (batch_size, seq_len, model_dim)

        # Transformer encoder expects input of shape (seq_len, batch_size, model_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
        x = self.transformer_encoder(x)  # (seq_len, batch_size, model_dim)

        x = x.permute(1, 2, 0)  # (batch_size, model_dim, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch_size, model_dim)

        # Classification head
        x = self.fc(x)  # (batch_size, 1 or embedding_dim for unsup learning)
        x = torch.sigmoid(x)

        if x.shape[1] == 1:
            x = x.squeeze(1)

        return x

class StepwiseMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_prob=0.3):
        super(StepwiseMLPClassifier, self).__init__()
        
        # List to store the layers
        layers = []
        
        # Input layer to the first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        
        # Loop through hidden layers and dynamically add them
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
        
        # Final output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        # Combine layers into a Sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.model(x))
    
    def embedding(self, x):
        for layer in list(self.model.children())[:-3]:
            x = layer(x)
        return x
    
def create_model(step_shape, classifier_type, position_enc_dim=None, pool='max', dropout_prob=0.3, model_dim=128, num_heads=4, num_layers=4,
                 enc_hidden_sizes=[32, 32], dec_hidden_sizes=[32, 32], dim_feedforward=18):

    if classifier_type == "mlp_pool_mlp":
        return MLPPoolMLP(step_shape, enc_hidden_sizes, dec_hidden_sizes, dropout_prob=dropout_prob, pos_enc=position_enc_dim, pool=pool).cuda()
    elif classifier_type == 'transformer':
        return TransformerClassifier(step_shape, model_dim=model_dim,
                                     num_heads=num_heads,
                                     num_layers=num_layers,
                                     dropout_prob=dropout_prob,
                                     dim_feedforward=dim_feedforward).cuda()