import torch


class Attention3D(torch.nn.Module):
    """
        Converts 3D tensor (context_length, batch_size, n_features) 
        to 3D tensor (context_length, batch_size, n_features)
        with attention mechanism for recurrent networks
        https://jalammar.github.io/illustrated-transformer/
    """
    def __init__(self, n_features):
        super(Attention3D, self).__init__()

        self.query, self.key, self.value = [
            torch.nn.Linear(n_features, n_features) 
            for _ in range(3)
        ]

        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, input):
        context_length, batch_size, n_features = input.shape
        output = input.clone().detach()
        
        for i in range(batch_size):
            input_2d = output[:, i: i+1, :].clone().detach().reshape(context_length, n_features)
            Q, K, V = self.query(input_2d), self.key(input_2d), self.value(input_2d)

            attention_2d = self.softmax( (Q @ K.T) / n_features**0.5) @ V
            output[:, i: i+1, :] = attention_2d.reshape(context_length, 1, n_features)
        
        return output


class Attention2D(torch.nn.Module):
    """
        Converts 3D tensor (context_length, batch_size, n_features) 
        to 2D tensor (batch_size, 1) as a weighted sum of contexts, 
        like here https://arxiv.org/pdf/1803.09473.pdf
    """
    def __init__(self, input: torch.Tensor):
        super(Attention2D).__init__()

    def forward(self, input):
        # TODO using Linear(n_features, 1)
        pass
