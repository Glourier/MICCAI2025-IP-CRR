import torch
import torch.nn as nn


def hardargmax(logits):
    '''Return vector version of argmax. '''
    _, d = logits.shape
    y = torch.argmax(logits, 1)
    return onehot(y, d)


def onehot(y, num_classes):
    '''Turn discrete labels into onehot vectors. '''
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to(y.device)
    return zeros.scatter(scatter_dim, y_tensor, 1)


class FullyConnectedClassifier(nn.Module):
    def __init__(self, query_size, output_size=1):
        super().__init__()
        self.input_dim = query_size
        self.n_classes = output_size

        # Architecture
        self.layer1 = nn.Linear(self.input_dim, 2000)
        self.layer2 = nn.Linear(2000, 2000)
        self.layer3 = nn.Linear(2000, 500)
        self.layer4 = nn.Linear(500, 200)

        self.norm1 = nn.LayerNorm(2000)
        self.norm2 = nn.LayerNorm(2000)
        self.norm3 = nn.LayerNorm(500)
        self.norm4 = nn.LayerNorm(200)

        self.relu = nn.ReLU()
        self.flat = nn.Flatten()

        # heads
        self.classifier = nn.Linear(400, self.n_classes)

    def forward(self, x):
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.relu(self.norm2(self.layer2(x)))
        x = self.relu(self.norm3(self.layer3(x)))
        x = self.relu(self.norm4(self.layer4(x)))
        x = self.flat(x)
        return self.classifier(x)


class FullyConnectedQuerier(nn.Module):
    def __init__(self, query_size=16441, output_size=16441, tau=None):
        super().__init__()
        self.input_dim = query_size
        self.n_queries = output_size

        # Architecture
        self.layer1 = nn.Linear(self.input_dim, 2000)
        self.layer2 = nn.Linear(2000, 2000)
        self.layer3 = nn.Linear(2000, 500)
        self.layer4 = nn.Linear(500, 200)

        self.norm1 = nn.LayerNorm(2000)
        self.norm2 = nn.LayerNorm(2000)
        self.norm3 = nn.LayerNorm(500)
        self.norm4 = nn.LayerNorm(200)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.flat = nn.Flatten()

        # Set starting temperature
        self.tau = tau

        # heads
        self.querier = nn.Linear(400, self.n_queries)

    def update_tau(self, tau):
        self.tau = tau
        print(f'Changed temperature to: {self.tau}')

    def forward(self, x, mask=None):
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.relu(self.norm2(self.layer2(x)))
        x = self.relu(self.norm3(self.layer3(x)))
        x = self.relu(self.norm4(self.layer4(x)))
        x = self.flat(x)
        query_logits = self.querier(x)

        # querying
        if mask is not None:  # Changed mask to hard
            query_logits = query_logits.masked_fill_(mask == 1, float('-inf'))
        query = self.softmax(query_logits / self.tau)
        query = (hardargmax(query_logits) - query).detach() + query
        return query
