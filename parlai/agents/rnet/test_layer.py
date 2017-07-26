from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
# from layers import GatedMatchRNN
from layers import PointerNetwork, StackedBRNN

batch = 3
document = 3
question = 2
hidden = 4


d0 = torch.LongTensor([[1, 3]])
d1 = torch.LongTensor([[2, 3, 4]])
d2 = torch.LongTensor([[4]])

q0 = torch.LongTensor([[2]])
q1 = torch.LongTensor([[1, 2]])
q2 = torch.LongTensor([[1]])

# x = torch.zeros(3, 3, 1)
# y = torch.zeros(3, 2, 1)
x = torch.LongTensor(batch, document, 1).fill_(0)
x_mask = torch.ByteTensor(3, 3).fill_(1)
y = torch.LongTensor(batch, question, 1).fill_(0)
y_mask = torch.ByteTensor(3, 2).fill_(1)

for i in range(3):
    x[i][:eval('d%s' % i).size(-1)] = eval('d%s' % i)
    x_mask[i, :eval('d%s' % i).size(-1)].fill_(0)
    y[i][:eval('q%s' % i).size(-1)] = eval('q%s' % i)
    y_mask[i, :eval('q%s' % i).size(-1)].fill_(0)

emb = nn.Embedding(5, 4)
emb.weight.data[0] = torch.zeros(4)

x = x.squeeze()
y = y.squeeze()

x = Variable(x)
y = Variable(y)
x_mask = Variable(x_mask)
y_mask = Variable(y_mask)

# x_emb = emb(x.view(-1, x.size(-1))).view(x.size())
# y_emb = emb(y.view(-1, y.size(-1))).view(y.size())
x_emb = emb(x)
y_emb = emb(y)

# notice after embedding unknow or padding word is not non, so
# we must mask this vector to get the right matrix.


# m = GatedMatchRNN(4, padding=True)
# m = PointerNetwork(4, 4)
# output = m(x_emb, x_mask, y_emb, y_mask)
m = StackedBRNN(4, 8, 3, 0.2, False, nn.GRU, True, True)
output = m(x_emb, x_mask)
print(output)
