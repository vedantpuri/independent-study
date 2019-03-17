import torch
import torch.nn as nn
import torch.nn.functional as F

class RolePredictor(nn.Module):
    def __init__(self, predicate_size, argument_size, roles_size):
        super(RolePredictor, self).__init__()

        self.pred_embeddings = nn.Embedding(predicate_size, 25)
        self.argument_embeddings = nn.Embedding(argument_size, 25)

        # linear mapping defined here (could be mapped to any number)
        self.linear_in = nn.Linear(50, 128)
        self.linear_out = nn.Linear(128, roles_size)


    def forward(self, pa_tup, predict=False):
        p_embed = self.pred_embeddings(torch.tensor(pa_tup[0]))

        # dropout

        a_embed = self.argument_embeddings(torch.tensor(pa_tup[1]))

        # dropout

        pa_embed = torch.cat((p_embed,a_embed))

        # linearity_in
        # non_linearity

        # dropout

        # linear_out
        # softmax
        # X-Ent Loss (?)
        print("p", p_embed)
        print("a", a_embed)
        print("pa", pa_embed)
        return pa_tup




m = RolePredictor(409, 1831, 7)
# print(m.pred_embeddings[0].view((1, -1)))
m((8, 78))
