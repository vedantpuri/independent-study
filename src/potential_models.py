import torch
import torch.nn as nn
import torch.nn.functional as F

class RolePredictor(nn.Module):
    def __init__(self, predicate_size, argument_size, roles_size):
        super(RolePredictor, self).__init__()

        self.pred_embeddings = nn.Embedding(predicate_size, 25)
        self.argument_embeddings = nn.Embedding(argument_size, 25)

        # linear mapping defined here (could be mapped to any number) [HARD-CODING]
        self.linear_in = nn.Linear(50, 128)
        self.linear_out = nn.Linear(128, roles_size)

        # non-linearity [HARD-CODING]
        self.non_linearity = nn.Tanh()


    def forward(self, pa_tup, predict=False):
        p_embed = self.pred_embeddings(torch.tensor(pa_tup[0]))

        # dropout

        a_embed = self.argument_embeddings(torch.tensor(pa_tup[1]))

        # dropout

        pa_embed = torch.cat((p_embed,a_embed)).view(1, -1)

        # linearity_in
        x = self.linear_in(pa_embed)
        # non_linearity
        nl_out = self.non_linearity(x)

        # dropout

        # linear_out
        y = self.linear_out(nl_out)

        # softmax
        s_max = F.log_softmax(y, dim=1)


        # X-Ent Loss (?)

        # Debugging
        # print("p", p_embed)
        # print("a", a_embed)
        # print("pa", pa_embed)
        # print(s_max)
        return s_max




# m = RolePredictor(409, 1831, 7)
# # # print(m.pred_embeddings[0].view((1, -1)))
# m((8, 78))
