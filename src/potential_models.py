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
        # print(len(pa_tup))
        # exit()
        if len(pa_tup) > 2:
            x = [i[0] for i in pa_tup]
            y = [i[1] for i in pa_tup]
        else:
            x = pa_tup[0]
            y = pa_tup[1]

        # print(self.pred_embeddings(torch.tensor(x)))
        # # print(pa_tup[0])
        # exit()
        p_embed = self.pred_embeddings(torch.tensor(x))

        # dropout
        # p_embed = nn.functional.dropout(p_embed, p=0.2, training=not predict, inplace=True)

        a_embed = self.argument_embeddings(torch.tensor(y))

        # dropout
        # a_embed = nn.functional.dropout(a_embed, p=0.2, training=not predict, inplace=True)

        # NOT SURE ABOUT THIS, BUT FIXES
        # RuntimeError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
        if predict:
            # print(len(pa_tup))
            pa_embed = torch.cat((p_embed,a_embed)).view(1, -1)
        else:
            pa_embed = torch.cat((p_embed,a_embed), dim=1)

        # print(pa_embed.shape)


        # linearity_in
        x = self.linear_in(pa_embed)

        # non_linearity
        nl_out = self.non_linearity(x)

        # dropout
        # nl_out = nn.functional.dropout(nl_out, p=0.2, training=not predict)


        # linear_out
        y = self.linear_out(nl_out)


        # softmax
        s_max = F.log_softmax(y, dim=1)
        # print(s_max)
        # exit()


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
