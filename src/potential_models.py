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
        x = [i[0] for i in pa_tup]
        y = [i[1] for i in pa_tup]

        p_embed = self.pred_embeddings(torch.tensor(x))
        # dropout
        p_embed = nn.functional.dropout(p_embed, p=0.2, training=not predict, inplace=True)

        a_embed = self.argument_embeddings(torch.tensor(y))
        # dropout
        a_embed = nn.functional.dropout(a_embed, p=0.2, training=not predict, inplace=True)

        pa_embed = torch.cat((p_embed,a_embed), dim=1)

        # linearity_in
        x = self.linear_in(pa_embed)

        # non_linearity
        nl_out = self.non_linearity(x)
        # dropout
        nl_out = nn.functional.dropout(nl_out, p=0.2, training=not predict)

        # linear_out
        y = self.linear_out(nl_out)

        # softmax
        s_max = F.log_softmax(y, dim=1)

        # X-Ent Loss (?)
        return s_max
