import torch
import torch.nn as nn
import torch.nn.functional as F

class RolePredictor(nn.Module):
    def __init__(self, predicate_size, argument_size, roles_size, drop_p,
                                                    embed_size, linearity_size):
        super(RolePredictor, self).__init__()

        self.pred_embeddings = nn.Embedding(predicate_size, embed_size)
        self.argument_embeddings = nn.Embedding(argument_size, embed_size)

        self.linear_in = nn.Linear(embed_size * 2, linearity_size)
        self.linear_out = nn.Linear(linearity_size, roles_size)

        self.non_linearity = nn.Tanh()
        self.dropout = drop_p


    def forward(self, pa_tup, predict=False):
        # Separate preds and args
        x, y = zip(*pa_tup)

        p_embed = self.pred_embeddings(torch.tensor(x))
        # dropout
        p_embed = nn.functional.dropout(p_embed, p=self.dropout,
                                             training=not predict, inplace=True)

        a_embed = self.argument_embeddings(torch.tensor(y))
        # dropout
        a_embed = nn.functional.dropout(a_embed, p=self.dropout,
                                             training=not predict, inplace=True)

        pa_embed = torch.cat((p_embed,a_embed), dim=1)

        # linearity_in
        x = self.linear_in(pa_embed)

        # non_linearity
        nl_out = self.non_linearity(x)
        # dropout
        nl_out = nn.functional.dropout(nl_out, p=self.dropout,
                                                           training=not predict)

        # linear_out
        y = self.linear_out(nl_out)

        return y
