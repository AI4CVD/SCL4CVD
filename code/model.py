import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
from SupConLoss import SupConLoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        if args.mlp:
            self.MLP = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout_probability),
                nn.Linear(config.hidden_size, config.num_labels)
             )
        else:
            self.MLP = nn.Dropout(args.dropout_probability)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        #x = x.reshape(-1,x.size(-1)*2)
        #print("x:",x.shape)
        #x = self.dropout(x)
        x = self.MLP(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.classifier=RobertaClassificationHead(config, args)    
        
    def forward(self, inputs_ids=None,labels=None): 

        #embedding
        outputs = self.encoder.roberta(inputs_ids,attention_mask=inputs_ids.ne(1))[0]
        logits=self.classifier(outputs)
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            #cross_loss = self.bce_loss(prob[:, 0], labels)
            cross_loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            cross_loss = -cross_loss.mean()
            contrastive = SupConLoss()(outputs, labels)
            loss = (self.args.lam * contrastive) + (1 - self.args.lam) * (cross_loss)
            return loss,prob
        else:
            return prob
      
        

       