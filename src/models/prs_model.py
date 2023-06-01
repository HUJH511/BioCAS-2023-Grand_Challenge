import os
import torch
import torch.nn as nn
import src.models as mdl

class PRS_classifier(nn.Module):
    def __init__(self, opt, num_classes=10, pretrain=True):
        super().__init__()
        if pretrain == True:
            model_path = os.path.join(opt.save_folder, opt.ckpt)
            model_info = torch.load(model_path)
            self.encoder = mdl.SupConResNet(
                name=opt.model, 
                head=opt.head, 
                feat_dim=opt.embedding_size
            )
            self.encoder.load_state_dict(model_info["model"])
            self.classifier = mdl.LinearClassifier(
                dim_in=opt.embedding_size, 
                num_classes=num_classes
            )
        else:
            self.encoder = mdl.SupConResNet(
                name=opt.model_name, 
                head=opt.head, 
                feat_dim=opt.embedding_size
            )
            self.classifier = mdl.LinearClassifier(
                dim_in=opt.embedding_size, 
                num_classes=num_classes
            )

    def forward(self, x):
        encode = self.encoder(x)
        outputs = self.classifier(encode)
        return outputs
    

class PRS_Model(nn.Module):
    def __init__(self, model, head, embedding_size, num_classes=10):
        super().__init__()
        self.encoder = mdl.SupConResNet(
            name=model, 
            head=head, 
            feat_dim=embedding_size
        )
        self.classifier = mdl.LinearClassifier(
            dim_in=embedding_size, 
            num_classes=num_classes
        )

    def forward(self, x):
        encode = self.encoder(x)
        outputs = self.classifier(encode)
        return outputs