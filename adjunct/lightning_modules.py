import lightning as pl
import torchmetrics
import torch


import lightning as pl
import torchmetrics
import torch


class TorchWrapperForSequenceClassification(pl.LightningModule):
    def __init__(self, encoder, n_labels, dropout):
        super(TorchWrapperForSequenceClassification, self).__init__()
        self.encoder = encoder
        self.n_labels = n_labels
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(encoder.config.hidden_size, n_labels),
        )
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.val_loss = torch.nn.CrossEntropyLoss()
        self.test_loss = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_labels)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_labels)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_labels)

    def forward(self, inputs):
        enc_feats = self.encoder(**inputs)
        pooled_feats = torch.mean(enc_feats.last_hidden_state, dim=1)
        return self.classifier(pooled_feats)

    def training_step(self, batch, batch_idx):
        data = batch
        labels = data.pop("labels")
        logits = self.forward(data)
        loss = self.train_loss(logits, labels.long())
        acc = self.train_acc(logits, labels.long())
        self.log_dict({"acc": acc, "loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        labels = data.pop("labels")
        logits = self.forward(data)
        loss = self.val_loss(logits, labels.long())
        acc = self.val_acc(logits, labels.long())
        self.log_dict({"acc": acc, "loss": loss})

    def test_step(self, batch, batch_idx):
        data = batch
        labels = data.pop("labels")
        logits = self.forward(data)
        loss = self.test_loss(logits, labels.long())
        acc = self.test_acc(logits, labels.long())
        self.log_dict({"acc": acc, "loss": loss})


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
