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

    def forward(self, inputs):
        enc_feats = self.encoder(**inputs)
        pooled_feats = torch.mean(enc_feats.last_hidden_state, dim=1)
        return self.classifier(pooled_feats)

    def training_step(self, batch, batch_idx):
        data = batch
        labels = data.pop("labels")
        logits = self.forward(data)
        loss = self.train_loss(logits, labels.long())
        self.log("training loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        labels = data.pop("labels")
        logits = self.forward(data)
        loss = self.val_loss(logits, labels.long())
        self.log("val loss: ", loss)

    def test_step(self, batch, batch_idx):
        data = batch
        labels = data.pop("labels")
        logits = self.forward(data)
        loss = self.val_loss(logits, labels.long())
        self.log("test loss: ", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
