import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoModel, DistilBertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from diplom.bert_training.configs import *


class BertHudLit(pl.LightningModule):
    def __init__(self, model_params: MyBertConfig, train_params: MyTrainConfig,
                 optim_params: MyOptimizerConfig,
                 metrics_params: MyMetricsConfig):
        super().__init__()
        self.optim_params = optim_params
        self.metric_params = metrics_params
        hid_dim = train_params.head_hidd_dim
        # self.bert: DistilBertModel = AutoModel.from_pretrained(model_params.model_name,
        #                                                        num_labels=train_params.NUM_LABELS)
        self.bert = (AutoModelForSequenceClassification.
                     from_pretrained(model_params.model_name,
                                     output_attentions=False,  # Whether the model returns attentions weights.
                                     output_hidden_states=False,  # Whether the model returns all hidden-states.
                                     num_labels=train_params.NUM_LABELS))
        self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, 2 * hid_dim)
        self.dropout = torch.nn.Dropout(train_params.out_head_dropout)
        self.classifier = torch.nn.Linear(2 * hid_dim, train_params.NUM_LABELS)
        self.criterion = train_params.criterion

    # def forward(self, input_ids, attention_mask):
    #     pooled_output = self.bert(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask
    #     )  # .pooler_output
    #     y = self.dropout(pooled_output)
    #     y = self.pre_classifier(y)
    #     y = self.drop1(y)
    #     return self.classifier(y)

    def training_step(self, batch, batch_idx):
        input_ids = batch["ids"]
        attention_mask = batch["mask"]
        labels = batch["targets"]
        outputs = self.bert(input_ids, attention_mask=attention_mask, labels=labels)#self(input_ids, attention_mask)
        loss = outputs["loss"]#F.cross_entropy(outputs, labels)  # self.criterion(outputs, labels)
        metrics_res = dict()
        metrics_res["train_loss"] = loss
        self.log_dict(metrics_res, on_step=False, on_epoch=True, prog_bar=True)
        return loss  # {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["ids"]
        attention_mask = batch["mask"]
        labels = batch["targets"]
        outputs = self.bert(input_ids, attention_mask=attention_mask, labels=labels)#self(input_ids, attention_mask)
        loss = outputs["loss"]#F.cross_entropy(outputs, labels)  # self.criterion(outputs, labels)
        metrics_res = dict()
        for name, metric in zip(self.metric_params.names, self.metric_params.metrics):
            # big_val, big_idx = torch.max(outputs.data, dim=-1)
            pred = outputs["logits"]
            metrics_res[name] = metric(pred.cpu().numpy(), labels.cpu().numpy())
        metrics_res["val_loss"] = loss
        self.log_dict(metrics_res, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["ids"]
        attention_mask = batch["mask"]
        labels = batch["targets"]
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)  # self.criterion(outputs, labels)
        metrics_res = dict()
        for name, metric in zip(self.metric_params.names, self.metric_params.metrics):
            metrics_res[name] = metric(outputs, labels)
        metrics_res["test_loss"] = loss
        self.log_dict(metrics_res, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optim_params.optimizer(params=self.parameters(),
                                                **self.optim_params.optimizer_params)
        if self.optim_params.scheduler is None:
            return optimizer

        scheduler = self.optim_params.scheduler(optimizer,
                                                **self.optim_params.scheduler_params)
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )
