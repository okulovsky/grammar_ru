from torch.nn import CrossEntropyLoss


class MyTrainConfig:
    def __init__(self, NUM_LABELS,
                 criterion=CrossEntropyLoss(), head_hidd_dim=768,
                 out_head_dropout=0.3):
        self.out_head_dropout = out_head_dropout
        self.head_hidd_dim = head_hidd_dim
        self.NUM_LABELS = NUM_LABELS
        self.criterion = criterion


class MyBertConfig:
    def __init__(self, id2label, label2id, model_name="distilbert-base-uncased", ):
        self.model_name = model_name
        self.id2label = id2label
        self.label2id = label2id


class MyOptimizerConfig:
    def __init__(self, optimizer, optimizer_params, scheduler, scheduler_params):
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params


class MyMetricsConfig:
    def __init__(self, metrics, names):
        assert len(metrics) == len(names)
        self.metrics = metrics
        self.names = names
