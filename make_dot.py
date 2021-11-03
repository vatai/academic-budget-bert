import torch
from torchviz import make_dot
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

output = model(
    input_ids=input_ids,
    attention_mask=input_mask,
    token_type_ids=token_type_ids,
    labels=torch.LongTensor([[1, 0]]),
)
pred, loss = output.logits, output.loss

params = dict(list(model.named_parameters()))
format = "svg"
make_dot(pred, params=params).render("pred_hf", format=format)
make_dot(loss, params=params).render("loss_hf", format=format)
