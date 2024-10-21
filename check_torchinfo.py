from torchinfo import summary
from cross_models.cross_former import CrossformerCircuit

model = CrossformerCircuit(data_dim=5, in_len=501, out_len=501, seg_len=6)
batch_size = 32
print(summary(model, input_size=(batch_size, 501, 5)))