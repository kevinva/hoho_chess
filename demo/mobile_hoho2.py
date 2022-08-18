import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import torch
import coremltools as ct
from Web.model import hoho_agent, hoho_utils

model = hoho_agent.AgentNet()
model.eval()

example_input = torch.rand((1, hoho_utils.IN_PLANES_NUM, hoho_utils.BOARD_HEIGHT, hoho_utils.BOARD_WIDTH))
traced_model = torch.jit.trace(model, example_input)
out = traced_model(example_input)
print(f'traced out: {out}')

model_out = ct.convert(traced_model, inputs=[ct.TensorType(shape=example_input.shape)])
model_out.save('hoho_agent.mlmodel')

