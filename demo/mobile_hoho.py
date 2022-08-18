import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import torch
from onnx_coreml import convert
from Web.model import hoho_agent, hoho_utils

# 模型pth导出到onnx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = hoho_agent.AgentNet()
model.load_state_dict(torch.load('./hoho_agent_1656576279_1.pth', map_location=device))
model.eval()

dummy_input = torch.rand((1, hoho_utils.IN_PLANES_NUM, hoho_utils.BOARD_HEIGHT, hoho_utils.BOARD_WIDTH))
input_names = ['hoho_in']
output_names = ['hoho_out']

torch.onnx.export(model, dummy_input, 'hoho_agent.onnx', verbose=True, input_names=input_names, output_names=output_names)


# onnx转iOS 的mlmodel
model = convert(model='hoho_agent.onnx', minimum_ios_deployment_target='13')
model.save('hoho_agent.mlmodel')

print(f'{torch.__version__}')