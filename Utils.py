import torch
from Params import params

class Utlis:
    def __init__(self):
        pass

    def load_model(self, agent):
        agent.model.load_state_dict(torch.load(params.save_path, map_location=params.device))
        print("Đã load model thành công!")


utils = Utlis()