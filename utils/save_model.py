import os
import torch


def save_model(data_name, model, optimizer, optimizer_head, current_epoch):
    out = ("save/" + data_name + "/checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'optimizer_head': optimizer_head.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)