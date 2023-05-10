import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

if __name__ == '__main__':
    # for reproduce
    torch.manual_seed(0)
    
    if world.simple_model[:3] != 'exp':
        Recmodel = register.MODELS[world.model_name](world.config, dataset)
        Recmodel = Recmodel.to(world.device)
        bpr = utils.BPRLoss(Recmodel, world.config)
    else:
        # if we are doing exp1, exp2 - disable grad
        torch.set_grad_enabled(False)

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD and world.simple_model[:3] != "exp":
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1

    # init tensorboard
    if world.tensorboard and world.simple_model[:3] != "exp":
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                        )
    elif world.tensorboard and world.simple_model[:3] == 'exp':
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, world.simple_model + "-" + world.dataset)
        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")

    try:
        if world.simple_model == 'exp1':
            epoch = 0
            cprint("[TEST]")
            start_time = time.time()
            Procedure.Test_exp1(dataset, epoch, w, world.config['multicore'])
            end_time = time.time()
            print(f"total time consumption: {end_time-start_time}s")
        elif world.simple_model == 'exp2':
            epoch = 0
            cprint("[TEST]")
            start_time = time.time()
            Procedure.Test_exp2(dataset, epoch, w, world.config['multicore'])
            end_time = time.time()
            print(f"total time consumption: {end_time-start_time}s")
        elif world.simple_model == 'exp3':
            epoch = 0
            cprint("[TEST]")
            start_time = time.time()
            # create tensorboard inside of procedure funtion
            Procedure.Test_exp3(dataset, epoch, w, world.config['multicore'])
            end_time = time.time()
            print(f"total time consumption: {end_time-start_time}s")
        elif(world.simple_model != 'none'):
            epoch = 0
            cprint("[TEST]")
            start_time = time.time()
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            end_time = time.time()
            print(f"total time consumption: {end_time-start_time}s")
        else:
            for epoch in range(world.TRAIN_epochs):
                if epoch %10 == 0:
                    cprint("[TEST]")
                    Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
                print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
                torch.save(Recmodel.state_dict(), weight_file)
    finally:
        if world.tensorboard:
            w.close()