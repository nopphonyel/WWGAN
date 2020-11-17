from training.semi_supervised.config import *  # Import configuration
import torch
from dataset.eeg_dataset import EEG_DATASET
from model.semi_supervised.loss_func import *
from torch.utils.data import DataLoader
from training.semi_supervised.visualize_helper import *
from tqdm import trange
import numpy as np

torch.autograd.set_detect_anomaly(True)

# Some constant defination
TRAIN = "t"
EVAL = "e"

# Initialization **All configuration store in config.py file
# Dataset initialization
dataset = EEG_DATASET(data_path='dataset/p_nice', img_path='dataset/png', device=DEV, pre_sel=1)
dataloader = DataLoader(dataset=dataset, batch_size=BS, shuffle=True)

# Model initialization
input_sample = next(iter(dataloader))[0]

sx = SemanticImageExtractor(output_class_num=10,
                            feature_size=feature_size).to(DEV)
# Argument expected_shape : send some sample data to let model determine its structure
sy = SemanticEEGExtractor(expected_shape=input_sample,
                          output_class_num=10,
                          feature_size=feature_size).to(DEV)
d1 = D1().to(DEV)
d2 = D2().to(DEV)
G = Generator(sem_size=feature_size).to(DEV)

# Optimizer initialization
sx_op = torch.optim.Adam(sx.parameters(), lr=mu1)
sy_op = torch.optim.Adam(sy.parameters(), lr=mu1)

d1_op = torch.optim.Adam(d1.parameters(), lr=mu2)
d2_op = torch.optim.Adam(d2.parameters(), lr=mu2)
G_op = torch.optim.Adam(G.parameters(), lr=mu2)

train_gan_loss = []
train_extr_loss = []
eval_gan_loss = []
eval_extr_loss = []
min_gan_loss = None
min_extr_loss = None
for i in range(EPCH + 1):
    print("Epoch", i, "/", EPCH)
    # Initialize before entering the training session
    epch_train_gan_loss = []  # GANs loss
    epch_train_extr_loss = []  # Feature extractor loss
    epch_eval_gan_loss = []
    epch_eval_extr_loss = []
    preview_img = None

    for session in [TRAIN, EVAL]:
        # Initialize stuff in each session
        if session is TRAIN:
            dataset.training_mode()
        else:
            dataset.eval_mode()

        # Start training the model
        for y_p, l_real_p, x_p in dataloader:
            # SEMANTIC NETWORK TRAINING SECTION
            # print("UPDATING SEMANTIC EXTRACTOR")

            if session is TRAIN:
                sx_op.zero_grad()
                sy_op.zero_grad()

            x_u = x_p
            # For this dataset, I think its still make sense to do this
            l_real_p = l_real_p.unsqueeze(1)
            l_real_u = l_real_p

            fx_p, lx_p = sx(x_p)
            fy_p, ly_p = sy(y_p)

            fx_u = fx_p  # For this dataset, I think its still make sense to do this
            lx_u = lx_p

            j1 = j1_loss(l=l_real_p, fx=fx_p, fy=fy_p)
            j2 = j2_loss(l_real=l_real_p, lx=lx_p)
            j3 = j3_loss(l_real=l_real_p, ly=ly_p)
            j4 = j4_loss(fy_p=fy_p, l_p=l_real_p, fx_u=fx_u, l_u=l_real_u)
            j5 = j5_loss(l_real_u=l_real_u, lx_u=lx_u)
            j_loss = j1 + (alp1 * j2) + (alp2 * j3) + j4 + (alp3 * j5)

            if session is TRAIN:
                j_loss.backward()
                sx_op.step()
                sy_op.step()

            # DISCRIMINATOR TRAINING SECTION
            if session is TRAIN:
                # print("UPDATING DISCRIM")
                d1_op.zero_grad()
                d2_op.zero_grad()

            # Reshape the tensor corresponding to the generator and detach everything
            fy_p = fy_p.squeeze(1).detach()
            ly_p = ly_p.squeeze(1).detach()
            fx_u = fx_u.squeeze(1).detach()
            lx_u = lx_u.squeeze(1).detach()

            curr_BS = y_p.shape[0]

            x_p_gen = G.forward(z=torch.rand((curr_BS, feature_size)).to(DEV), semantic=fy_p, label=ly_p)
            x_p_gen_dtch = x_p_gen.detach()
            x_u_gen = G.forward(z=torch.rand((curr_BS, feature_size)).to(DEV), semantic=fx_u, label=lx_u)
            x_u_gen_dtch = x_u_gen.detach()

            if session is EVAL:
                preview_img = x_u_gen_dtch

            l1 = l1_loss(d1, x_p, x_p_gen_dtch)
            l2 = l2_loss(d2, x_p, x_p_gen_dtch, fy_p, ly_p)
            l3 = l3_loss(d1, x_u, x_u_gen_dtch)
            l4 = l4_loss(d2, x_u, x_u_gen_dtch, fx_u, lx_u)

            dl_loss = -((ld1 * l1) + l2 + (ld2 * l3) + l4)

            if session is TRAIN:
                dl_loss.backward()
                d1_op.step()
                d2_op.step()

            # GENERATOR TRAINING SECTION
            if session is TRAIN:
                # print("UPDATING GENERATOR")
                G_op.zero_grad()

            l1 = l1_loss(d1, x_p, x_p_gen, True)
            l2 = l2_loss(d2, x_p, x_p_gen, fy_p, ly_p, True)
            l3 = l3_loss(d1, x_u, x_u_gen, True)
            l4 = l4_loss(d2, x_u, x_u_gen, fx_u, lx_u, True)

            # Need to recalculate loss to re-backpropagation
            gl_loss = -10000 * ((ld1 * l1) + l2 + (ld2 * l3) + l4)

            if session is TRAIN:
                gl_loss.backward()
                # print("L LOSS", l1.item(), l2.item(), l3.item(), l4.item())
                G_op.step()
                # print(j_loss.item(), dl_loss.item(), gl_loss.item())

            # Append loss value
            if session is TRAIN:
                epch_train_extr_loss.append(j_loss.item())
                epch_train_gan_loss.append(gl_loss.item())
            else:
                epch_eval_extr_loss.append(j_loss.item())
                epch_eval_gan_loss.append(gl_loss.item())

    # Now after finish all dataset in each epoch, average all loss
    train_gan_loss.append(np.average(epch_train_gan_loss))
    train_extr_loss.append(np.average(epch_train_extr_loss))
    eval_gan_loss.append(np.average(epch_eval_gan_loss))
    eval_extr_loss.append(np.average(epch_eval_extr_loss))

    # Find new min loss value and export the model
    if min_extr_loss is None:
        min_extr_loss = eval_extr_loss[0]
    else:
        if eval_extr_loss[-1] < min_extr_loss:
            min_extr_loss = eval_extr_loss[-1]
            torch.save(sx.state_dict(), EXPORT_PATH + "sx.pth")
            torch.save(sy.state_dict(), EXPORT_PATH + "sy.pth")

    if min_gan_loss is None:
        min_gan_loss = eval_gan_loss[0]
    else:
        if eval_gan_loss[-1] < min_gan_loss:
            min_gan_loss = eval_gan_loss[-1]
            torch.save(d1.state_dict(), EXPORT_PATH + "d1.pth")
            torch.save(d2.state_dict(), EXPORT_PATH + "d2.pth")
            torch.save(G.state_dict(), EXPORT_PATH + "g.pth")

    if i % 5 == 0:
        print("\t > Logging...")
        # Dumping loss file
        dump_loss_log(train_gan_loss, "training_generator_loss.csv")
        dump_loss_log(train_extr_loss, "training_extractor_loss.csv")
        dump_loss_log(eval_gan_loss, "eval_generator_loss.csv")
        dump_loss_log(eval_extr_loss, "eval_extractor_loss.csv")

        show_loss_graph("Feature extractor loss", train_extr_loss, eval_extr_loss, True, 'extr_loss.png')
        show_loss_graph("Generator loss", train_gan_loss, eval_gan_loss, True, 'gen_loss.png')
        show_gen_res(preview_img, True)
