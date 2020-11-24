from training.semi_supervised.config import *
from model.semi_supervised.model import *
import torch
import matplotlib.pyplot as plt
import csv


def show_loss_graph(title: str, train_loss: list, eval_loss: list, show=True, export=False, filename='loss_graph.png'):
    """
    This function will visualize the loss value as graph
    :param title: Title of this figure
    :param train_loss: List of training loss value
    :param eval_loss: List of evaluation loss value
    :param show: Specify to show the plot or not
    :param export: Specify to export the loss graph as image file or not (.png file)
    :param filename: Specify filename to export
    :return: nothing
    """
    plt.title(title)
    plt.plot(train_loss, label="Training loss")
    plt.plot(eval_loss, label="Eval loss")
    plt.legend()
    if show:
        plt.show()
    if export:
        plt.savefig(LOG_PATH + filename)
    plt.clf()


def show_gan_loss_graph(title: str, dis_loss: list, gen_loss: list, export=False, filename='GANs_loss.png'):
    """
    This function will visualize the loss value of GANs
    :param title: Title of figure
    :param dis_loss: Discriminator loss array list
    :param gen_loss: Generator loss array list
    :param export: Specify to export graph as image or not
    :param filename: Filename to export
    :return:
    """
    plt.title(title)
    plt.plot(dis_loss, label="Discriminator loss")
    plt.plot(gen_loss, label="Generator loss")
    plt.legend()
    plt.show()
    if export:
        plt.savefig(LOG_PATH + filename)
    plt.clf()


def show_gen_res(x_gen: torch.Tensor, label_title: torch.Tensor, show=True, export=False):
    """
    This function will show the generated image
    :param x_gen: The generated image from the generator (Expected to be 4D tensor)
    :param label_title: Just want to compare the real label and generated image
    :param show: Specify to show the plot or not
    :param export: Specify to export the generation preview as image file or not (.png file)
    :return:
    """
    x_gen_show = x_gen[0:4]
    label_show = label_title[0:4]
    plt.title("Generated image")
    for each_x_gen_show, each_label_show, plt_idx in zip(x_gen_show, label_show, [221, 222, 223, 224]):
        img = each_x_gen_show.squeeze(0).cpu().numpy()
        sub_title = str(each_label_show.squeeze(0).cpu().numpy().argmax())
        plt.subplot(plt_idx)
        plt.gca().set_title(sub_title)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    if show:
        plt.show()
    if export:
        plt.savefig(LOG_PATH + 'generation_preview.png')
    plt.clf()


def dump_loss_log(loss_list: list, filename: str):
    """
    This function will dump all of loss value into a .csv file to investigate later
    :param loss_list: list of loss value to investigate later
    :param filename: filename
    :return:
    """
    with open(LOG_PATH + filename, 'w', newline='') as csv_log_file:
        wr = csv.writer(csv_log_file, quoting=csv.QUOTE_ALL)
        wr.writerow(loss_list)
