import sys

import numpy as np
from numpy.core.numeric import zeros_like
from torch._C import unify_type_list

sys.path.append("..")
sys.path.append(".")

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp import loaders
from exp.tabnet import TabNet
from exp.settings import get_dataset
from exp.utils import *
from exp.models import FCNN, TabNet_ieeecis, TabNet_homecredit, Perc, FCNN_small

import argparse

default_model_dict = {
    "home_credit": TabNet_homecredit,
    "ieeecis": TabNet_ieeecis,
    "tabnet_ieeecis": TabNet_ieeecis,
    "twitter_bot": TabNet,
    "syn": TabNet,
}
model_dict = {
    "tabnet_homecredit": TabNet_homecredit,
    "tabnet_ieeecis": TabNet_ieeecis,
    "tabnet": TabNet,
    "perc": Perc,
    "fcnn": FCNN,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--data_dir", default="../data", type=str)
    parser.add_argument(
        "--dataset",
        default="ieeecis",
        choices=["ieeecis", "twitter_bot", "home_credit", "syn"],
        type=str,
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--lr_schedule", default="piecewise", choices=["cyclic", "piecewise"]
    )
    parser.add_argument(
        "--lr_max", default=0.1, type=float, help="0.05 in Table 1, 0.2 in Figure 2"
    )
    parser.add_argument("--attack", default="none", type=str, choices=["none", "pgd"])
    parser.add_argument("--eps", default=1.0, type=float)
    parser.add_argument(
        "--attack_iters", default=5, type=int, help="n_iter of pgd for evaluation"
    )
    parser.add_argument("--pgd_alpha_train", default=0.4, type=float)
    parser.add_argument(
        "--normreg", default=0.0, type=float
    )  # Can be used for numerical stability
    parser.add_argument("--grad-reg", default=0.00, type=float)
    parser.add_argument("--lamb", default=1.00, type=float)
    parser.add_argument("--distance", default="l1", type=str)
    parser.add_argument("--model", default="default", type=str)

    # TabNet specific parameters
    parser.add_argument("--n_d", default=16, type=int)
    parser.add_argument("--n_a", default=16, type=int)
    parser.add_argument("--n_shared", default=2, type=int)
    parser.add_argument("--n_ind", default=2, type=int)
    parser.add_argument("--n_steps", default=4, type=int)
    parser.add_argument("--relax", default=1.2, type=float)
    parser.add_argument("--vbs", default=512, type=int)


    parser.add_argument("--model_path", default="../models/default.pt", type=str)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument(
        "--n_train", default=-1, type=int, help="Number of training points."
    )
    parser.add_argument(
        "--weight_decay",
        default=5e-4,
        type=float,
        help="weight decay aka l2 regularization",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--eps-sched", action="store_true")
    parser.add_argument("--keep-one-hot", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--unit-ball", action="store_true")  # for unweighted l1
    parser.add_argument("--same-cost", action="store_true")
    parser.add_argument("--utility-max", action="store_true")
    parser.add_argument("--robust-baseline", action="store_true")
    parser.add_argument(
        "--utility-type",
        default="constant",
        choices=["constant", "additive", "multiplicative"],
        type=str,
    )
    parser.add_argument("--mixed-loss", default=False, action="store_true")
    return parser.parse_args()

def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

def set_bn_train(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()

def train(
    model,
    x,
    y,
    optimizer,
    criterion,
    eps=0.0,
    pgd_steps=0,
    pgd_alpha=0.5,
    costs=None,
    w_vec=None,
    mixed_loss=False,
    utility_max=False,
    _lambda=0,
    cat_map=None,
    utility_type="constant",
    eps_part=1,
    eps_max=0
):
    model.zero_grad()

    w_rep = w_vec.repeat(x.shape[0], 1)
    x[torch.where(w_rep == -1)] = 0
    ut_loss = 0
    if pgd_steps != 0:
        model.eval()
        if utility_max:
            #delta = max_util_delta(
            #    model, x, y, pgd_alpha, pgd_steps, dist="l1", costs=costs, w_vec=w_vec, cat_map=cat_map, eps_part=eps_part

            eps=20.4
            delta = attack_pgd_training(
                model,
                x,
                y,
                eps,
                pgd_alpha,
                pgd_steps,
                dist="l1",
                gains=costs,
                w_vec=w_vec,
                utility_type="constant",
                eps_part=eps_part,
                utility_max=utility_max,
                eps_max=eps_max
            )
        else:
            delta = attack_pgd_training(
                model,
                x,
                y,
                eps,
                pgd_alpha,
                pgd_steps,
                dist="l1",
                gains=costs,
                w_vec=w_vec,
                utility_type=utility_type,
                eps_part=eps_part,
                eps_max=eps_max
            )
        model.train()
        optimizer.zero_grad()
        # model.zero_grad()

        if mixed_loss:
            if utility_max:
                x_adv = torch.autograd.Variable(x.data + delta, requires_grad=False)
                #model.apply(set_bn_eval)
                adv_output = model(x_adv)
                #model.apply(set_bn_train)
                cl_output = model(x)
                adv_loss = util_loss(adv_output, delta, y, costs, w_vec)
                cl_loss = criterion(cl_output, y)
                loss = adv_loss * _lambda + cl_loss * (1 - _lambda)
                ut_loss = adv_loss.item()
                #print(delta[0], ut_loss)
            else:
                b_size = x.size()[0]
                x_adv = torch.autograd.Variable(x.data + delta, requires_grad=False)

                if cat_map is not None:
                    for cat in cat_map:
                        i, j = cat_map[cat]
                        max_ind = torch.argmax(x_adv[:, i:j + 1], 1)
                        #print(cat, max_ind[0], x_adv[:, i:j + 1])
                        x_adv[:, i:j + 1] = 0
                        x_adv[:, i:j + 1][range(x.shape[0]), max_ind] = 1

                #model.apply(set_bn_eval)
                adv_output = model(x_adv)
                #model.apply(set_bn_train)
                cl_output = model(x)
                
                adv_loss = criterion(adv_output, y)
                cl_loss = criterion(cl_output, y)

                if cl_loss > adv_loss and not utility_max:
                    loss = cl_loss
                else:
                    loss = adv_loss * _lambda + cl_loss * (1 - _lambda)
                ut_loss = adv_loss.item()
        else:
            x_adv = x + delta
            if cat_map is not None:
                for cat in cat_map:
                    i, j = cat_map[cat]
                    max_ind = torch.argmax(x_adv[:, i:j + 1], 1)
                    #print(cat, max_ind[0], x_adv[:, i:j + 1])
                    x_adv[:, i:j + 1] = 0
                    x_adv[:, i:j + 1][range(x.shape[0]), max_ind] = 1

            adv_output = model(x_adv)
            cl_output = model(x)
            loss = criterion(adv_output, y)

    else:
        delta = torch.zeros_like(x)
        output = model(x)
        cl_output = output
        adv_output = output
        loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return (
        loss.item(),
        torch.round(torch.sigmoid(adv_output)).detach(),
        ut_loss * delta.shape[0],
        torch.round(torch.sigmoid(cl_output)).detach(),
    )


def train_model(
    epochs,
    train_loader,
    test_loader,
    validation_loader,
    dataset,
    model_name,
    device=torch.device("cpu"),
    eps=0.0,
    pgd_steps=0,
    unit_ball=False,
    utility_max=False,
    mixed_loss=False,
    eps_sched=False,
    _lambda=0,
    utility_type="constant",
    model_path=None,
    n_d=16,
    n_a=16,
    n_shared=2,
    n_ind=2,
    n_steps=4,
    relax=1.2,
    vbs=512
):
    start_eps = eps
    criterion = nn.BCEWithLogitsLoss()
    #inp_dim = loaders.shape_dict[dataset]
    inp_dim = train_loader.dataset.X_train.shape[1]
    print(inp_dim)
    if pgd_steps != 0:
        pgd_alpha = 2 * (1 / pgd_steps)
    else:
        pgd_alpha = 0

    if model_name == "tabnet_ieeecis":
        net = model_dict[model_name](inp_dim=inp_dim, n_d=n_d, n_a=n_a, n_shared=n_shared, n_ind=n_ind, n_steps=n_steps, relax=relax, vbs=vbs).to(device)
    elif model_name != "default":
        net = model_dict[model_name](inp_dim=inp_dim).to(device)
    else:
        net = default_model_dict[dataset](inp_dim=inp_dim).to(device)

    optm = Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
    #optm = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-4)

    # Variables for validation step
    best_epoch = 0
    best_validation_acc = 0
    best_validation_loss = 0
    delay = 10 # TODO: FIGURE OUT IDEAL DELAY VALUE
    delay_starting_threshold = 15 # TODO: TO CHANGE IF WE FIND A BETTER VALUE
    best_model = None

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        clean_correct = 0
        ut_avg = 0
        eps_part = 1
        if eps_sched:
            if utility_max:
                eps_part = 1 * (epoch / epochs)
            else:
                eps_part = 1 * (2 * epoch / epochs)
            if eps_part > 1:
                eps_part = 1

        for bidx, (x_train, y_train, c) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            c = c.to(device)
            loss, predictions, ut_loss, clean_pred = train(
                net,
                x_train,
                y_train,
                optm,
                criterion,
                eps=eps,
                pgd_steps=pgd_steps,
                pgd_alpha=pgd_alpha,
                costs=c,
                w_vec=train_loader.dataset.w.to(device),
                utility_max=utility_max,
                mixed_loss=mixed_loss,
                _lambda=_lambda,
                cat_map=train_loader.dataset.cat_map,
                utility_type=utility_type,
                eps_part=eps_part,
                eps_max=train_loader.dataset.max_eps
            )
            for idx, i in enumerate(predictions):
                i = torch.round(i)
                if i == y_train[idx]:
                    correct += 1
            for idx, i in enumerate(clean_pred):
                i = torch.round(i)
                if i == y_train[idx]:
                    clean_correct += 1
            ut_avg += ut_loss
            acc = correct / len(train_loader.dataset)
            clean_acc = clean_correct / len(train_loader.dataset)
            epoch_loss += loss

        epoch_loss = epoch_loss / len(train_loader.dataset)
        test_acc, test_loss = dataset_eval(net, test_loader, criterion, score="acc", device=device)
        validation_acc, validation_loss = dataset_eval(net, validation_loader, criterion, score="acc", device=device)
        test_rob_acc, _ = dataset_eval_rob(
            net,
            test_loader,
            eps=eps,
            pgd_steps=pgd_steps,
            pgd_alpha=pgd_alpha,
            score="acc",
            w_vec=train_loader.dataset.w.to(device),
            device=device,
            cat_map=train_loader.dataset.cat_map,
            utility_type=utility_type,
            utility_max=utility_max,
            eps_max=train_loader.dataset.max_eps
        )

        # If we get a new best accuracy => save the model somewhere
        if validation_loss < best_validation_loss or (epoch+1) < delay_starting_threshold:
            best_epoch = epoch + 1
            best_validation_acc = validation_acc
            best_validation_loss = validation_loss
            best_model = net

        print(
            "Epoch {} Train Accuracy : {} (clean {}), Test Accuracy : {}, Test Robust Accuracy: {}, Validation Accuracy: {}".format(
                (epoch + 1), acc, clean_acc, test_acc, test_rob_acc, validation_acc
            )
        )
        print(
            "Epoch {} Train Loss : {}, Test Loss : {} Validation Loss : {}".format(
                (epoch + 1), epoch_loss, test_loss, validation_loss
            )
        )
        print(
            "Epoch {} Train Utility : {}".format(
                (epoch + 1), ut_avg / len(train_loader.dataset)
            )
        )
        print(
            "Delay: {} Current delay: {} Best Epoch: {} Best Validation Acc: {} Best Validation Loss {}".format(
                delay, ((epoch + 1) - best_epoch), best_epoch, best_validation_acc, best_validation_loss
            )
        )
        if((epoch + 1) - best_epoch >= delay):
            return best_model
            # torch.save(best_model.state_dict(), model_path)
    return best_model

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Cuda Device Available")
        print("Name of the Cuda Device: ", torch.cuda.get_device_name())
        print("GPU Computational Capablity: ", torch.cuda.get_device_capability())
    else:
        device = torch.device("cpu")

    folder_path = args.data_dir

    data_train = get_dataset(
        args.dataset, args.data_dir, mode="train", same_cost=args.same_cost, cat_map=args.keep_one_hot
    )
    data_test = get_dataset(
        args.dataset, args.data_dir, mode="test", same_cost=args.same_cost, cat_map=args.keep_one_hot
    )
    data_validation = get_dataset(
        args.dataset, args.data_dir, mode="validation", same_cost=args.same_cost, cat_map=args.keep_one_hot
    )
    train_loader = DataLoader(
        dataset=data_train, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        dataset=data_test, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    validation_loader = DataLoader(
        dataset=data_validation, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    model = train_model(
        args.epochs,
        train_loader,
        test_loader,
        validation_loader,
        args.dataset,
        args.model,
        device=device,
        eps=args.eps,
        pgd_steps=args.attack_iters,
        unit_ball=args.unit_ball,
        utility_max=args.utility_max,
        mixed_loss=args.mixed_loss,
        _lambda=args.lamb,
        eps_sched=args.eps_sched,
        utility_type=args.utility_type,
        model_path=args.model_path,
        n_d=args.n_d,
        n_a=args.n_a,
        n_shared=args.n_shared,
        n_ind=args.n_ind,
        n_steps=args.n_steps,
        relax=args.relax,
        vbs=args.vbs
    )
    torch.save(model.state_dict(), args.model_path)


if __name__ == "__main__":
    main()
