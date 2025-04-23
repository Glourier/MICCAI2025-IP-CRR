import argparse
import random
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from arch.mimic_split_5 import FullyConnectedClassifier, FullyConnectedQuerier
import dataset
import ops
import utils
import wandb


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argsument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--class_name', type=str, default='Lung Opacity', help='Class name.')
    parser.add_argument('--debug', action='store_true', help='Whether use small dataset to debug')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weighted', action='store_true', help='Whether use weighted cross entropy loss')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load checkpoint')
    parser.add_argument('--threshold', type=float, default=None, help='Threshold for binary answers')
    # Following args are usually default
    parser.add_argument('--sampling', type=str, default='biased')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--data_dir', type=str, default='./data/MIMIC', help='data directory')
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    # Following args are not controlled by the input
    parser.add_argument('--n_queries', type=int, default=520)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--max_queries', type=int, default=200)
    parser.add_argument('--max_queries_test', type=int, default=50)
    parser.add_argument('--model', type=str, default='NetworkMIMIC')
    parser.add_argument('--name', type=str, default='mimic')
    parser.add_argument('--data_name', type=str, default='_mimic_v27_hash1_nli_whole.pkl')

    args = parser.parse_args()
    return args


def adaptive_sampling(x, max_queries, model):
    model.requires_grad_(False)
    device = x.device
    N, D = x.shape

    rand_history_length = torch.randint(low=0, high=max_queries, size=(N,)).to(device)
    mask = torch.zeros((N, D), requires_grad=False).to(device)
    for _ in range(max_queries + 1):  # +1 because we start from empty history
        masked_input = torch.stack([x * mask, mask]).transpose(0, 1)  # Stacked
        with torch.no_grad():
            query = model(masked_input, mask)

        # index only the rows smaller than rand_history_length
        idx = mask.sum(1) <= rand_history_length
        mask[idx] = mask[idx] + query[idx]
    model.requires_grad_(True)  # work around for unused parameter error
    return mask


def get_weight(file_name, class_name=None):
    import json
    with open(file_name, 'r') as f:
        data = json.load(f)
    n_total = data['n_total']

    if class_name is not None:
        n_pos = np.array([data['n_positive'][class_name]])
    else:
        n_pos = np.array(list(data['n_positive'].values()))

    # wBCE loss
    n_neg = n_total - n_pos
    weight = n_neg / n_pos

    return torch.tensor(weight)


def main(args):
    ## Setup
    # wandb
    PROJECT_NAME = "VIP-MIMIC"
    args.layers = 5
    SAVE_CODE_LIST = ['utils.py', 'ops.py', 'dataset.py', './arch/mimic_split_5.py', 'main_mimic.py',
                      'train_mimic.job']
    TIME = utils.get_time()

    args.n_queries = 520
    args.max_queries = 200
    args.max_queries_test = 200

    args.data_name = '_mimic.pkl'
    args.name = f'mimic'
    args.weight_file = 'train_weights.json'

    args.slurm_job_id = os.getenv('SLURM_JOB_ID', None)
    if args.slurm_job_id is not None:
        args.name = f'{args.slurm_job_id}_' + args.name

    if args.debug == True:
        args.max_queries = 50
        args.max_queries_test = 50
        args.data_name = args.data_name.replace('.pkl', '_try.pkl')
        args.name = args.name + '_try'
        print(f"Using debugging dataset: {args.data_name}")

    run = wandb.init(project=PROJECT_NAME, name=TIME + '-' + args.name, mode=args.mode)
    model_dir = os.path.join(args.save_dir, f'{TIME}-{PROJECT_NAME}-{args.name}-{run.id}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
    utils.save_params(model_dir, vars(args))
    utils.save_code(model_dir, SAVE_CODE_LIST)
    wandb.config.update(args)
    print(f"Save to {model_dir}")

    # cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)

    # random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## constants
    N_CLASSES = args.n_classes
    N_QUERIES = args.n_queries
    Cosine_T_Max = args.epochs
    ITERATIONS_TO_DECREASE_TAU_OVER = args.epochs

    ## Data
    trainset, valset, _ = dataset.load_mimic_nli(args.data_dir, args.data_name, n_concept=N_QUERIES,
                                                 class_names=[args.class_name])

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=4)
    print(f"Loaded {len(trainset)} training samples and {len(valset)} validation samples.")

    classifier = FullyConnectedClassifier(query_size=N_QUERIES, output_size=N_CLASSES)
    classifier = nn.DataParallel(classifier).cuda()
    querier = FullyConnectedQuerier(query_size=N_QUERIES, output_size=N_QUERIES, tau=args.tau_start)
    querier = nn.DataParallel(querier).cuda()

    ## Optimization
    if args.weighted:
        weight = get_weight(os.path.join(args.data_dir, args.weight_file), class_name=args.class_name).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(list(classifier.parameters()) + list(querier.parameters()), amsgrad=True, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Cosine_T_Max)
    tau_vals = np.linspace(args.tau_start, args.tau_end, ITERATIONS_TO_DECREASE_TAU_OVER)
    sigmoid = nn.Sigmoid()

    ## Load checkpoint
    if args.ckpt_path is not None:
        ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
        classifier.load_state_dict(ckpt_dict['classifier'])
        classifier.to(device)
        querier.load_state_dict(ckpt_dict['querier'])
        querier.to(device)
        print(f"Checkpoint Loaded at epoch {ckpt_dict['epoch']} from {args.ckpt_path}!")

    ## Train
    batch_idx = 0
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"Training epoch {epoch}:")

        # training
        classifier.train()
        querier.train()
        tau = tau_vals[epoch]

        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_train_f1 = 0
        epoch_train_auc = 0
        epoch_train_ap = 0

        for train_labels, train_features in tqdm(trainloader, desc=f'Training epoch {epoch}'):

            train_labels = train_labels.float().squeeze().to(device)  # Size [B]
            train_features = train_features.to(device)
            train_bs = train_labels.shape[0]

            querier.module.update_tau(tau)
            optimizer.zero_grad()

            # initial random sampling
            if args.sampling == 'biased':
                mask = adaptive_sampling(train_features, args.max_queries, querier).to(device).float()
            elif args.sampling == 'random':
                mask = ops.random_sampling(args.max_queries, N_QUERIES, train_bs).to(device).float()

            history = torch.stack([train_features * mask, mask]).transpose(0, 1)
            query = querier(history, mask)
            history = history + torch.stack([query * train_features, query]).transpose(0, 1)

            # prediction
            train_logits = classifier(history).squeeze()  # Size [B]
            print(f"train_logits: {train_logits}")
            print(f"train_labels: {train_labels}")

            # backprop
            loss = criterion(train_logits, train_labels)
            loss.backward()
            optimizer.step()

            train_prob = sigmoid(train_logits).detach().cpu()
            train_labels = train_labels.cpu()
            train_pred = (train_prob > 0.5).float()
            train_acc = (train_pred == train_labels).float().sum()
            epoch_train_acc += train_acc
            epoch_train_loss += loss.item()
            train_f1 = f1_score(train_labels, train_pred,
                                average='binary', zero_division=1)
            epoch_train_f1 += train_f1
            train_ap = average_precision_score(train_labels, train_prob)
            epoch_train_ap += train_ap

            if len(set(train_labels.cpu().numpy())) > 1:
                train_auc = roc_auc_score(train_labels, train_prob)
            else:
                train_auc = 1  # Set default value as 1 if not calculable
            epoch_train_auc += train_auc

            # logging
            wandb.log({
                'train/batch': batch_idx,
                'train/loss': loss.item(),
                'train/acc': train_acc / train_bs,
                'train/f1': train_f1,
                'train/auc': train_auc,
                'train/ap': train_ap,
                'lr': utils.get_lr(optimizer),
                'train/gradnorm_cls': utils.get_grad_norm(classifier),
                'train/gradnorm_qry': utils.get_grad_norm(querier)
            })
            batch_idx += 1
        scheduler.step()

        epoch_train_loss /= len(trainloader)
        epoch_train_acc /= len(trainset)
        epoch_train_f1 /= len(trainloader)
        epoch_train_auc /= len(trainloader)
        epoch_train_ap /= len(trainloader)

        wandb.log({
            'train/epoch': epoch,
            'train/epoch_loss': epoch_train_loss,
            'train/epoch_acc': epoch_train_acc,
            'train/epoch_f1': epoch_train_f1,
            'train/epoch_auc': epoch_train_auc,
            'train/epoch_ap': epoch_train_ap
        })

        print(
            f"Finished training epoch {epoch}, loss: {epoch_train_loss}, accuracy: {epoch_train_acc}, f1: {epoch_train_f1}, auc-roc: {epoch_train_auc}, ap: {epoch_train_ap}, lr: {utils.get_lr(optimizer)}, gradnorm_cls: {utils.get_grad_norm(classifier)}, gradnorm_qry: {utils.get_grad_norm(querier)}. ")

        # saving
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'classifier': classifier.state_dict(),
                'querier': querier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            },
                os.path.join(model_dir, 'ckpt', f'epoch{epoch}.ckpt'))

        # evaluation
        if epoch % 10 == 0 or epoch == args.epochs - 1:

            print(f"Validating at epoch {epoch}: ")

            classifier.eval()
            querier.eval()
            epoch_val_qry_need = []
            epoch_val_acc_max = 0
            epoch_val_acc_ip = 0
            val_label_all = []
            val_prob_ip_all = []
            val_prob_max_all = []

            for val_labels, val_features in tqdm(valloader, desc=f'Validation epoch {epoch}'):
                val_labels = val_labels.float().to(device)  # Size [B]
                val_features = val_features.to(device)
                val_bs = val_labels.shape[0]

                # Compute logits for all queries
                mask = torch.zeros(val_bs, N_QUERIES).to(device)
                logits, queries = [], []
                for i in range(args.max_queries_test):
                    with torch.no_grad():
                        history = torch.stack([val_features * mask, mask]).transpose(0, 1)
                        query = querier(history, mask)
                        mask = mask + query
                        history = history + torch.stack([val_features * mask, query]).transpose(0, 1)

                        # predict with updated history
                        label_logits = classifier(history).squeeze()

                    logits.append(label_logits)
                    queries.append(query)
                logits = torch.stack(logits).permute(1, 0)  # Shape [B, |Q|]

                # accuracy using all queries
                val_prob_max = sigmoid(logits[:, -1])  # Size [B]
                val_pred_max = (val_prob_max > 0.5).float()
                val_acc_max = (val_pred_max == val_labels).float().sum()
                epoch_val_acc_max += val_acc_max

                # compute query needed
                qry_need = ops.compute_queries_needed_bi(logits, threshold=THRESHOLD)
                epoch_val_qry_need.append(qry_need)

                # accuracy using IP
                val_prob_ip = sigmoid(logits[torch.arange(
                    len(qry_need)), qry_need - 1])  # Why torch.arange(len(qry_need)), rather than :? The first dimension is B?
                val_pred_ip = (val_prob_ip > 0.5).float()
                val_acc_ip = (val_pred_ip == val_labels.squeeze()).float().sum()
                epoch_val_acc_ip += val_acc_ip

                val_label_all.append(val_labels)
                val_prob_ip_all.append(val_prob_ip)
                val_prob_max_all.append(val_prob_max)

            # Calculate F1 Score at the end of the epoch
            val_label_all = torch.cat(val_label_all).cpu().detach().numpy()
            val_prob_ip_all = torch.cat(val_prob_ip_all).cpu().detach().numpy()
            val_prob_max_all = torch.cat(val_prob_max_all).cpu().detach().numpy()
            val_f1_max = f1_score(val_label_all, val_prob_max_all > 0.5, average='binary')
            val_f1_ip = f1_score(val_label_all, val_prob_ip_all > 0.5, average='binary')
            val_auc_max = roc_auc_score(val_label_all, val_prob_max_all)
            val_auc_ip = roc_auc_score(val_label_all, val_prob_ip_all)
            val_ap_max = average_precision_score(val_label_all, val_prob_max_all)
            val_ap_ip = average_precision_score(val_label_all, val_prob_ip_all)

            epoch_val_acc_max = epoch_val_acc_max / len(valset)
            epoch_val_acc_ip = epoch_val_acc_ip / len(valset)

            # mean and std of queries needed
            epoch_val_qry_need = torch.hstack(epoch_val_qry_need).float()
            qry_need_avg = epoch_val_qry_need.mean()
            qry_need_std = epoch_val_qry_need.std()

            # logging
            wandb.log({
                'val/epoch': epoch,
                'val/acc_max': epoch_val_acc_max,
                'val/acc_ip': epoch_val_acc_ip,
                'val/f1_max': val_f1_max,
                'val/f1_ip': val_f1_ip,
                'val/auc_max': val_auc_max,
                'val/auc_ip': val_auc_ip,
                'val/ap_max': val_ap_max,
                'val/ap_ip': val_ap_ip,
                'val/qry_need_avg': qry_need_avg,
                'val/qry_need_std': qry_need_std
            })

            print(
                f"Finished validating at epoch {epoch}, val_acc_max: {epoch_val_acc_max}, val_acc_ip: {epoch_val_acc_ip}, val_f1_max: {val_f1_max}, val_f1_ip: {val_f1_ip}, val_auc_max: {val_auc_max}, val_auc_ip: {val_auc_ip}, val_ap_max: {val_ap_max}, val_ap_ip: {val_ap_ip}, qry_need_avg: {qry_need_avg}, qry_need_std: {qry_need_std}. ")

            if epoch_val_acc_ip > best_val_acc:
                torch.save({
                    'classifier': classifier.state_dict(),
                    'querier': querier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                },
                    os.path.join(model_dir, 'ckpt', f'best.ckpt'))
                best_val_acc = epoch_val_acc_ip


if __name__ == '__main__':
    args = parseargs()
    main(args)
