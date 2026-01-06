%%writefile evaluate.py

import torch
import torch.linalg as la
import numpy as np
from torchvision.transforms import Normalize
from model import getModel
from dataset import getDataLoader
from utils import *
from torchattacks import FGSM, PGD, MIFGSM, AutoAttack
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_toy(config):
    seed_everything(config.seed)
    
    # cria e paraleliza igual ao treino
    model = getModel(config).cuda()
    model = torch.nn.DataParallel(model)
    
    txtlog = TxtLogger(config)
    
    xshape = (config.lip_batch_size, config.in_channels)
    x = (torch.rand(xshape) + 0.3 * torch.randn(xshape)).cuda()
    model(x)
    
    # carrega o checkpoint
    model_state = torch.load(f"{config.train_dir}/model.ckpt")
    model.load_state_dict(model_state)  # agora as chaves vão bater

    # Avaliação com ataques adversariais
    def run_attack(attack_name, attack):
        n, acc = 0.0, 0.0
        stats = np.zeros((1, 2))

        x_test = (torch.rand(xshape) + 0.3 * torch.randn(xshape)).cuda()
        y_test = model(x_test).argmax(1)

        x_adv = attack(x_test, y_test)
        y_pred = model(x_adv).detach()
        correct = (y_pred.max(1)[1] == y_test)

        stats[0, 0] = y_test.size(0)
        stats[0, 1] = correct.sum().item()

        n += stats[0, 0]
        acc += stats[0, 1]

        final_acc = 100 * acc / n
        txtlog(f"Final accuracy under {attack_name}: {final_acc:.2f}%")
        np.savetxt(f"{config.train_dir}/{attack_name}.csv", stats)

    fgsm = FGSM(model, eps=8 / 255.)
    run_attack("fgsm_eps8_255", fgsm)

    fgsm2 = FGSM(model, eps=0.1)
    run_attack("fgsm_eps0.1", fgsm2)

    pgd = PGD(model, eps=8 / 255., alpha=2 / 255., steps=10)
    run_attack("pgd_eps8_255", pgd)

    pgd2 = PGD(model, eps=0.1, alpha=0.02, steps=10)
    run_attack("pgd_eps0.1", pgd2)

class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model 
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.model(self.normalize(x))

def evaluate(config):
    seed_everything(config.seed)
    model = getModel(config).cuda()

    _, testLoader = getDataLoader(config)
    txtlog = TxtLogger(config)

    xshape = (1, config.in_channels, config.img_size, config.img_size)
    x = (torch.rand(xshape) + 0.3 * torch.randn(xshape)).cuda()
    model(x)

    model_state = torch.load(f"/kaggle/input/lbdn-model/pytorch/default/2/kwl_sand_large_bstl.ckpt")
    # Carregue diretamente sem remover 'module.'
    model.load_state_dict(model_state)

    model(x)

    if config.normalized:
        mean = {
            'cifar10': [0.4914, 0.4822, 0.4465],
            'cifar100': [0.5071, 0.4865, 0.4409],
            'lisa': [0.485, 0.456, 0.406],
            'bstl': [0.485, 0.456, 0.406],
            'tiny_imagenet': [0.485, 0.456, 0.406]
        }[config.dataset]
        std = {
            'cifar10': [0.2470, 0.2435, 0.2616],
            'cifar100': [0.2675, 0.2565, 0.2761],
            'lisa': [0.229, 0.224, 0.225],
            'bstl': [0.229, 0.224, 0.225],
            'tiny_imagenet': [0.229, 0.224, 0.225]
        }[config.dataset]
    else:
        mean = [1.0, 1.0, 1.0]
        std = [1.0, 1.0, 1.0]

    mu = torch.Tensor(mean)[:, None, None].cuda()
    sg = torch.Tensor(std)[:, None, None].cuda()

    aa_model = NormalizedModel(model, mean, std).cuda()
    aa_model.eval()

    # Avaliar em dados limpos
    y_true = []
    y_pred = []
    
    for batch_idx, batch in enumerate(testLoader):
        x, y = batch[0].cuda(), batch[1].cuda()
    
        # desfaz normalização
        xg = x.clone()
        xg.mul_(sg).add_(mu)
    
        # inferência limpa
        logits = aa_model(xg).detach()
        preds = logits.argmax(dim=1)
    
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
    
    # concatena batches
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    # métricas
    clean_acc       = accuracy_score(y_true, y_pred) * 100
    clean_precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    clean_recall    = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    clean_f1        = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    
    txtlog(
        f"[CLEAN] "
        f"Acc: {clean_acc:.2f}% | "
        f"Prec: {clean_precision:.2f}% | "
        f"Rec: {clean_recall:.2f}% | "
        f"F1: {clean_f1:.2f}%"
    )

    def run_attack(attack_name, attack):
        y_true = []
        y_pred = []
    
        for batch_idx, batch in enumerate(testLoader):
            x, y = batch[0].cuda(), batch[1].cuda()
    
            # desfaz normalização
            xg = x.clone()
            xg.mul_(sg).add_(mu)
    
            # gera amostras adversariais
            xd = attack(xg, y)
    
            # inferência
            logits = aa_model(xd).detach()
            preds = logits.argmax(dim=1)
    
            # acumula rótulos
            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
    
        # concatena todos os batches
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
    
        acc       = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
        return {
            "attack": attack_name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    all_eps = [0.01, 8/255, 0.04, 0.055, 0.07, 0.085, 0.1, 0.115, 0.13, 0.15, 0.175, 0.2]
   # all_eps = [0.085, 0.1, 0.115, 0.13, 0.15, 0.175, 0.2]
    attack_fns = {
        "FGSM": lambda eps: FGSM(aa_model, eps=eps),
        "PGD":  lambda eps: PGD(aa_model, eps=eps),
        "MIM":  lambda eps: MIFGSM(model, eps=eps),
        "AutoAttack":  lambda eps: AutoAttack(aa_model, eps=eps, n_classes=4)

    }

    for attack_name, attack_fn in attack_fns.items():
        print("")
        for eps in all_eps:
            attack = attack_fn(eps)
    
            metrics = run_attack(f"{attack_name} ε={eps:.4f}", attack)
            
            print(
                f"[{metrics['attack']}] "
                f"Accuracy: {metrics['accuracy']*100:.2f}% | "
                f"Precision: {metrics['precision']*100:.2f}% | "
                f"Recall: {metrics['recall']*100:.2f}% | "
                f"F1-score: {metrics['f1_score']*100:.2f}%"
            )
