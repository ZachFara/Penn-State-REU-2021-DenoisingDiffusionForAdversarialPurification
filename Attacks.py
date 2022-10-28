####################### RayS attack ##############################
import numpy as np
import torch


class RayS(object):
    def __init__(self, model, epsilon=0.031, order=np.inf):
        self.model = model
        self.ord = order
        self.epsilon = epsilon
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.queries = None

    def get_xadv(self, x, v, d, lb=0., ub=1.):
        if isinstance(d, int):
            d = torch.tensor(d).repeat(len(x)).cuda()
        out = x + d.view(len(x), 1, 1, 1) * v
        out = torch.clamp(out, lb, ub)
        return out

    def attack_hard_label(self, x, y, target=None, query_limit=10000, seed=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        # init variables
        self.queries = torch.zeros_like(y).cuda()
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.d_t = torch.ones_like(y).float().fill_(float("Inf")).cuda()
        working_ind = (self.d_t > self.epsilon).nonzero().flatten()

        stop_queries = self.queries.clone()
        dist = self.d_t.clone()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
 
        block_level = 0
        block_ind = 0
        for i in range(query_limit):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            valid_mask = (self.queries < query_limit) 
            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, target, attempt, valid_mask)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm((self.x_final - x).view(shape[0], -1), self.ord, 1)
            stop_queries[working_ind] = self.queries[working_ind]
            working_ind = (dist > self.epsilon).nonzero().flatten()

            if torch.sum(self.queries >= query_limit) == shape[0]:
                print('out of queries')
                break

            progress_bar(torch.min(self.queries.float()), query_limit,
                         'd_t: %.4f | adbd: %.4f | queries: %.4f | rob acc: %.4f | iter: %d'
                         % (torch.mean(self.d_t), torch.mean(dist), torch.mean(self.queries.float()),
                            len(working_ind) / len(x), i + 1))
 

        stop_queries = torch.clamp(stop_queries, 0, query_limit)
        return self.x_final, stop_queries, dist, (dist <= self.epsilon)

    # check whether solution is found
    def search_succ(self, x, y, target, mask):
        self.queries[mask] += 1
        if target:
            return self.model.predict_label(x[mask]) == target[mask]
        else:
            return self.model.predict_label(x[mask]) != y[mask]

    # binary search for decision boundary along sgn direction
    def binary_search(self, x, y, target, sgn, valid_mask, tol=1e-3):
        sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

        d_start = torch.zeros_like(y).float().cuda()
        d_end = self.d_t.clone()

        initial_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target, valid_mask)
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            search_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target, to_search_ind)
            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]

    def __call__(self, data, label, target=None, query_limit=10000):
        return self.attack_hard_label(data, label, target=target, query_limit=query_limit)

import torch
import numpy as np
import torch.nn as nn


class GeneralTorchModel(nn.Module):
    def __init__(self, model, n_class=10, im_mean=None, im_std=None):
        super(GeneralTorchModel, self).__init__()
        self.model = model
        self.model.eval()
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class

    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self.model(image)
        return logits

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = torch.from_numpy(image).type(torch.FloatTensor)
        else:
            processed = image

        if self.im_mean is not None and self.im_std is not None:
            im_mean = torch.tensor(self.im_mean).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            im_std = torch.tensor(self.im_std).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            processed = (processed - im_mean) / im_std
        return processed

    def predict_prob(self, image):
        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            image = self.preprocess(image)
            logits = self.model(image)
            self.num_queries += image.size(0)
        return logits

    def predict_label(self, image):
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict

import os, torch
import sys
import time

term_width = 80
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



###############  BPDA Attack ####################
import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tnrange


class BPDAattack(object):
    def __init__(self, model=None, defense=None, device=None, epsilon=None, learning_rate=0.5,
                 max_iterations=100, clip_min=0, clip_max=1):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.defense = defense
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.device = device

    def generate(self, x, y):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """

        adv = x.detach().clone()

        lower = np.clip(x.detach().cpu().numpy() - self.epsilon, self.clip_min, self.clip_max)
        upper = np.clip(x.detach().cpu().numpy() + self.epsilon, self.clip_min, self.clip_max)

        for i in range(self.MAX_ITERATIONS):
            adv_purified = self.defense(adv)
            adv_purified.requires_grad_()
            adv_purified.retain_grad()

            scores = self.model(adv_purified)
            loss = self.loss_fn(scores, y)
            loss.backward()

            grad_sign = adv_purified.grad.data.sign()

            # early stop, only for batch_size = 1
            # p = torch.argmax(F.softmax(scores), 1)
            # if y != p:
            #     break

            adv += self.LEARNING_RATE * grad_sign

            adv_img = np.clip(adv.detach().cpu().numpy(), lower, upper)
            adv = torch.Tensor(adv_img).to(self.device)
        return adv
    
    

######BPDA EOT ATTACK#############


import torch
import torch.nn.functional as F

criterion = torch.nn.CrossEntropyLoss()

#Try 100 attack_reps

class BPDA_EOT_Attack():
    def __init__(self, model, adv_eps=8.0/255, eot_defense_reps=150, eot_attack_reps=15):
        self.model = model

        self.config = {
            'eot_defense_ave': 'logits',
            'eot_attack_ave': 'logits',
            'eot_defense_reps': eot_defense_reps,
            'eot_attack_reps': eot_attack_reps,
            'adv_steps': 50,
            'adv_norm': 'l_inf',
            'adv_eps': adv_eps,
            'adv_eta': 2.0 / 255,
            'log_freq': 10
        }

        print(f'BPDA_EOT config: {self.config}')

    def purify(self, x):
        return self.model(x, mode='purify')

    def eot_defense_prediction(seslf, logits, reps=1, eot_defense_ave=None):
        if eot_defense_ave == 'logits':
            logits_pred = logits.view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
        elif eot_defense_ave == 'softmax':
            logits_pred = F.softmax(logits, dim=1).view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
        elif eot_defense_ave == 'logsoftmax':
            logits_pred = F.log_softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
        elif reps == 1:
            logits_pred = logits
        else:
            raise RuntimeError('Invalid ave_method_pred (use "logits" or "softmax" or "logsoftmax")')
        _, y_pred = torch.max(logits_pred, 1)
        return y_pred

    def eot_attack_loss(self, logits, y, reps=1, eot_attack_ave='loss'):
        if eot_attack_ave == 'logits':
            logits_loss = logits.view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
            y_loss = y
        elif eot_attack_ave == 'softmax':
            logits_loss = torch.log(F.softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0))
            y_loss = y
        elif eot_attack_ave == 'logsoftmax':
            logits_loss = F.log_softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
            y_loss = y
        elif eot_attack_ave == 'loss':
            logits_loss = logits
            y_loss = y.repeat(reps)
        else:
            raise RuntimeError('Invalid ave_method_eot ("logits", "softmax", "logsoftmax", "loss")')
        loss = criterion(logits_loss, y_loss)
        return loss

    def predict(self, X, y, requires_grad=True, reps=1, eot_defense_ave=None, eot_attack_ave='loss'):
        if requires_grad:
            logits = self.model(X, mode='classify')
        else:
            with torch.no_grad():
                logits = self.model(X.data, mode='classify')

        y_pred = self.eot_defense_prediction(logits.detach(), reps, eot_defense_ave)
        correct = torch.eq(y_pred, y)
        loss = self.eot_attack_loss(logits, y, reps, eot_attack_ave)

        return correct.detach(), loss

    def pgd_update(self, X_adv, grad, X, adv_norm, adv_eps, adv_eta, eps=1e-10):
        if adv_norm == 'l_inf':
            X_adv.data += adv_eta * torch.sign(grad)
            X_adv = torch.clamp(torch.min(X + adv_eps, torch.max(X - adv_eps, X_adv)), min=0, max=1)
        elif adv_norm == 'l_2':
            X_adv.data += adv_eta * grad / grad.view(X.shape[0], -1).norm(p=2, dim=1).view(X.shape[0], 1, 1, 1)
            dists = (X_adv - X).view(X.shape[0], -1).norm(dim=1, p=2).view(X.shape[0], 1, 1, 1)
            X_adv = torch.clamp(X + torch.min(dists, adv_eps*torch.ones_like(dists))*(X_adv-X)/(dists+eps), min=0, max=1)
        else:
            raise RuntimeError('Invalid adv_norm ("l_inf" or "l_2"')
        return X_adv

    def purify_and_predict(self, X, y, purify_reps=1, requires_grad=True):
        X_repeat = X.repeat([purify_reps, 1, 1, 1])
        X_repeat_purified = self.purify(X_repeat).detach().clone()
        X_repeat_purified.requires_grad_()
        correct, loss = self.predict(X_repeat_purified, y, requires_grad, purify_reps,
                                     self.config['eot_defense_ave'], self.config['eot_attack_ave'])
        if requires_grad:
            X_grads = torch.autograd.grad(loss, [X_repeat_purified])[0]
            # average gradients over parallel samples for EOT attack
            attack_grad = X_grads.view([purify_reps]+list(X.shape)).mean(dim=0)
            return correct, attack_grad
        else:
            return correct, None

    def eot_defense_verification(self, X_adv, y, correct, defended):
        for verify_ind in range(correct.nelement()):
            if correct[verify_ind] == 0 and defended[verify_ind] == 1:
                defended[verify_ind] = self.purify_and_predict(X_adv[verify_ind].unsqueeze(0), y[verify_ind].view([1]),
                                                               self.config['eot_defense_reps'], requires_grad=False)[0]
        return defended

    def eval_and_bpda_eot_grad(self, X_adv, y, defended, requires_grad=True):
        correct, attack_grad = self.purify_and_predict(X_adv, y, self.config['eot_attack_reps'], requires_grad)
        if self.config['eot_defense_reps'] > 0:
            defended = self.eot_defense_verification(X_adv, y, correct, defended)
        else:
            defended *= correct
        return defended, attack_grad

    def attack_batch(self, X, y):
        # get baseline accuracy for natural images
        defended = self.eval_and_bpda_eot_grad(X, y, torch.ones_like(y).bool(), False)[0]
        print('Baseline: {} of {}'.format(defended.sum(), len(defended)))

        class_batch = torch.zeros([self.config['adv_steps'] + 2, X.shape[0]]).bool()
        class_batch[0] = defended.cpu()
        ims_adv_batch = torch.zeros(X.shape)
        for ind in range(defended.nelement()):
            if defended[ind] == 0:
                ims_adv_batch[ind] = X[ind].cpu()

        X_adv = X.clone()

        # adversarial attacks on a single batch of images
        for step in range(self.config['adv_steps'] + 1):
            defended, attack_grad = self.eval_and_bpda_eot_grad(X_adv, y, defended)

            class_batch[step+1] = defended.cpu()
            for ind in range(defended.nelement()):
                if class_batch[step, ind] == 1 and defended[ind] == 0:
                    ims_adv_batch[ind] = X_adv[ind].cpu()

            # update adversarial images (except on final iteration so final adv images match final eval)
            if step < self.config['adv_steps']:
                X_adv = self.pgd_update(X_adv, attack_grad, X, self.config['adv_norm'], self.config['adv_eps'], self.config['adv_eta'])
                X_adv = X_adv.detach().clone()

            if step == 1 or step % self.config['log_freq'] == 0 or step == self.config['adv_steps']:
                print('Attack {} of {}   Batch defended: {} of {}'.
                      format(step, self.config['adv_steps'], int(torch.sum(defended).cpu().numpy()), X_adv.shape[0]))

            if int(torch.sum(defended).cpu().numpy()) == 0:
                print('Attack successfully to the batch!')
                break

        for ind in range(defended.nelement()):
            if defended[ind] == 1:
                ims_adv_batch[ind] = X_adv[ind].cpu()

        return class_batch, ims_adv_batch

    def attack_all(self, X, y, batch_size):
        class_path = torch.zeros([self.config['adv_steps'] + 2, 0]).bool()
        ims_adv = torch.zeros(0)

        n_batches = X.shape[0] // batch_size
        if n_batches == 0 and X.shape[0] > 0:
            n_batches = 1
        for counter in range(n_batches):
            X_batch = X[counter * batch_size:min((counter + 1) * batch_size, X.shape[0])].clone().to(X.device)
            y_batch = y[counter * batch_size:min((counter + 1) * batch_size, X.shape[0])].clone().to(X.device)

            class_batch, ims_adv_batch = self.attack_batch(X_batch.contiguous(), y_batch.contiguous())
            class_path = torch.cat((class_path, class_batch), dim=1)
            ims_adv = torch.cat((ims_adv, ims_adv_batch), dim=0)
            print(f'finished {counter}-th batch in attack_all')

        return class_path, ims_adv
    
    
#############NaiveFool#######

from tqdm.notebook import tnrange
class NaiveFool:
    def __init__(self, classifier,eps):
        self.classifier = classifier
        self.eps = eps
    def attack(self,x,y,num_iter = 100):
        x_orig = torch.clone(x)
        x_adv = torch.clone(x)
        idx_broke = set()
        for i in tnrange(num_iter):
            if len(idx_broke) == x_adv.shape[0]:
                break
            
            noise = (torch.rand_like(x_orig)*2-1) *self.eps
            
            #Add noise from this step
            x_noised = x + noise
            x_noised = torch.clamp(x_noised,0,1)
            #Check for robustness
            y_pred = torch.argmax(self.classifier(x_noised),dim = 1)
            adv_idx = torch.where(y_pred != y)[0].long()
            for idx in adv_idx:
                if idx.item() not in idx_broke:
                    x_adv[idx] = x_noised[idx]
                    idx_broke.add(idx.item())
            print(str((len(idx_broke)/x_adv.shape[0])*100)+ "% successfully perturbed", end = "\r")
        return x_adv, idx_broke