import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, confidence_bs, class_num, temperature_ins, temperature_clu, device):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.confidence_bs = confidence_bs
        self.class_num = class_num
        self.temperature_ins = temperature_ins
        self.temperature_clu = temperature_clu
        self.device = device

        self.mask_ins = self.mask_correlated(batch_size)
        self.mask_clu = self.mask_correlated(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated(self, size):
        N = 2 * size
        mask = torch.ones((N, N)).to(self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(size):
            mask[i, size + i] = 0
            mask[size + i, i] = 0
        mask = mask.bool()
        return mask

    def generate_pseudo_labels(self, c, class_num):
        pseudo_label = -torch.ones(self.confidence_bs, dtype=torch.long).to(self.device)
        tmp = torch.arange(0, self.confidence_bs).to(self.device)
        with torch.no_grad():
            prediction = c.argmax(dim=1)
            confidence = c.max(dim=1).values
            pseudo_per_class = math.ceil(self.confidence_bs / class_num * 0.5)
            for i in range(class_num):
                class_idx = (prediction == i)
                confidence_class = confidence[class_idx]
                num = min(confidence_class.shape[0], pseudo_per_class)
                confident_idx = torch.argsort(-confidence_class)
                for j in range(num):
                    idx = tmp[class_idx][confident_idx[j]]
                    pseudo_label[idx] = i
        return pseudo_label


    def forward_weighted_ce(self, c_, pseudo_label, class_num):
        idx, counts = torch.unique(pseudo_label, return_counts=True)
        freq = pseudo_label.shape[0] / counts.float()
        weight = torch.ones(class_num).to(pseudo_label.device)
        weight[idx] = freq
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss_ce = criterion(c_, pseudo_label)
        return loss_ce

    def forward(self, args, epoch, z_i, z_j, c_i, c_j, pseudo_label, h_hat1, S_weight1, h_hat2, S_weight2):
        co_i, co_j = c_i, c_j
        ci_max, _ = torch.max(c_i, dim=1)
        cj_max, _ = torch.max(c_j, dim=1)
        cop_i = c_i - ci_max[:, np.newaxis]
        cop_j = c_j - cj_max[:, np.newaxis]
        c_i, c_j = nn.functional.softmax(cop_i, dim=1), nn.functional.softmax(cop_j, dim=1)

        cop_i = torch.sum(torch.exp(cop_i), dim=1)
        cop_j = torch.sum(torch.exp(cop_j), dim=1)


        # pseudo-label cross entropy loss(L_p loss)
        if epoch <= args.stage1_epoch:
            cross_loss = 0
        elif args.stage1_epoch < epoch <= args.stage1_epoch + args.stage2_epoch:
            log_pred = co_j - (cj_max + torch.log(cop_j))[:, np.newaxis]
            cross_loss = F.nll_loss(log_pred, pseudo_label, reduction='none').mean()
        elif epoch > args.stage1_epoch + args.stage2_epoch:
            weight, _ = c_i.max(dim=-1)
            condition_weight = (weight >= args.threshold).float()
            log_pred = co_j - (cj_max + torch.log(cop_j))[:, np.newaxis]
            cross_loss = F.nll_loss(log_pred, pseudo_label, reduction='none')
            cross_loss = (cross_loss * condition_weight).mean()


        if epoch > args.stage1_epoch + args.stage2_epoch:
            # column entropy loss (L_e1 loss)
            p_i = c_i.sum(0).view(-1)
            p_i /= p_i.sum()
            ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
            p_j = c_j.sum(0).view(-1)
            p_j /= p_j.sum()
            ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
            ne_loss = ne_i + ne_j

            # entropy entropy loss (L_e2 loss)
            log_pred_c = co_i - (ci_max + torch.log(cop_i))[:, np.newaxis]
            h_loss = -(c_i * log_pred_c).sum() / c_i.size(0)

            # class-level contrastive Loss(L_c loss)
            c_i = c_i.t()
            c_j = c_j.t()
            N = 2 * self.class_num
            c = torch.cat((c_i, c_j), dim=0)
            sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature_clu
            sim_i_j = torch.diag(sim, self.class_num)
            sim_j_i = torch.diag(sim, -self.class_num)
            positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            negative_clusters = sim[self.mask_clu].reshape(N, -1)
            labels = torch.zeros(N).to(positive_clusters.device).long()
            logits = torch.cat((positive_clusters, negative_clusters), dim=1)
            cluster_loss = self.criterion(logits, labels)
            cluster_loss /= N
        else:
            cluster_loss = 0
            ne_loss = 0
            h_loss = 0




        # Instance-level contrastive loss(L_i loss)
        num_sample = z_i.size(0)
        pre_label = torch.argmax(co_i, dim=1).reshape(num_sample, 1)
        category_matrix = (pre_label == pre_label.T).float()
        category_matrix = category_matrix.repeat(2, 2)
        #################### item1 ##################
        z0 = torch.cat((z_i, z_j), dim=0)
        sim0 = torch.matmul(z0, z0.T) / self.temperature_ins
        e_sim0 = torch.exp(sim0)

        sim_i_j = torch.diag(e_sim0, num_sample)
        sim_j_i = torch.diag(e_sim0, -num_sample)
        numerator1 = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * num_sample, 1)
        negative_samples = e_sim0[self.mask_correlated(num_sample)].reshape(2 * num_sample, -1)
        denominator1 = torch.sum(negative_samples, dim=1)
        soft0 = numerator1 / denominator1

        #################### item2 ##################
        S1 = S_weight1.repeat(2, 2)
        if args.diagonal == 0:
            S1 = S1.fill_diagonal_(0)
        z1 = torch.cat((z_i, h_hat1), dim=0)
        sim1 = torch.matmul(z1, z1.T) / self.temperature_ins
        e_sim1 = torch.exp(sim1)

        weight_e_sim1 = e_sim1 * (S1+1e-5) * category_matrix
        numerator1 = torch.sum(weight_e_sim1, dim=1)
        negative_samples = e_sim1[self.mask_correlated(num_sample)].reshape(2 * num_sample, -1)
        denominator1 = torch.sum(negative_samples, dim=1)
        soft1 = numerator1 / denominator1
        log_soft1 = torch.log(soft1+soft0)
        instance_loss1 = - torch.mean(log_soft1)

        #################### item2 ##################
        S2 = S_weight2.repeat(2, 2)
        if args.diagonal == 0:
            S2 = S2.fill_diagonal_(0)
        z2 = torch.cat((z_j, h_hat2), dim=0)
        sim2 = torch.matmul(z2, z2.T) / self.temperature_ins
        e_sim2 = torch.exp(sim2)

        weight_e_sim2 = e_sim2 * (S2+1e-5) * category_matrix
        numerator2 = torch.sum(weight_e_sim2, dim=1)
        negative_samples = e_sim2[self.mask_correlated(num_sample)].reshape(2 * num_sample, -1)
        denominator2 = torch.sum(negative_samples, dim=1)
        soft2 = numerator2 / denominator2
        log_soft2 = torch.log(soft2 + soft0)
        instance_loss2 = - torch.mean(log_soft2)

        #################### all ##################
        instance_loss = (instance_loss1 + instance_loss2) / 2


        return cross_loss, instance_loss, cluster_loss, ne_loss, h_loss



    def forward_instance_elim(self, z_i, z_j, pseudo_labels):
        # instance loss
        invalid_index = (pseudo_labels == -1)
        mask = torch.eq(pseudo_labels.view(-1, 1),
                        pseudo_labels.view(1, -1)).to(z_i.device)
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(self.batch_size).float().to(z_i.device)
        mask &= (~(mask_eye.bool()).to(z_i.device))
        mask = mask.float()

        contrast_count = 2
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature_ins)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask_with_eye = mask | mask_eye.bool()
        # mask = torch.cat(mask)
        mask = mask.repeat(anchor_count, contrast_count)
        mask_eye = mask_eye.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(self.batch_size * anchor_count).view(-1, 1).to(
                z_i.device), 0)
        logits_mask *= (1 - mask)
        mask_eye = mask_eye * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_eye * log_prob).sum(1) / mask_eye.sum(1)

        # loss
        instance_loss = -mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count,
                                           self.batch_size).mean()

        return instance_loss
