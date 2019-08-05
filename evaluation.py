import numpy as np
import torch
from sklearn import metrics

class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def rmse_ite(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    def abs_ate(self, ypred1, ypred0):
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def pehe(self, ypred1, ypred0):
        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)
        return ite, ate, pehe

    # from CIB research
    def auc(self, ypred1, ypred0):
        y_label = np.concatenate((self.mu0, self.mu1), axis=0)
        y_label_pred = np.concatenate((ypred0, ypred1), axis=0)
        fpr, tpr, thresholds = metrics.roc_curve(y_label, y_label_pred)
        auc = metrics.auc(fpr, tpr)
        roc_auc = metrics.roc_auc_score(y_label, y_label_pred)
        # yf = self.y
        # fact_fpr, fact_tpr, fact_thresholds = metrics.roc_curve(yf, yf_p)
        # fact_auc = metrics.auc(fpr, tpr)
        # fact_roc_auc = roc_auc_score(yf, yf_p)
        return {'auc': auc,'roc_auc':roc_auc}#, 'fact_auc':fact_auc,'fact_roc_auc':fact_roc_auc }

    def policy(self, pred_y_1, pred_y_0):
        e = self.e

        yf = self.y[e==1]
        t = self.t

        pred_policy = (pred_y_1 > pred_y_0).astype(float)
        pred_policy_t = np.mean(pred_policy)
        pred_policy_c = 1 - pred_policy_t
        if np.sum((pred_policy==1) * (t_true==1)) == 0:
            pred_y_t = 0.
        else:
            pred_y_t = np.mean(y_true[(pred_policy==1) * (t_true==1)])

        if np.sum((pred_policy==0) * (t_true==0)) == 0:
            pred_y_c = 0.
        else:
            pred_y_c = np.mean(y_true[(pred_policy==0) * (t_true==0)])

        risk = 1 - (pred_y_t*pred_policy_t + pred_y_c*pred_policy_c)

        return risk


def get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, x_train, t_train, L=1):
    y_infer = q_y_xt_dist(x_train.float(), t_train.float())
    # use inferred y
    xy = torch.cat((x_train.float(), y_infer.mean), 1)  # TODO take mean?
    z_infer = q_z_tyx_dist(xy=xy, t=t_train.float())
    # Manually input zeros and ones
    y0 = p_y_zt_dist(z_infer.mean, torch.zeros(z_infer.mean.shape).cuda()).mean  # TODO take mean?
    y1 = p_y_zt_dist(z_infer.mean, torch.ones(z_infer.mean.shape).cuda()).mean  # TODO take mean?

    return y0.cpu().detach().numpy(), y1.cpu().detach().numpy()
