from torch.utils.tensorboard import SummaryWriter
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from models import ConflictNet
from velocity_utils import *
from dataset import Conflict_DataSet
import shutil
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
from loss_utils import AUC_loss


def validation(dataloader_val, epoch, model, thresholds, writer_val, logger, device, loss_func, global_step, window, debug=False):
    loss_, roc_, pr_, focal_, entropy_ = [], [], [], [], []
    metrics = {
        'TPR': {},
        'TNR': {},
        'PR': {},
        'Acc': {},
        'F_1': {}
    }
    for key in metrics.keys():
        for value in thresholds:
            metrics[key][str(value)] = []
    BS, BSS, ROC_AUC, PR_AUC = [], [], [], []



    with torch.no_grad():
        tic = time.time()
        model.eval()

        for i, (feature, adj, label, feat_long, index, adj25, spatial_idx, label_day, death_day) in enumerate(dataloader_val):
            '''
            features: [B, T, N, C]
            adj: [T, 2, 9*N]
            labels: [B, T, N]
            index: [B]
            '''
            feature = feature.float().to(device)
            adj = adj.long().to(device)
            adj25 = adj25.long().to(device)
            label = label.float().to(device)
            label_day = label_day.float().to(device)
            feat_long = feat_long.float().to(device)
            spatial_idx = spatial_idx.to(device)

            B, _, N = feature.shape[:3]
            sharpness = 1 / (global_step + 1)
            sharpness = max(1e-3, sharpness)

            probs, phi, phi_next, probs_day, mix_probs, q0, feat_df = model(feature, adj, feat_long, adj25=adj25, spatial_idx=spatial_idx, sharpness=sharpness) # [B, N, 6]
            probs = probs.flatten(-2, -1)
            label = label.flatten(-2, -1)
            probs_loss = probs
            label_loss = label
            loss_pr, loss_roc, loss_focal, loss_entropy = loss_func(probs_loss, label_loss, phi, phi_next, label_day, probs_day, q0)
            loss = loss_pr + loss_roc + loss_focal + loss_entropy

            pred_p, pred_n = [], []
            for value in thresholds:
                pred_p.append((probs>value))
                pred_n.append((probs<=value))
            gt_p = (label>0.5)
            gt_n = (label<=0.5)

            loss_.append(loss.item())
            pr_.append(loss_pr.item())
            roc_.append(loss_roc.item())
            focal_.append(loss_focal.item())
            entropy_.append(loss_entropy.item())
            for k, value in enumerate(thresholds):
                tpr = torch.sum(pred_p[k]*gt_p, axis=-1) / (torch.sum(gt_p, axis=-1)+1e-3)
                tnr = torch.sum(pred_n[k]*gt_n, axis=-1) / (torch.sum(gt_n, axis=-1)+1e-3)
                pr = torch.sum(pred_p[k]*gt_p, axis=-1) / (torch.sum(pred_p[k], axis=-1)+1e-3)
                acc = torch.sum(pred_p[k]*gt_p + pred_n[k]*gt_n, axis=-1) / N / window
                f1 = 2 * pr*tpr / (pr+tpr+1e-3)
                metrics['TPR'][str(value)].append(torch.mean(tpr).item())
                metrics['TNR'][str(value)].append(torch.mean(tnr).item())
                metrics['PR'][str(value)].append(torch.mean(pr).item())
                metrics['Acc'][str(value)].append(torch.mean(acc).item())
                metrics['F_1'][str(value)].append(torch.mean(f1).item())
            bs = torch.mean(
                (probs-label)**2,
                axis=-1
                )
            bs_ref = torch.mean(
                (torch.mean(label, keepdim=True, axis=-1) - label) ** 2,
                axis=-1
            )
            BS.append(torch.mean(bs).item())
            BSS.append(torch.mean(1 - bs/(bs_ref+1e-5)).item())
            roc_auc = []
            pr_auc = []
            for b in range(B):
                roc_auc.append(
                    roc_auc_score(label[b].detach().cpu().numpy(), probs[b].detach().cpu().numpy())
                )
                precision_logit, recall_logit, _ = precision_recall_curve(label[b].detach().cpu().numpy(), probs[b].detach().cpu().numpy())
                pr_auc.append(auc(recall_logit, precision_logit))
            ROC_AUC.append(mean(roc_auc))
            PR_AUC.append(mean(pr_auc))
        
        toc = time.time()
        repo_str = 'Validating [{}, {}]\t loss={:.5f}\t pr={:.5f}\t roc={:.5f}\t focal={:.5f}\t entropy={:.5f}\t \n BS={:.5f}\t BSS={:.5f}\t ROC_AUC={:.5f}\t PR_AUC={:.5f}\t time={:.2f}\n'.format(global_step, epoch, mean(loss_), mean(pr_), mean(roc_), mean(focal_), mean(entropy_), mean(BS), mean(BSS), mean(ROC_AUC), mean(PR_AUC), toc-tic)
        for value in thresholds:
            repo_str += f'[{value}]:\t '
            for key in metrics.keys():
                repo_value = mean(metrics[key][str(value)])
                repo_str += f'{key}='
                if key != 'F_1':
                    repo_str += '{:.2f}%\t'.format(repo_value*100)
                    writer_val.add_scalar(f'metrics_{value}/{key}', repo_value*100, global_step)
                else:
                    repo_str += '{:.5f}\t'.format(repo_value)
                    writer_val.add_scalar(f'metrics_{value}/{key}', repo_value, global_step)
            repo_str += '\n'
        logger.info(repo_str)

        writer_val.add_scalar('loss/loss', mean(loss_), global_step)
        writer_val.add_scalar('loss/loss_pr', mean(pr_), global_step)
        writer_val.add_scalar('loss/loss_roc', mean(roc_), global_step)
        writer_val.add_scalar('loss/loss_focal', mean(focal_), global_step)
        writer_val.add_scalar('loss/loss_entropy', mean(entropy_), global_step)
        writer_val.add_scalar('metric/BS', mean(BS), global_step)
        writer_val.add_scalar('metric/BSS', mean(BSS), global_step)
        writer_val.add_scalar('metric/ROC_AUC', mean(ROC_AUC), global_step)
        writer_val.add_scalar('metric/PR_AUC', mean(PR_AUC), global_step)

def train(data_path, exp_path):
    # fix the seed
    setup_seed(3407)

    # parameters
    save_freq = 500
    report_freq = 100
    max_epoch = 1000
    batch_size = 4
    num_workers = 4
    lr = 3e-4
    pretrained_ckpt_path = None
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    long_term = 12
    short_term = 2
    step = 30
    window = 1
    diffuse_step = 5

    # log
    log_filename = "train_log.log"
    log_path = os.path.join(exp_path, log_filename)
    logger = get_logger(log_path)

    # ckpts
    ckpt_path = os.path.join(exp_path, 'ckpts/')
    os.makedirs(ckpt_path, exist_ok=True)

    # tensorboard
    tensorboard_path = os.path.join(exp_path, 'tensorboard/')
    os.makedirs(tensorboard_path, exist_ok=True)
    writer_train = SummaryWriter(tensorboard_path + '/t')
    writer_val = SummaryWriter(tensorboard_path + '/v')

    # code
    code_list = os.listdir('.')
    code_path = os.path.join(exp_path, 'code')
    os.makedirs(code_path, exist_ok=True)
    for file in code_list:
        if file.endswith('.py'):
            shutil.copy(file, code_path)

    logger.info('Using single GPU...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = 0
    world_size = 1

    # dataset
    logger.info('Loading dynamic Data...')
    dataset_train = Conflict_DataSet(data_path, long_term=long_term, step=step, window=window, diffuse_step=diffuse_step, short_term=short_term, val=False)
    dataset_val = Conflict_DataSet(data_path, long_term=long_term, step=step, window=window, diffuse_step=diffuse_step, short_term=short_term, val=True)
    logger.info(f'Training: {len(dataset_train)}!')
    logger.info(f'Val: {len(dataset_val)}!')
    dataloader_train = DataLoader(dataset_train,
                                batch_size=batch_size,
                                num_workers=int(num_workers),
                                drop_last=False,
                                shuffle=True)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                num_workers=1,
                                drop_last=False,
                                shuffle=False)
    logger.info('Data Loaded!')

    # model
    logger.info('Loading Model...')
    model = ConflictNet(diffuse_step=diffuse_step, pred_window=window, device=device)
    if pretrained_ckpt_path is not None:
        state_dict = torch.load(pretrained_ckpt_path, map_location='cpu')
        msg = model.load_state_dict(state_dict, strict=True)
    model.to(device)
    logger.info('Model Loaded!')
    total_params = sum(p.numel() for p in model.parameters())
    total_mb = total_params * 4 / (1024 * 1024)
    logger.info(f"Total parameters: {total_mb:.2f} MB")

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, 
        betas=(0.9, 0.999))

    scheduler = CosineLRScheduler(optimizer,
                t_initial=max_epoch+1,
                lr_min=1e-6,
                warmup_lr_init=1e-6,
                warmup_t=0,
                cycle_limit=1,
                t_in_epochs=True)

    logger.info('Start training')
    model.zero_grad()

    loss_func = AUC_loss(num_thresholds=100, device=device)
    loss_, roc_, pr_, focal_, entropy_ = [], [], [], [], []
    metrics = {
        'TPR': {},
        'TNR': {},
        'PR': {},
        'Acc': {},
        'F_1': {}
    }
    for key in metrics.keys():
        for value in thresholds:
            metrics[key][str(value)] = []
    BS, BSS, ROC_AUC, PR_AUC = [], [], [], []

    global_step = 0

    if pretrained_ckpt_path is not None:
        validation(dataloader_val, 0, model, thresholds, writer_val, logger, device, loss_func, 0, window=window, debug=True)
    for epoch in range(max_epoch):
        tic = time.time()
        model.train()

        for i, (feature, adj, label, feat_long, index, adj25, spatial_idx, label_day, death_day) in enumerate(dataloader_train):

            feature = feature.float().to(device)
            adj = adj.long().to(device)
            adj25 = adj25.long().to(device)
            label = label.float().to(device)
            label_day = label_day.float().to(device)
            index = index.to(device)
            feat_long = feat_long.float().to(device)
            spatial_idx = spatial_idx.to(device)

            B, _, N = feature.shape[:3]
            sharpness = 1 / (global_step + 1)
            sharpness = max(1e-3, sharpness)

            probs, phi, phi_next, probs_day, mix_probs, q0, feat_df = model(feature, adj, feat_long, adj25=adj25, spatial_idx=spatial_idx, sharpness=sharpness) # [B, N, 6]
            probs = probs.flatten(-2, -1)
            label = label.flatten(-2, -1)
            probs_loss = probs
            label_loss = label
            loss_pr, loss_roc, loss_focal, loss_entropy = loss_func(probs_loss, label_loss, phi, phi_next, label_day, probs_day, q0)
            loss = loss_pr + loss_roc + loss_focal + loss_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_p, pred_n = [], []
            p_str = f'label:{torch.sum((label>0.5)).item()}, '
            for value in thresholds:
                pred_p.append((probs>value))
                pred_n.append((probs<=value))
                p_str += f'{torch.sum(pred_p[-1]).item()}, '
            if i == len(dataloader_train)-1:
                print(p_str)
            gt_p = (label>0.5)
            gt_n = (label<=0.5)

            loss_.append(loss.item())
            pr_.append(loss_pr.item())
            roc_.append(loss_roc.item())
            focal_.append(loss_focal.item())
            entropy_.append(loss_entropy.item())
            for k, value in enumerate(thresholds):
                tpr = torch.sum(pred_p[k]*gt_p, axis=-1) / (torch.sum(gt_p, axis=-1)+1e-3)
                tnr = torch.sum(pred_n[k]*gt_n, axis=-1) / (torch.sum(gt_n, axis=-1)+1e-3)
                pr = torch.sum(pred_p[k]*gt_p, axis=-1) / (torch.sum(pred_p[k], axis=-1)+1e-3)
                acc = torch.sum(pred_p[k]*gt_p + pred_n[k]*gt_n, axis=-1) / N / window
                f1 = 2 * pr*tpr / (pr+tpr+1e-3)
                metrics['TPR'][str(value)].append(torch.mean(tpr).item())
                metrics['TNR'][str(value)].append(torch.mean(tnr).item())
                metrics['PR'][str(value)].append(torch.mean(pr).item())
                metrics['Acc'][str(value)].append(torch.mean(acc).item())
                metrics['F_1'][str(value)].append(torch.mean(f1).item())
            bs = torch.mean(
                (probs-label)**2,
                axis=-1
                )
            bs_ref = torch.mean(
                (torch.mean(label, keepdim=True, axis=-1) - label) ** 2,
                axis=-1
            )
            BS.append(torch.mean(bs).item())
            BSS.append(torch.mean(1 - bs/(bs_ref+1e-5)).item())
            roc_auc = []
            pr_auc = []
            for b in range(B):
                roc_auc.append(
                    roc_auc_score(label[b].detach().cpu().numpy(), probs[b].detach().cpu().numpy())
                )
                precision_logit, recall_logit, _ = precision_recall_curve(label[b].detach().cpu().numpy(), probs[b].detach().cpu().numpy())
                pr_auc.append(auc(recall_logit, precision_logit))
            ROC_AUC.append(mean(roc_auc))
            PR_AUC.append(mean(pr_auc))

            global_step += batch_size
        
            if global_step % report_freq == 0 and rank == 0:
                toc = time.time()
                lr_now = optimizer.state_dict()['param_groups'][0]['lr']
                repo_str = '[{}, {}/{}]\t lr={:.7f}\t loss={:.5f}\t pr={:.5f}\t roc={:.5f}\t focal={:.5f}\t entropy={:.5f}\t \n BS={:.5f}\t BSS={:.5f}\t ROC_AUC={:.5f}\t PR_AUC={:.5f}\t time={:.2f}\n'.format(global_step, epoch, max_epoch, lr_now, mean(loss_), mean(pr_), mean(roc_), mean(focal_), mean(entropy_), mean(BS), mean(BSS), mean(ROC_AUC), mean(PR_AUC), toc-tic)
                for value in thresholds:
                    repo_str += f'[{value}]:\t '
                    for key in metrics.keys():
                        repo_value = mean(metrics[key][str(value)])
                        repo_str += f'{key}='
                        if key != 'F_1':
                            repo_str += '{:.2f}%\t'.format(repo_value*100)
                            writer_train.add_scalar(f'metrics_{value}/{key}', repo_value*100, global_step)
                        else:
                            repo_str += '{:.5f}\t'.format(repo_value)
                            writer_train.add_scalar(f'metrics_{value}/{key}', repo_value, global_step)
                    repo_str += '\n'
                logger.info(repo_str)
                
                writer_train.add_scalar('lr', lr_now, global_step)
                writer_train.add_scalar('loss/loss', mean(loss_), global_step)
                writer_train.add_scalar('loss/loss_pr', mean(pr_), global_step)
                writer_train.add_scalar('loss/loss_roc', mean(roc_), global_step)
                writer_train.add_scalar('loss/loss_focal', mean(focal_), global_step)
                writer_train.add_scalar('loss/loss_entropy', mean(entropy_), global_step)
                writer_train.add_scalar('metric/BS', mean(BS), global_step)
                writer_train.add_scalar('metric/BSS', mean(BSS), global_step)
                writer_train.add_scalar('metric/ROC_AUC', mean(ROC_AUC), global_step)
                writer_train.add_scalar('metric/PR_AUC', mean(PR_AUC), global_step)

                loss_, roc_, pr_, focal_, entropy_ = [], [], [], [], []
                BS, BSS, ROC_AUC, PR_AUC = [], [], [], []
                for key in metrics.keys():
                    for value in thresholds:
                        metrics[key][str(value)] = []
                tic = time.time()

            if global_step % save_freq == 0 and rank == 0:
                logger.info('Validating...')
                # validation(dataloader_val, epoch, model, thresholds, writer_val, logger, device, loss_func, global_step, window)
                saving_path = os.path.join(ckpt_path, 'last.pt')
                torch.save(model.state_dict(), saving_path)
                logger.info('save ckpts!')
                model.train()

            if global_step % report_freq == 0:
                scheduler.step(global_step // report_freq)

    writer_train.close()
    logger.info('Finsh')


if __name__ == '__main__':
    data_path = './dataset'
    exp_path = './exp'

    os.makedirs(exp_path, exist_ok=True)
    train(data_path, exp_path)
