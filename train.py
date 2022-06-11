import torch.optim as optim
import glob
import json
import argparse
import os
import random
import math
from time import gmtime, strftime
from models import *
from dataset_GBU import FeatDataLayer, DATA_LOADER
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import torch.backends.cudnn as cudnn
import classifier


def train(opt):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    print("Random Seed: ", opt.manualSeed)
    np.random.seed(opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True
    print('Running parameters:')
    print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
    opt.gpu = torch.device("cuda:"+opt.gpu if torch.cuda.is_available() else "cpu")

    dataset = DATA_LOADER(opt)
    opt.A_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class
    out_dir = opt.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    if opt.zsl:
        log_dir = out_dir +'/log_ZSL_{}_finetune_{}_{}.txt'.format(opt.dataset, opt.finetune, opt.file_name)
    else:
        log_dir = out_dir +'/log_GZSL_{}_finetune_{}_{}.txt'.format(opt.dataset, opt.finetune, opt.file_name)
    with open(log_dir, 'w') as f:
        f.write('Option:' + '\n')
        f.write(str(opt) + '\n\n')
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class

    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)

    opt.niter = int(dataset.ntrain/opt.batchsize) * opt.gen_nepoch

    result_gzsl_soft = Result()
    result_zsl_soft = Result()

    vaeNet = VAE(opt).to(opt.gpu)
    relationNet = RelationNet(opt).to(opt.gpu)
    cntnt_enc_dec = AutoEncoder(opt).to(opt.gpu)
    attr_encoder = Encoder(opt).to(opt.gpu)

    best_performance = [0.0, 0.0, 0.0] # [unseen, seen, H_mean]

    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('Generative Model Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0
    # optimizer
    optimizer = optim.Adam(vaeNet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    relation_optimizer = optim.Adam(relationNet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    attribute_encoder_optimizer = optim.Adam(
        list(attr_encoder.parameters()) + list(cntnt_enc_dec.parameters()),
        lr=opt.lr, weight_decay=opt.weight_decay
    )

    mse = nn.MSELoss().to(opt.gpu)

    iters = math.ceil(dataset.ntrain/opt.batchsize)
    beta = 0.01
    best_precision_25 = 0.0
    for it in range(start_step, opt.niter+1):
        vaeNet.train()
        relationNet.train()
        cntnt_enc_dec.train()
        attr_encoder.train()

        if it % iters == 0:
            beta = min(opt.kl_warmup*(it/iters), 1)

        batch_sample = data_layer.forward()
        feat_data = batch_sample['data']
        labels_numpy = batch_sample['labels'].astype(int)
        labels = torch.from_numpy(labels_numpy.astype('int')).to(opt.gpu)

        A = np.array([dataset.train_att[i,:] for i in labels])
        A = torch.from_numpy(A.astype('float32')).to(opt.gpu)
        X = torch.from_numpy(feat_data).to(opt.gpu)
        seen_A = np.array([dataset.train_att])
        seen_A = torch.from_numpy(seen_A).to(opt.gpu)
        seen_A = seen_A.squeeze()
        seen_A_n = seen_A.shape[0]
        
        x_mean, z_mu, z_var, z = vaeNet(X, A)

        vae_loss, kl = multinomial_loss_function(x_mean, X, z_mu, z_var, z, beta=beta)

        labels = labels.cpu()
        one_hot_labels = torch.zeros(opt.batchsize, seen_A_n).scatter_(1, labels.view(-1, 1), 1).to(opt.gpu)

        # real part
        attrs_real = attr_encoder(X)
        real_X_hat = cntnt_enc_dec(X, attrs_real)
        real_recon_loss = mse(X, real_X_hat)

        relations = relationNet(attrs_real, seen_A)
        relations = relations.view(-1, seen_A_n)
        real_relation_loss = mse(relations, one_hot_labels)

        # fake part
        attrs_fake = attr_encoder(x_mean)
        fake_X_hat = cntnt_enc_dec(x_mean, attrs_fake)
        fake_recon_loss = mse(x_mean, fake_X_hat)

        relations = relationNet(attrs_fake, seen_A)
        relations = relations.view(-1, seen_A_n)
        fake_relation_loss = mse(relations, one_hot_labels)

        # Add all loss
        recon_loss = real_recon_loss + fake_recon_loss
        relation_loss = real_relation_loss + fake_relation_loss

        loss = opt.vae_weight * vae_loss + opt.relation_weight * relation_loss + opt.recon_weight * recon_loss

        optimizer.zero_grad()
        relation_optimizer.zero_grad()
        attribute_encoder_optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        relation_optimizer.step()
        attribute_encoder_optimizer.step()

        if it % opt.disp_interval == 0 and it:
            log_text = 'Iter-[{}/{}]; loss: {:.3f}; kl:{:.3f}; vae_loss:{:.3f}; relation_loss:{:.3f}; recon_loss:{:.3f}'.format(it,
                                             opt.niter, loss.item(), kl.item(), vae_loss.item(), relation_loss.item(), recon_loss.item())
            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it > opt.evl_start:            
            vaeNet.eval()
            attr_encoder.eval()

            gen_feat, gen_label = synthesize_feature_test(vaeNet, attr_encoder, dataset, opt)
            with torch.no_grad():
                train_feature = attr_encoder(dataset.train_feature.to(opt.gpu))
                test_unseen_feature = attr_encoder(dataset.test_unseen_feature.to(opt.gpu))
                test_seen_feature = attr_encoder(dataset.test_seen_feature.to(opt.gpu))

            train_feature = train_feature.cpu()
            test_unseen_feature = test_unseen_feature.cpu()
            test_seen_feature = test_seen_feature.cpu()
            
            train_X = torch.cat((train_feature, gen_feat), 0)
            train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)
            if opt.zsl:
                # ZSL
                cls = classifier.CLASSIFIER(opt, gen_feat, gen_label, dataset, test_seen_feature, test_unseen_feature,
                                            dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5, 20,
                                            opt.nSample, False)
                result_zsl_soft.update(it, cls.acc)
                log_print("ZSL Softmax:", log_dir)
                log_print("Acc {:.2f}%  Best_acc [{:.2f}% | Iter-{}]".format(
                    cls.acc, result_zsl_soft.best_acc, result_zsl_soft.best_iter), log_dir)

            else:
                # GZSL
                cls = classifier.CLASSIFIER(opt, train_X, train_Y, dataset, test_seen_feature, test_unseen_feature,
                                    dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5,
                                            opt.classifier_steps, opt.nSample, True)

                result_gzsl_soft.update_gzsl(it, cls.acc_seen, cls.acc_unseen, cls.H)

                log_print("GZSL Softmax:", log_dir)
                log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                    cls.acc_unseen, cls.acc_seen, cls.H,  result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                    result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

            if opt.zsl:
                if cls.acc > best_performance[-1]:
                    best_performance = [cls.acc]
                    checkpoint = {}
                    checkpoint["iter"] = it
                    checkpoint["VAE"] = vaeNet.state_dict()
                    checkpoint["RelationNet"] = relationNet.state_dict()
                    checkpoint["AutoEncoder"] = cntnt_enc_dec.state_dict()
                    checkpoint["Encoder"] = attr_encoder.state_dict()
                    checkpoint["Classifier"] = cls.best_model_state_dict
                    #optimizer
                    checkpoint["optimizer"] = optimizer.state_dict()
                    checkpoint["relation_optimizer"] = relation_optimizer.state_dict()
                    checkpoint["attribute_encoder_optimizer"] = attribute_encoder_optimizer.state_dict()
                    checkpoint["classifier_optimizer"] = cls.best_optimizer_state_dict
                    torch.save(checkpoint, out_dir + '/Best_model_ZSL_' + opt.dataset + '_finetune_{}_{}.pth'.format(opt.finetune, opt.file_name))
                    log_print('Save model: ' + out_dir + '/Best_model_ZSL_' + opt.dataset + '_finetune_{}_{}.pth'.format(opt.finetune, opt.file_name), log_dir)
            else:
                if cls.H > best_performance[-1]:
                    best_performance = [cls.acc_unseen, cls.acc_seen, cls.H]
                    checkpoint = {}
                    checkpoint["iter"] = it
                    checkpoint["VAE"] = vaeNet.state_dict()
                    checkpoint["RelationNet"] = relationNet.state_dict()
                    checkpoint["AutoEncoder"] = cntnt_enc_dec.state_dict()
                    checkpoint["Encoder"] = attr_encoder.state_dict()
                    checkpoint["Classifier"] = cls.best_model_state_dict
                    # optimizer
                    checkpoint["optimizer"] = optimizer.state_dict()
                    checkpoint["relation_optimizer"] = relation_optimizer.state_dict()
                    checkpoint["attribute_encoder_optimizer"] = attribute_encoder_optimizer.state_dict()
                    checkpoint["classifier_optimizer"] = cls.best_optimizer_state_dict

                    torch.save(checkpoint, out_dir + '/Best_model_GZSL_' + opt.dataset + '_finetune_{}_{}.pth'.format(opt.finetune, opt.file_name))
                    log_print('Save model: ' + out_dir + '/Best_model_GZSL_' + opt.dataset + '_finetune_{}_{}.pth'.format(opt.finetune, opt.file_name), log_dir)

            # Zero-shot retrieval
            cls_centrild = np.zeros((dataset.ntest_class, test_unseen_feature.shape[1]))
            for i in range(dataset.ntest_class):
                cls_centrild[i] = torch.mean(gen_feat[gen_label == i,], dim=0)
            dist = cosine_similarity(cls_centrild, test_unseen_feature)

            precision_100 = torch.zeros(dataset.ntest_class)
            precision_50 = torch.zeros(dataset.ntest_class)
            precision_25 = torch.zeros(dataset.ntest_class)

            dist = torch.from_numpy(-dist)
            for i in range(dataset.ntest_class):
                is_class = dataset.test_unseen_label == i
                cls_num = int(is_class.sum())

                # 100%
                _, idx = torch.topk(dist[i, :], cls_num, largest=False)
                precision_100[i] = (is_class[idx]).sum().float() / cls_num

                # 50%
                cls_num_50 = int(cls_num / 2)
                _, idx = torch.topk(dist[i, :], cls_num_50, largest=False)
                precision_50[i] = (is_class[idx]).sum().float() / cls_num_50

                # 25%
                cls_num_25 = int(cls_num / 4)
                _, idx = torch.topk(dist[i, :], cls_num_25, largest=False)
                precision_25[i] = (is_class[idx]).sum().float() / cls_num_25

            precision_100 = precision_100.mean().item()
            precision_50 = precision_50.mean().item()
            precision_25 = precision_25.mean().item()

            log_print("retrieval results 100%%: %.3f 50%%: %.3f 25%%: %.3f" % (precision_100, precision_50, precision_25), log_dir)

            if best_precision_25 < precision_25:
                best_precision_25 = precision_25
                checkpoint = {}
                checkpoint["iter"] = it
                checkpoint["VAE"] = vaeNet.state_dict()
                checkpoint["RelationNet"] = relationNet.state_dict()
                checkpoint["AutoEncoder"] = cntnt_enc_dec.state_dict()
                checkpoint["Encoder"] = attr_encoder.state_dict()
                # optimizer
                checkpoint["optimizer"] = optimizer.state_dict()
                checkpoint["relation_optimizer"] = relation_optimizer.state_dict()
                checkpoint["attribute_encoder_optimizer"] = attribute_encoder_optimizer.state_dict()

                torch.save(checkpoint, out_dir + '/Best_model_RETRIEVAL_' + opt.dataset + '_finetune_{}_{}.pth'.format(opt.finetune, opt.file_name))
                log_print('Save retrieval model: ' + out_dir + '/Best_model_RETRIEVAL_' + opt.dataset + '_finetune_{}_{}.pth'.format(opt.finetune, opt.file_name), log_dir)
                log_print(f'Retrieval performance: 100%: {precision_100}, 50%: {precision_50}, 25%: {precision_25}', log_dir)
 
    if opt.zsl:
        return result_zsl_soft.best_acc
    else:
        return result_gzsl_soft.best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SUN',help='dataset: CUB, AWA2, APY, FLO, SUN')
    parser.add_argument('--out_dir', default='outputs',help='output directory path of experiment')
    parser.add_argument('--dataroot', default='data/SDGZSL_data', help='path to dataset')
    parser.add_argument('--file_name', default='0', type=str, help='set the file number')

    parser.add_argument('--image_embedding', default='res101', type=str)
    parser.add_argument('--class_embedding', default='att', type=str)

    parser.add_argument('--gen_nepoch', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train generater')

    parser.add_argument('--zsl', type=bool, default=False, help='Evaluate ZSL or GZSL')
    parser.add_argument('--finetune', type=bool, default=False, help='Use fine-tuned feature')
    parser.add_argument('--beta', type=float, default=1, help='tc weight')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight_decay')

    parser.add_argument('--kl_warmup', type=float, default=0.01, help='kl warm-up for VAE')
    parser.add_argument('--tc_warmup', type=float, default=0.001, help='tc warm-up')

    parser.add_argument('--vae_dec_drop', type=float, default=0.5, help='dropout rate in the VAE decoder')
    parser.add_argument('--vae_enc_drop', type=float, default=0.4, help='dropout rate in the VAE encoder')

    parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
    parser.add_argument('--classifier_steps', type=int, default=50, help='training steps of the classifier')

    parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
    parser.add_argument('--nSample', type=int, default=1200, help='number features to generate per class')

    parser.add_argument('--disp_interval', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--evl_interval',  type=int, default=400)
    parser.add_argument('--evl_start',  type=int, default=0)
    parser.add_argument('--manualSeed', type=int, default=5606, help='manual seed')

    parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
    parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
    parser.add_argument('--S_dim', type=int, default=1024)

    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--vae_weight', type=float, default=1.0, help='VAE loss weight')
    parser.add_argument('--relation_weight', type=float, default=15.0, help='relation loss weight')
    parser.add_argument('--recon_weight', type=float, default=15.0, help='reconstruction loss weight')
    opt = parser.parse_args()

    _ = train(opt)
