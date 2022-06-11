import os
import shutil
from cgi import test
import argparse
import random
from models import *
from dataset_GBU import DATA_LOADER
from utils import *
import torch.backends.cudnn as cudnn
from classifier import LINEAR_LOGSOFTMAX


def eval(
    classifier,
    test_X,
    test_label,
    ntest,
    batch_size,
    num_total_classes
):
    predicted_label = torch.zeros(
        (test_label.size()[0], num_total_classes),
        dtype=torch.float32
    )

    start = 0
    with torch.no_grad():
        for i in range(0, ntest, batch_size):
            end = min(ntest, start + batch_size)
            if torch.cuda.is_available():
                output = classifier(test_X[start:end].to(opt.gpu))
            else:
                output = classifier(test_X[start:end])

            predicted_label[start:end, :] = output
            start = end

    scores = predicted_label.clone()
    scores = F.softmax(scores, dim=1)
    _, preds = torch.max(scores, 1)

    acc = eval_MCA(preds.numpy(), test_label.numpy())

    return acc

def main(opt):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    print("Random Seed: ", opt.manualSeed)
    np.random.seed(opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True

    opt.gpu = torch.device("cuda:"+opt.gpu if torch.cuda.is_available() else "cpu")

    dataset = DATA_LOADER(opt)
    dataset.feature_dim = dataset.train_feature.shape[1]
    batch_size = opt.nSample
    
    nclass = dataset.ntrain_class + dataset.ntest_class

    checkpoint = torch.load(opt.ckpt_path)

    opt.A_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim

    attr_encoder = Encoder(opt).to(opt.gpu)
    attr_encoder.to(opt.gpu)
    attr_encoder.load_state_dict(checkpoint['Encoder'])
    attr_encoder.eval()

    test_X = attr_encoder(dataset.test_unseen_feature.to(opt.gpu))
    test_X = test_X.cpu()
    test_unseen_label = dataset.test_unseen_label + dataset.ntrain_class

    num_total_classes = dataset.seenclasses.shape[0] + dataset.unseenclasses.shape[0]
    ntest = test_X.size()[0]

    classifier = LINEAR_LOGSOFTMAX(opt.A_dim, nclass)
    classifier.to(opt.gpu)
    classifier.load_state_dict(checkpoint['Classifier'])
    classifier.eval()

    acc_u = eval(
        classifier=classifier,
        test_X=test_X,
        test_label=test_unseen_label,
        ntest=ntest,
        batch_size=batch_size,
        num_total_classes=num_total_classes
    )

    test_X = attr_encoder(dataset.test_seen_feature.to(opt.gpu))
    test_X = test_X.cpu()
    ntest = test_X.size()[0]
    test_seen_label = dataset.test_seen_label

    acc_s = eval(
        classifier=classifier,
        test_X=test_X,
        test_label=test_seen_label,
        ntest=ntest,
        batch_size=batch_size,
        num_total_classes=num_total_classes
    )

    h = (2 * acc_u * acc_s) / (acc_u + acc_s)

    print(f'Unseen: {acc_u}, Seen: {acc_s}, H-mean: {h}')


def eval_MCA(preds, y):
    cls_label = np.unique(y)
    acc = list()
    for i in cls_label:
        acc.append((preds[y == i] == i).mean())
    return np.asarray(acc).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SUN',help='dataset: CUB, AWA2, APY, FLO, SUN')
    parser.add_argument('--dataroot', default='data/SDGZSL_data', help='path to dataset')
    parser.add_argument('--ckpt_path', default='weights/best_model.pth', type=str, help='path of checkpoint to load')

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

    _ = main(opt)
