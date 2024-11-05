import numpy as np
import torch
import argparse
import network
import loss
from torch.utils import data
import itertools
from utils.optimizer import get_bert
import loda_data
from utils.kmeans import get_kmeans_centers
from sklearn.cluster import KMeans
from utils.metric import Confusion



def get_args_parser():
    parser = argparse.ArgumentParser("CCLA for clustering", add_help=False)
    parser.add_argument(
        "--batch_size", default=400, type=int, help="Batch size per GPU")
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    # hyper-parameters
    parser.add_argument("--epochs", default=70, type=int)
    parser.add_argument("--lamda1", default=0.18, type=float)
    parser.add_argument("--lamda2", default=0.01, type=float)
    parser.add_argument("--lamda3", default=5, type=int)
    parser.add_argument("--lamda4", default=10, type=int)
    parser.add_argument("--threshold", default=0.95, type=float)
    parser.add_argument("--stage1_epoch", default=10, type=int)
    parser.add_argument("--stage2_epoch", default=1, type=int)
    parser.add_argument("--diagonal", default=0, type=int)


    # Model parameters
    parser.add_argument('--use_pretrain', type=str, default='SBERT', choices=["BERT", "SBERT", "PAIRSUPCON"])
    parser.add_argument("--feature_dim", default=128, type=int, help="dimension of ICH")
    parser.add_argument(
        "--instance_temperature",
        default=0.5,
        type=float,
        help="temperature of instance-level contrastive loss")
    parser.add_argument(
        "--cluster_temperature",
        default=1.0,
        type=float,
        help="temperature of cluster-level contrastive loss")


    # Optimizer parameters
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=5e-6,
        help="learning rate of backbone")
    parser.add_argument(
        "--lr_head",
        type=float,
        default=5e-4,
        help="learning rate of head")


    # Dataset parameters
    parser.add_argument(
        "--data_name",
        default="searchsnippets",
        type=str,
        help="dataset",
        choices=["agnews", "searchsnippets", "stackoverflow", "biomedical", "TS", "T", "S", "tweet"])
    parser.add_argument(
        "--class_num", default=8, type=int, help="number of the clusters")
    parser.add_argument('--max_length', type=int, default=32)


    # save and load parameters
    parser.add_argument(
        "--model_path",
        default="save/model/",
        help="path where to save, empty for no saving",)
    parser.add_argument("--save_freq", default=10, type=int, help="saving frequency")
    parser.add_argument(
        "--resume",
        default=False,
        help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, help="start epoch")

    return parser





def train(args, epoch, kmeans_labels):
    model.train()
    loss_epoch = 0
    clu_loss = 0
    ins_loss = 0
    ent_loss = 0
    cro_loss = 0
    h_cro_loss = 0

    sim_matrix_test = torch.zeros((args.class_num, args.class_num))
    num_matrix_test = torch.zeros((args.class_num, args.class_num))

    for step, (input_ids, attention_mask, index) in enumerate(train_Dataloader):
        z_i, z_j, c_i, c_j, h_hat1, h_hat2, S_weight1, S_weight2 = model(input_ids, attention_mask)
        if epoch <= args.stage1_epoch + args.stage2_epoch:
            pseudo_label = kmeans_labels[index].to('cuda')
        else:
            pseudo_label = torch.argmax(c_i, dim=1).to('cuda')

        loss_cross, loss_instance, loss_cluster, loss_entropy, h_loss = criterion.forward(args, epoch, z_i, z_j, c_i, c_j, pseudo_label, h_hat1, S_weight1, h_hat2, S_weight2)

        if epoch <= args.stage1_epoch:
            loss = args.lamda4 * loss_instance
        if args.stage1_epoch < epoch <= args.stage1_epoch + args.stage2_epoch:
            loss = args.lamda4 * loss_instance + args.lamda3 * loss_cross
        if epoch > args.stage1_epoch + args.stage2_epoch:
            loss = loss_cluster + args.lamda1 * loss_entropy - args.lamda2*h_loss + args.lamda3 * loss_cross + args.lamda4*loss_instance

        loss.backward()
        optimizer.step()
        optimizer_head.step()
        optimizer.zero_grad()
        optimizer_head.zero_grad()
        loss_epoch += loss.item()
        clu_loss += loss_cluster
        ins_loss += loss_instance
        ent_loss += loss_entropy
        cro_loss += loss_cross
        h_cro_loss += h_loss


    model.eval()
    with torch.no_grad():
        true_labels = []
        for step, (input_ids, attention_mask, true_label, index) in enumerate(test_Dataloader):
            corpus_embeddings, consistent_embed, mlp_label, mlp_prob, S_weight_test = model.forward_cluster(input_ids, attention_mask)
            if step == 0:
                true_labels = true_label.detach().cpu()
                mlp_labels = mlp_label.detach().cpu()
                all_embeddings = corpus_embeddings.detach().cpu().numpy()
                all_con_embeddings = consistent_embed.detach().cpu().numpy()
            else:
                true_labels = torch.cat((true_labels, true_label.detach().cpu()), dim=0)
                mlp_labels = torch.cat((mlp_labels, mlp_label.detach().cpu()), dim=0)
                all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().cpu().numpy()), axis=0)
                all_con_embeddings = np.concatenate((all_con_embeddings, consistent_embed.detach().cpu().numpy()), axis=0)

            true_label = true_label - 1
            for i in range(args.class_num):
                first_indices1 = (true_label == i)
                for j in range(args.class_num):
                    secon_indices1 = (true_label == j)
                    sub_matrix1 = S_weight_test[first_indices1][:, secon_indices1]
                    average_similarity1 = torch.sum(sub_matrix1)
                    number1 = sub_matrix1.shape[0]
                    sim_matrix_test[i, j] = sim_matrix_test[i, j] + average_similarity1
                    num_matrix_test[i, j] = num_matrix_test[i, j] + number1
        similarity_test = sim_matrix_test / num_matrix_test
        diagonal_elements = torch.diag(similarity_test)
        current_simi = torch.mean(diagonal_elements)
        print("test similarity matrix：", similarity_test)
        print('category similarity', current_simi)

        # MLP predict performance
        confusion1 = Confusion(args.class_num)
        confusion1.add(mlp_labels, true_labels)
        _, corresponding_index1 = confusion1.optimal_assignment(args.class_num)
        acc1 = confusion1.acc()
        clusterscores1 = confusion1.clusterscores(true_labels, mlp_labels)
        print(f'our model：    acc:{acc1:.4f}', 'Clustering scores:', clusterscores1)

        if epoch <= args.stage1_epoch + args.stage2_epoch:
            clustering_model = KMeans(n_clusters=args.class_num, random_state=args.seed, max_iter=3000, tol=0.01, n_init='auto')
            clustering_model.fit(all_embeddings)
            kmeans_labels = torch.tensor(clustering_model.labels_.astype(int))

    return loss_epoch, clu_loss, ins_loss, ent_loss, cro_loss, h_cro_loss, kmeans_labels



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('**************************************************************************************************************')
    print('lamda1:',args.lamda1, 'lamda2:',args.lamda2,'lamda3:',args.lamda3,'lamda4:',args.lamda4, 'threshold：', args.threshold, 'stage1_epoch:',args.stage1_epoch, 'stage2_interval:',args.stage2_epoch)
    print('**************************************************************************************************************')

    # load model
    text_model, tokenizer = get_bert(args)
    model = network.Network(text_model, tokenizer, args.feature_dim, args.class_num)
    model = model.to('cuda')

    # load data
    text0, text1, label = loda_data.explict_augmentation_loader(args.data_name)
    indices = torch.randperm(label.size)
    text0_shuffled = text0[indices]
    label_shuffled = label[indices]

    dataset_T = loda_data.train_Dataset(text0, text1, tokenizer, args)
    train_Dataloader = torch.utils.data.DataLoader(dataset_T, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
    dataset_C = loda_data.test_Dataset(text0_shuffled, label_shuffled, tokenizer, args)
    test_Dataloader = torch.utils.data.DataLoader(dataset_C, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=1)

    # loss
    loss_device = torch.device("cuda")
    criterion = loss.ContrastiveLoss(args.batch_size, args.batch_size, args.class_num, args.instance_temperature,
                                     args.cluster_temperature, loss_device).to(loss_device)


    # optimizer
    optimizer = torch.optim.Adam(model.backbone.parameters(),
                                lr=args.lr_backbone)
    optimizer_head = torch.optim.Adam(itertools.chain(model.instance_projector.parameters(),
                                                      model.cluster_projector.parameters(),
                                                      model.TransformerEncoderLayer.parameters()),lr=args.lr_head)

    # generate original pseudo-label
    distance_lists = []
    kmeans_labels = get_kmeans_centers(args, model, test_Dataloader, args.class_num)
    restord_labels = torch.empty_like(kmeans_labels)
    restord_labels[indices] = kmeans_labels
    kmeans_labels = restord_labels


    loss_all = []
    Acc_all = []
    NMI_all = []
    similarity_lists = []
    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch, clu_loss, ins_loss, ent_loss, cro_loss, h_cro_loss, pseudo_label = train(args, epoch, kmeans_labels)
        if epoch <= args.stage1_epoch:
            kmeans_labels[indices] = pseudo_label
        loss_all.append(loss_epoch/ len(train_Dataloader))

        print(f"Name: {args.data_name}\t "
              f"Epoch [{epoch}/{args.epochs}]\t "
              f"Loss: {loss_epoch / len(train_Dataloader):.3f}\t "
              f"clu_Loss: {clu_loss / len(train_Dataloader):.3f}\t "
              f"ins_Loss: {ins_loss / len(train_Dataloader):.3f}\t "
              f"ent_Loss: {ent_loss / len(train_Dataloader):.3f}\t "
              f"cro_Loss: {cro_loss / len(train_Dataloader):.3f}\t "
              f"h_cro_Loss: {h_cro_loss / len(train_Dataloader):.3f}\t ")


