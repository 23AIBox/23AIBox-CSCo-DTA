def train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, lr, epoch,
          batch_size, affinity_graph, drug_pos, target_pos):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=lr, weight_decay=0)
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        ssl_loss, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs,
                                                                  target_graph_batchs, drug_pos, target_pos)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + ssl_loss
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


def test(model, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_pos,
         target_pos):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            _, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs, target_graph_batchs, drug_pos, target_pos)    # 消融实验
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train_predict():
    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    affinity_mat = load_data(args.dataset)
    train_data, test_data, affinity_graph, drug_pos, target_pos = process_data(affinity_mat, args.dataset, args.num_pos, args.pos_threshold)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    drug_graphs_dict = get_drug_molecule_graph(
        json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=affinity_graph.num_drug)
    target_graphs_dict = get_target_molecule_graph(
        json.load(open(f'data/{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), args.dataset)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=affinity_graph.num_target)

    print("Model preparation... ")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = CSCoDTA(tau=args.tau,
                    lam=args.lam,
                    ns_dims=[affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256],
                    embedding_dim=128,
                    dropout_rate=args.edge_dropout_rate)
    predictor = PredictModule()
    drug_pos = drug_pos.to(device)
    target_pos = target_pos.to(device)
    model.to(device)
    predictor.to(device)

    print("Start training...")
    for epoch in range(args.epochs):
        train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, args.lr, epoch+1,
              args.batch_size, affinity_graph, drug_pos, target_pos)
        G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader,
                    affinity_graph, drug_pos, target_pos)

        r = model_evaluate(G, P)
        print("result: ", r)

    print('\npredicting for test data')
    G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph,
                drug_pos, target_pos)
    result = model_evaluate(G, P)
    print("result:", result)


if __name__ == '__main__':
    import os
    import argparse
    import torch
    import json

    from collections import OrderedDict
    from torch import nn
    from itertools import chain
    from data_process import load_data, process_data, get_drug_molecule_graph, get_target_molecule_graph
    from utils import GraphDataset, collate, model_evaluate
    from models import CSCoDTA, PredictModule

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--num_pos', type=int, default=3)
    parser.add_argument('--pos_threshold', type=float, default=8.0)
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    train_predict()



