import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
import torch
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
import optuna
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from model.MLP import MLP
from model.GAT_1ly import GAT_1ly
from model.GAT_3ly import GAT_3ly
from model.GCN_1ly import GCN_1ly
from model.GCN_3ly import GCN_3ly

#exp_model = 'MLP'
exp_model = 'GAT_3ly'
#exp_model = 'GAT_1ly'
# exp_model = 'GCN_1ly'
#exp_model = 'GCN_3ly'

data=torch.load(proj_dir+'\\data\\dataset.pth')

sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                        batch_size=32, num_neighbors=[4],directed=False,
                        sampler=sampler)
if exp_model != 'MLP':
    print()
    total_num_nodes = 0
    for step, sub_data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
        print(sub_data)
        print()
        total_num_nodes += sub_data.num_nodes

    print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')

def train(model,lr,criterion,loss_list,val_loss_list):
    total_loss=0
    total_val_loss=0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    model.train()
    for sub_data in train_loader:
        optimizer.zero_grad()  # Clear gradients.
        out = model(sub_data.x, sub_data.edge_index)  # Perform a single forward pass.
        loss= criterion(torch.reshape(out[sub_data.train_mask], (-1,)), sub_data.y[:,0][sub_data.train_mask].type(torch.FloatTensor))
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        val_loss = criterion(torch.reshape(out[sub_data.val_mask], (-1,)), sub_data.y[:,0][sub_data.val_mask].type(torch.FloatTensor))
        total_loss+=loss
        total_val_loss+=val_loss
        output=torch.reshape(torch.round(out),(-1,))
    loss_list.append(total_loss.detach().numpy()/len(train_loader))
    val_loss_list.append(total_val_loss.detach().numpy()/len(train_loader))
    return loss,val_loss,loss_list,val_loss_list

def train_mlp(model,lr,criterion,loss_list,val_loss_list):
      optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x)  # Perform a single forward pass.
      loss = criterion(torch.reshape(out[data.train_mask], (-1,)), data.y[:,0][data.train_mask].type(torch.FloatTensor))
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      val_loss= criterion(torch.reshape(out[data.val_mask], (-1,)), data.y[:,0][data.val_mask].type(torch.FloatTensor))
      loss_list.append(loss.detach().numpy())
      val_loss_list.append(val_loss.detach())
      return loss,val_loss,loss_list,val_loss_list

def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

def test(model,data):
    model.eval()
    if exp_model == 'MLP':
        out = model(data.x)
    else:
        out = model(data.x, data.edge_index)
    output=torch.reshape(torch.round(out),(-1,))
    acc = accuracy(output[data.test_mask], data.y[:,0][data.test_mask])
    return acc,output,out

def objective(trial):
    if exp_model == 'GAT_3ly':
        params = {
              'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.001),
              'n_unit_1': trial.suggest_int("n_unit_1", 10,50),
              'n_unit_2': trial.suggest_int("n_unit_2", 10,50),
              'n_unit_3': trial.suggest_int("n_unit_3", 10,20)
              } 
        model = GAT_3ly(data.num_features,params['n_unit_1'],params['n_unit_2'],params['n_unit_3'])
        for epoch in range(1, 20):
            loss,val_loss,loss_list,val_loss_list = train(model,params['learning_rate'],torch.nn.BCELoss(),[],[])
    elif exp_model == 'GCN_3ly':
        params = {
              'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.001),
              'n_unit_1': trial.suggest_int("n_unit_1", 10,50),
              'n_unit_2': trial.suggest_int("n_unit_2", 10,50),
              'n_unit_3': trial.suggest_int("n_unit_3", 10,20)
              }
        model = GCN_3ly(data.num_features,params['n_unit_1'],params['n_unit_2'],params['n_unit_3'])
        for epoch in range(1, 20):
            loss,val_loss,loss_list,val_loss_list = train(model,params['learning_rate'],torch.nn.BCELoss(),[],[])
    elif exp_model == 'GAT_1ly':
        params = {
              'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.001),
              'n_unit_1': trial.suggest_int("n_unit_1", 10,50)
              }
        model = GAT_1ly(data.num_features,params['n_unit_1'])
        for epoch in range(1, 20):
            loss,val_loss,loss_list,val_loss_list = train(model,params['learning_rate'],torch.nn.BCELoss(),[],[])
    elif exp_model == 'GCN_1ly':
        params = {
              'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.001),
              'n_unit_1': trial.suggest_int("n_unit_1", 10,50)
              }
        model = GCN_1ly(data.num_features,params['n_unit_1'])
        for epoch in range(1, 20):
            loss,val_loss,loss_list,val_loss_list = train(model,params['learning_rate'],torch.nn.BCELoss(),[],[])
    elif exp_model == 'MLP':
        params = {
              'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.001),
              'n_unit_1': trial.suggest_int("n_unit_1", 10,100)
              }
        model = MLP(data.num_features,params['n_unit_1'])
        for epoch in range(1, 20):
            loss,val_loss,loss_list,val_loss_list = train_mlp(model,params['learning_rate'],torch.nn.BCELoss(),[],[])
        
    # Train
    return val_loss

def get_model(in_dim,n_uni1,n_unit2,n_unit3):
    if exp_model == 'GAT_3ly':
        model = GAT_3ly(in_dim,n_uni1,n_unit2,n_unit3)
    elif exp_model == 'GCN_3ly':
        model = GCN_3ly(in_dim,n_uni1,n_unit2,n_unit3)
    elif exp_model == 'GAT_1ly':
        model = GAT_1ly(in_dim,n_uni1)
    elif exp_model == 'GCN_1ly':
        model = GCN_1ly(in_dim,n_uni1)
    elif exp_model == 'MLP':
        model = MLP(in_dim,n_uni1)
    return model

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))

lr = best_trial.params['learning_rate']

if exp_model == 'GCN_3ly' or exp_model == 'GAT_3ly':
    model = get_model(data.num_features,best_trial.params['n_unit_1'],best_trial.params['n_unit_2'],best_trial.params['n_unit_3'])
else:
    model = get_model(data.num_features,best_trial.params['n_unit_1'],None,None)

n_epochs = 200

for epoch in range(1, n_epochs):
    if exp_model == 'MLP':
        loss,val_loss,loss_list,val_loss_list = train_mlp(model,lr,torch.nn.BCELoss(),[],[])
    else:
        loss,val_loss,loss_list,val_loss_list = train(model,lr,torch.nn.BCELoss(),[],[])
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, val Loss: {val_loss:.4f}')

acc,out,prob=test(model,data)
print(f'Test accuracy seag: {acc:.4f}')

conf_mat = confusion_matrix(data.y[:,0][data.test_mask],out[data.test_mask].detach().numpy())
cmn = conf_mat.astype('float')/conf_mat.sum(axis=1)[:, np.newaxis]
print('Seagrasses distribution Normalized Confusion Matrix - '+exp_model)
print(cmn)

print(classification_report(data.y[:,0][data.train_mask],out[data.train_mask].detach().numpy()))
df = pd.DataFrame(classification_report(data.y[:,0][data.train_mask],out[data.train_mask].detach().numpy(),output_dict=True)).transpose()
df.to_excel(proj_dir+'\\results\\'+exp_model+'_performances.xlsx')

torch.save(model, proj_dir+'\\results\\trained_models\\'+exp_model+'.pth')

