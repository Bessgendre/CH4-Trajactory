import numpy as np
import os
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import train_test_split

from e3nn import o3

from mace.modules import MACE, RealAgnosticResidualInteractionBlock

from graph2mat import metrics

from graph2mat import (
    BasisConfiguration,
    PointBasis,
    BasisTableWithEdges,
    MatrixDataProcessor,
)
from graph2mat.bindings.torch import TorchBasisMatrixDataset, TorchBasisMatrixData

from graph2mat.bindings.e3nn import E3nnGraph2Mat

# from graph2mat.tools.viz import plot_basis_matrix

from graph2mat.models import MatrixMACE

from graph2mat.bindings.e3nn import E3nnEdgeMessageBlock

from tools import get_atom_list, get_atom_position_list

import pickle

# load the original data

with open('data_631g.pkl', 'rb') as f:
    data = pickle.load(f)
    
# The basis 6-31g
hydrogen = PointBasis("H", R=5, basis="2x0e", basis_convention="spherical")
carbon = PointBasis("C", R=5, basis="3x0e + 2x1o", basis_convention="spherical")
basis_list = [hydrogen, carbon]

# prepare basis table and preprocessor
table = BasisTableWithEdges([hydrogen, carbon])

processor = MatrixDataProcessor(
    basis_table=table, symmetric_matrix=True, sub_point_matrix=False
)

# form the dataset

def prepare_config(atom_data, density_matrixes):
# def prepare_config(atom_data):
    configs = []
    for i in range(len(atom_data)):
        config = BasisConfiguration(
            point_types = ["C", "H", "H", "H", "H"],
            positions = get_atom_position_list(atom_data[i]),
            basis = basis_list,
            cell=np.eye(3) * 100,
            pbc=(False, False, False),
            matrix = np.array(density_matrixes[i])
        )
        configs.append(config)
    return configs

configs = prepare_config(data['feature'], data['target'])

# split the data and create the loaders

train_configs, val_configs = train_test_split(configs, test_size=0.25, random_state=56)

train_dataset = TorchBasisMatrixDataset(train_configs, data_processor=processor)
val_dataset = TorchBasisMatrixDataset(val_configs, data_processor=processor)

train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True, pin_memory=True)

# define the model

num_interactions = 3
hidden_irreps = o3.Irreps("1x0e + 1x1o")

mace_model = MACE(
    r_max=10,
    num_bessel=10,
    num_polynomial_cutoff=10,
    max_ell=2,  # 1,
    interaction_cls=RealAgnosticResidualInteractionBlock,
    interaction_cls_first=RealAgnosticResidualInteractionBlock,
    num_interactions=num_interactions,
    num_elements=2,
    hidden_irreps=hidden_irreps,
    MLP_irreps=o3.Irreps("2x0e"),
    atomic_energies=torch.tensor([0, 0]),
    avg_num_neighbors=2,
    atomic_numbers=[0, 1],
    correlation=2,
    gate=None,
)

matrix_mace_model = MatrixMACE(
    mace_model,
    unique_basis=table,
    readout_per_interaction=True,
    edge_hidden_irreps=o3.Irreps("10x0e + 10x1o + 10x2e"),
    preprocessing_edges=E3nnEdgeMessageBlock,
    preprocessing_edges_reuse_nodes=False,
)

matrix_mace_model.load_state_dict(torch.load('./models_52/model_1000.pt'))


# prepare the trianing process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = matrix_mace_model.to(device)

loss_function = metrics.elementwise_mse

optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器
scheduler = StepLR(optimizer, step_size=200, gamma=0.732)

# Set up the training loop

n_steps = 2000
writer = SummaryWriter('runs_55_go_on')

# create the model directory
if not os.path.exists('./models_55_go_on'):
    os.makedirs('./models_55_go_on')

# Loop over the number of steps
for step in range(n_steps):
    model.train()
    for batch in train_loader:
        nodes_ref = batch.point_labels.to(device)
        edges_ref = batch.edge_labels.to(device)
        # Reset gradients
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        optimizer.zero_grad()

        # Make predictions for this batch
        predictions = model(batch)
        
        # Compute the loss
        loss, info = loss_function(
            nodes_pred=predictions['node_labels'],
            nodes_ref=nodes_ref,
            edges_pred=predictions['edge_labels'],
            edges_ref=edges_ref,
        )
        train_loss = loss.item()

        # Log the loss and RMSE to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), step)
        writer.add_scalar('Node RMSE/train', info["node_rmse"], step)
        writer.add_scalar('Edge RMSE/train', info["edge_rmse"], step)

        # Compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    scheduler.step()

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            nodes_ref=torch.tensor(batch.point_labels).to(device)
            edges_ref=torch.tensor(batch.edge_labels).to(device)
            
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            # Make predictions for this batch
            predictions = model(batch)
            
            # Compute the loss
            loss, info = loss_function(
                nodes_pred=predictions['node_labels'],
                nodes_ref=nodes_ref,
                edges_pred=predictions['edge_labels'],
                edges_ref=edges_ref,
            )
            test_loss = loss.item()
            
            # Log the validation loss and RMSE to TensorBoard
            writer.add_scalar('Loss/val', loss.item(), step)
            writer.add_scalar('Node RMSE/val', info["node_rmse"], step)
            writer.add_scalar('Edge RMSE/val', info["edge_rmse"], step)

    print(f"Step {step + 1}/{n_steps}\tTrain Loss: {train_loss}\tTest Loss: {test_loss}")
    
    # save the model every 100 steps
    if (step + 1) % 100 == 0:
        torch.save(model.state_dict(), f"./models_55_go_on/model_{1000 + step + 1}.pt")

# Close the writer
writer.close()
