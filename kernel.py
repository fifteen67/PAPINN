import torch
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class _PropagatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, domain_field, indices_row, indices_col, values, geometry, partition_size, gain):
        ctx.save_for_backward(indices_row, indices_col, values)
        ctx.geometry = geometry
        ctx.partition_size = partition_size
        ctx.gain = gain
        
        sensor_response = torch.zeros(geometry[0], 1, device=DEVICE)
        total_elements = len(values)
        
        for i in range(0, total_elements, partition_size):
            end = min(i + partition_size, total_elements)
            r = indices_row[i:end].long().to(DEVICE)
            c = indices_col[i:end].long().to(DEVICE)
            v = values[i:end].to(DEVICE)
            
            sub_kernel = torch.sparse_coo_tensor(torch.stack([r, c]), v, size=geometry)
            sensor_response += torch.sparse.mm(sub_kernel, domain_field)
            
        return sensor_response * gain

    @staticmethod
    def backward(ctx, grad_signal):
        indices_row, indices_col, values = ctx.saved_tensors
        
        grad_field = torch.zeros(ctx.geometry[1], 1, device=DEVICE)
        grad_scaled = grad_signal * ctx.gain
        total_elements = len(values)
        
        for i in range(0, total_elements, ctx.partition_size):
            end = min(i + ctx.partition_size, total_elements)
            r = indices_row[i:end].long().to(DEVICE)
            c = indices_col[i:end].long().to(DEVICE)
            v = values[i:end].to(DEVICE)
            
            sub_kernel_T = torch.sparse_coo_tensor(torch.stack([c, r]), v, size=(ctx.geometry[1], ctx.geometry[0]))
            grad_field += torch.sparse.mm(sub_kernel_T, grad_scaled)
            
        return grad_field, None, None, None, None, None, None

class AcousticPhysicsEngine(nn.Module):
    def __init__(self, state_path, grid_dim, sensor_dim, temporal_dim, partition_mb=5):
        super().__init__()
        
        state = torch.load(state_path, map_location='cpu')
        self.idx_row = state['indices_row']
        self.idx_col = state['indices_col']
        self.vals = state['values']
        
        self.grid_dim = grid_dim
        self.sensor_dim = sensor_dim
        self.temporal_dim = temporal_dim
        
        self.geometry = (sensor_dim * temporal_dim, grid_dim * grid_dim)
        self.partition_size = int(partition_mb * 1e6)
        self.system_gain = 1.0 
        
    def _domain_flatten(self, tensor_2d):
        return tensor_2d.permute(1, 0).reshape(-1, 1)

    def _domain_reconstruct(self, tensor_1d):
        return tensor_1d.view(self.grid_dim, self.grid_dim).permute(1, 0)

    def forward(self, field_map):
        flat_field = self._domain_flatten(field_map)
        
        response = _PropagatorFunction.apply(
            flat_field, self.idx_row, self.idx_col, self.vals, 
            self.geometry, self.partition_size, self.system_gain
        )
        
        return response.view(self.temporal_dim, self.sensor_dim)

    def adjoint_projection(self, sensor_data):
        flat_data = sensor_data.reshape(-1, 1)
        projected_field = torch.zeros(self.geometry[1], 1, device=DEVICE)
        
        total_nnz = len(self.vals)
        for i in range(0, total_nnz, self.partition_size):
            end = min(i + self.partition_size, total_nnz)
            r = self.idx_row[i:end].long().to(DEVICE)
            c = self.idx_col[i:end].long().to(DEVICE)
            v = self.vals[i:end].to(DEVICE)
            
            sub_T = torch.sparse_coo_tensor(torch.stack([c, r]), v, size=(self.geometry[1], self.geometry[0]))
            projected_field += torch.sparse.mm(sub_T, flat_data)
            
        return self._domain_reconstruct(projected_field)