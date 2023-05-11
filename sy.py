import torch
# Create tensor A
A = torch.randn(3, 128, 2)

# Reshape A to be a 3D tensor with an extra singleton dimension
A = A.unsqueeze(2)

# Calculate pairwise differences using broadcasting
B = A - A.transpose(1, 2)

# Mask the diagonal elements
mask = torch.eye(128).bool()
B = B[:, ~mask, :].view(2, -1)

# Verify that B has shape [2, 16128]
assert B.shape == (2, 16128)