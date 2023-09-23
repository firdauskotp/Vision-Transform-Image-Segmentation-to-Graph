import torch.nn as tnn
import torch

class patch(tnn.Module):
    '''
    Referring to https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/vision_transformer/custom.py

    Solit image into patches and then embed them

    PARAMATERS | DATA TYPE - MEANING

    img_size | int - image size
    patch_size | int - patch size
    ch_in | int - number of input channels
    embed_dim | int - embedding dimension

    ATTRIBUTES | DATA TYPE - MEANING

    n_patches | int - number of patches inside of our image
    proj | tnn.Conv2d - Convolutional layer that does both the splitting into patches and their embedding
    '''

    def __init__(self, img_size, patch_size, ch_in=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        #self.ch_in = ch_in
        #self.embed_dim = embed_dim

        self.n_patches = (img_size // patch_size) ** 2

        self.proj = tnn.Conv2d(ch_in,embed_dim,kernel_size=patch_size,stride=patch_size)

    def forward(self, t):
        '''
        Run forward pass.

        PARAMETERS | PACKAGE - METHOD
        
        t | torch.Tensor -  Shape `(n_samples, n_patches, embed_dim)`

        RETURNS - METHOD

        torch.Tensor - Shape `(n_samples, n_patches, embed_dim)`

        '''

        t = self.proj(t) # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        t = t.flatten(2) # (n_samples, embed_dim, n_patches)
        t = t.transpose(1,2) # (n_samples, n_patches, embed_dim)

        return t
    
class Attention(tnn.Module):
    '''
    Attention mechanism

    PARAMETERS | DATA TYPE - MEANING

    dim | int - The input and out dimension of per token features
    n_heads | int - Number of attention heads
    qkv_bias | bool - If true then we include bias to the query, key and value projections
    attn_p | float - Dropout probability applied to the query, key and value tensors.
    proj_p | float - Dropout probability applied to the output tensor.

    ATTRIBUTES | DATA TYPE / PACKAGE - MEANING / METHOD

    scale | float - Normalising consent for the dot product
    qkv | tnn.Linear - Linear projection for the query, key, and value
    proj | tnn.Linear - Linear mapping that takes in the concatenated output of all attention heads and maps it into a new space
    attn_dop, proj_drop | tnn.Dropout - Dropout layers
    '''

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = tnn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = tnn.Dropout(attn_p)
        self.proj = tnn.Linear(dim, dim)
        self.proj_drop = tnn.Dropout(proj_p)

    def forward(self, t):
        '''
        Run forward pass

        PARAMETERS | PACKAGE - METHOD

        t | torch.Tensor - Shape `(n_samples, n_patches+1, dim)`

        RETURNS - METHOD

        torch.Tensor - Shape `(n_samples, n_patches+1, dim)`

        '''
    
        n_samples, n_tokens, dim = t.shape

        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(t) # (n_samples, n_patches +1, 3* dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim) # (n_samples, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2,0,3,1,4) # (3,n_samples, n_heads, head_dim, n_patches+1)

        q,k,v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale # (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches+1, n_patches+1)
        attn  = self.attn_drop(attn)

        weighted_avg = attn@v #(n_samples, n_heads, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(1,2) # (n_samples, n_patches+1, dim)

        t = self.proj(weighted_avg) # (n_samples, n_patches+1, dim)
        t = self.proj_drop(t) # (n_samples, n_patches+1, dim)

        return t
    
class MLP(tnn.Module):
    '''
    Multilayer perceptron

    PARAMETERS | DATA TYPE - MEANING

    in_features | int - Number of input features
    hidden_features | int - Number of nodes in the hidden layer
    out_features | int - Number of output features
    p | float - Dropout probability

    ATTRIBUTES | METHOD - MEANING

    fc | tnn.Linear - The first linear layer
    act | tnn.GELU - GELU activation function
    fc2 | tnn.Linear - The second linear layer
    drop | tnn.Dropout - Dropout layer
    '''

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = tnn.Linear(in_features, hidden_features)
        self.act = tnn.GELU()
        self.fc2 = tnn.Linear(hidden_features, out_features)
        self.drop = tnn.Dropout(p)

    def forward(self, t):
        '''
        Run forward pass

        PARAMETERS | PACKAGE - METHOD

        t | torch.Tensor - Shape `(n_samples, n_patches+1, in_features)`

        RETURNS - METHOD

        torch.Tensor - Shape `(n_samples, n_patches+1, out_features)`
        '''

        t = self.fc1(t) #(n_samples, n_patches+1, hidden_features)
        t = self.act(t) # (n_samples, n_patches+1, hidden_features)
        t = self.drop(t) # (n_samples, n_patches+1, hidden_features)
        t = self.fc2(t) # (n_samples, n_patches+1, out_features)
        t = self.drop(t) # (n_samples, n_patches+1)

        return t
    
class Block(tnn.Module):
    '''
    Transformer block

    PARAMETERS | DATA TYPES - MEANING

    dim | int - Embedding dimension
    n_heads | int - Number of attention heads
    mlp_ratio | float - Determines the hidden dimension of the MLP module with respect to dim
    qkv_bias | bool - if True, then we include bias to the query, key and value projections.
    P, attn_p | float - Dropout probability

    ATTRIBUTES | METHOD - MEANING

    norm1, norm2 | LayerNorm - Layer normalization
    attn | Attention - Attention module
    mlp | MLP - MLP module
    '''

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = tnn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = tnn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,hidden_features=hidden_features, out_features=dim)

    def forward(self, t):
        '''
        Run forward pass

        PARAMETERS | PACKAGE - METHOD

        t | torch.Tensor - Shape `(n_samples, n_patches+1, dim)`

        RETURNS - METHOD

        torch.Tensor - Shape `(n_samples, n_patches+1, dim)`
        '''

        t = t+self.attn(self.norm1(t))
        t = t + self.mlp(self.norm2(t))

        return t

class VisionTransformer(tnn.Module):
    '''
    Simplified implementation of the Vision transformer

    PARAMETERS | DATA TYPES - MEANING

    img_size | int - Both height and the width of the image
    patch_size | int - Both height and the width of the patch
    ch_in | int - Number of input channels
    n_classes | int  - Number of classes
    embed_dim | int - Dimensionality of the token/patch embeddings
    depth | int - Number of blocks
    n_heads | int - Number of attention heads
    mlp_ratio | float - Determines the hidden dimension of the `MLP` module
    qkv_bias | bool - If True, then we include bias to the query, key and value projections
    p, attn_p | float - Dropout probability

    ATTRIBUTES | PACKAGE/METHOD - MEANING

    patch_embed | patch - Instance of `patch` layer
    cls_token | tnn.Parameter - Learnable parameter that will represent the first token in the sequence. Has `embed_dim` elements
    pos_emb | tnn.Parameter - Positional embedding of the cls token + all the patches. Has `(n_patches+1) * embed_dim` elements
    pos_drop | tnn.Dropout - Dropout layer
    blocks tnn.ModuleList - List of `Block` modules
    norm | tnn.LayerNorm - Layer normalization
    '''

    def __init__(self, img_size=384, patch_size=16, ch_in=3, n_classes=1000, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.patch_embed = patch(img_size=img_size, patch_size=patch_size, ch_in = ch_in, embed_dim = embed_dim)
        self.cls_token = tnn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = tnn.Parameter(torch.zeros(1,1+self.patch_embed.n_patches, embed_dim))
        self.pos_drop = tnn.Dropout(p=p)
        self.blocks = tnn.ModuleList([
            Block(dim=embed_dim, n_heads=n_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p)
            for _ in range(depth)
        ])

        self.norm = tnn.LayerNorm(embed_dim, eps=1e-6)
        self.head = tnn.Linear(embed_dim, n_classes)

    def forward(self, t):
        '''
        Run the forward pass

        PARAMETERS | PACKAGE.METHOD - MEANING(?)

        t | torch.Tensor - Shape `(n_samples, ch_in, img_size, img_size)`

        RETURNS | PACKAGE.METHOD - MEANING

        logits | torch.Tensor - Logits over all the classes - `(n_samples, n_classes)`
        '''

        n_samples = t.shape[0]
        t=self.patch_embed(t)
        cls_token = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)
        t = torch.cat((cls_token, t), dim=1) # (n_samples, 1+n_patches, embed_dim)
        t=t+self.pos_embed #(n_samples, 1+n_patches, embed_dim)
        t=self.pos_drop(t)

        for block in self.blocks:
            t=block(t)

        t=self.norm(t)

        cls_token_final = t[:,0] #just the CLS token
        t = self.head(cls_token_final)

        return t










