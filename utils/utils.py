def atleast_kdim(x, ndim):
    shape = x.shape + (1,) * (ndim - len(x.shape))
    # print("in atleast_kdim():---------------------------")
    # print("the x.shape is: ", x.shape)
    # print("the len(x.shape) is: ", len(x.shape)) # torch.Size([2])->1
    # print("the ndim is: ", ndim) # 4
    # print("the ndim - len(x.shape) is: ", ndim - len(x.shape)) # 3
    # print("the (1, ) * (ndim - len(x.shape)) is: ",(1,) * (ndim - len(x.shape))) # (1, 1, 1)
    # print("the shape is: ", shape) # torch.Size([2]) + (1,1,1) = torch.Size(2, 1, 1, 1)
    # print("the return thing x.reshape(shape) is: ", x.reshape(shape))
    # print("----------------------------------------------")
    return x.reshape(shape)
    #x = tensor([False, False], device='cuda:0'), ndim = torch.Size([2, 3, 32, 32])