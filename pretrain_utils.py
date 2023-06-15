from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch


'''
gen_matched_data(data, n)
inputs: 
- data, a zpr data loader generated to load jet data
- n, the amount of data you want to use to pretrain. n MUST be <= len(data)

returns:
- torch Dataloader object to load pretraining pairs of jets.
    - 50% of jet pairs will be matched correctly
    - 50% matched incorrectly (uniform incorrect distribution over all possible incorrect labels)

'''
def gen_matched_data(data, n):
    indices = torch.randperm(len(data))[:n]
    dt = data[indices]
    labels = dt[3]
    jets = dt[0]

    mask1 = torch.squeeze(torch.nonzero(torch.all(labels == torch.tensor([1., 0., 0., 0.]), dim=1)))
    mask2 = torch.squeeze(torch.nonzero(torch.all(labels == torch.tensor([0., 1., 0., 0.]), dim=1)))
    mask3 = torch.squeeze(torch.nonzero(torch.all(labels == torch.tensor([0., 0., 1., 0.]), dim=1)))
    mask4 = torch.squeeze(torch.nonzero(torch.all(labels == torch.tensor([0., 0., 0., 1.]), dim=1)))
    masks = [mask1,mask2,mask3,mask4]


    pre_train_jets = []
    truth = []
    for v in range(n):
        index = torch.randint(0, n, (1,)).item()
        cur_label = labels[index]
        cur_jet = jets[index]
        if v < n/2:
            #Get the same class of jets
            cur_mask = masks[torch.squeeze(torch.nonzero(cur_label)).item()]
            exclude_index = index
            num_elements = cur_mask.numel() - 1
            random_index = torch.randint(0, num_elements, (1,)).item()
            if random_index >= exclude_index:
                random_index += 1
            random_element = cur_mask[random_index]
            truth.append(1)
        else:
            #Get different classes of jets
            exclude_index = torch.squeeze(torch.nonzero(cur_label)).item()
            num_elements = 3
            random_index = torch.randint(0, num_elements, (1,)).item()
            if random_index >= exclude_index:
                random_index += 1
            cur_mask = masks[random_index]
            rand_mask_index= torch.randint(0, cur_mask.numel(), (1,)).item()
            random_element = cur_mask[rand_mask_index]
            truth.append(0)
        matched_jets = [jets[index], jets[random_element]]
        pre_train_jets.append(matched_jets)
    # Assuming you have a list of 5000 pairs of tensors
    pairs = pre_train_jets
    # Combine the tensors into a single tensor
    combined_matched_jets = torch.stack([torch.stack(pair, dim=0) for pair in pairs], dim=0)
    truth = torch.tensor(truth)
    permutation = torch.randperm(n)

    combined_matched_jets = combined_matched_jets[permutation]
    truth = truth[permutation]

    loader = DataLoader(TensorDataset(combined_matched_jets,truth), batch_size=20)#,shuffle=(train_sampler is None),)
    return loader