from numpy.random.mtrand import randint
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from glob import glob
from scipy.special import expit
#import Granules_labelling

EPS = 1e-10

def acc(label, predicted):
    seg_acc = (label.cpu() == torch.argmax(predicted, axis=1).cpu()).sum() / torch.numel(label.cpu())
    return seg_acc

def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])

def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist

def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc

def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.
    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.
    Args:
        hist: confusion matrix.
    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc

def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc

def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice

def Per_class_OPA(hist):
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    return per_class_acc

def Per_class_jaccard(hist):
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    return jaccard

def Per_class_dice(hist):
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    return dice

def eval_metrics_sem(true, pred, num_classes, device):
    """Computes various segmentation metrics on 2D feature maps.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.
    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    hist = torch.zeros((num_classes, num_classes)).to(device)
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    pc_opa = Per_class_OPA(hist)
    pc_j = Per_class_jaccard(hist)
    pc_d = Per_class_dice(hist)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice, pc_opa, pc_j, pc_d

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape[0], arr.shape[1] 
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
def model_eval(f, m, device, size):
    file = np.load(f)
    map_f = file['smap'].astype(np.float32)
    mask_map_f = file['cmask_map'].astype(np.float32)
    dx = blockshaped(map_f, size, size)
    dy = blockshaped(mask_map_f, size, size)
    data = torch.cat((torch.unsqueeze(torch.Tensor(dx),1),torch.unsqueeze(torch.Tensor(dy),1)), 1)
    
    model_unet = model.UNet(n_channels=1, n_classes=5, bilinear=False).to(device)
    model_unet.load_state_dict(m)
    model_unet.eval()

    x=torch.unsqueeze(data[:,0,:,:],1)
    y=torch.unsqueeze(data[:,1,:,:],1).to(torch.int32)
    
    with torch.no_grad():    
         pred_mask = model_unet(x.to(device))        
    pred_mask_np = pred_mask.cpu().detach().numpy()  
    pred_mask_class = torch.argmax(pred_mask, axis=1)
    pred_mask_class_np=pred_mask_class.cpu().detach().numpy()

    val_overall_pa, val_per_class_pa, val_jaccard_index, val_dice_index, val_pc_opa, val_pc_j, val_pc_d = eval_metrics_sem(y.to(device), pred_mask_class.to(device), 5, device)
    a = acc(y,pred_mask).numpy()

    prop_perclass = []
    for m in range(5):
        partial=[]
        if map_f.shape[0] % size == 0:
            slides = int(map_f.shape[0]/size)
            for i in range(slides):
                p = pred_mask_np[:,m,:,:]
                partial.append(np.concatenate([x for x in p[i*slides:slides*i+slides]], axis=1))
                pred_total_mask=np.concatenate(partial, axis=0)
        else:
            raise AttributeError('Full map can not be divided')
        prop_perclass.append(pred_total_mask)
    
    partial=[]
    if map_f.shape[0] % size == 0:
        slides = int(map_f.shape[0]/size)
        for i in range(slides):
            partial.append(np.concatenate([x for x in pred_mask_class_np[i*slides:slides*i+slides]], axis=1))
            pred_total_class_mask=np.concatenate(partial, axis=0)
    else:
        raise AttributeError('Full map can not be divided')

    losses = [a, val_overall_pa, val_per_class_pa, val_jaccard_index, val_dice_index, val_pc_opa, val_pc_j, val_pc_d]

    return map_f, mask_map_f, np.array(pred_total_class_mask), np.array(prop_perclass), losses
    
def model_eval_full(f, m, device, size=512):
    file = np.load(f)
    map_f = file['smap'].astype(np.float32)
    mask_map_f = file['cmask_map'].astype(np.float32)
    map_f = map_f/map_f.max()

    map_f_s = len(map_f) 
    dx=[]
    dy=[]
    if map_f_s >= size:
        dx.append(map_f[:size,:size])
        dx.append(map_f[-size:,:size])
        dx.append(map_f[:size,-size:])
        dx.append(map_f[-size:,-size:])

        dy.append(mask_map_f[:size,:size])
        dy.append(mask_map_f[-size:,:size])
        dy.append(mask_map_f[:size,-size:])
        dy.append(mask_map_f[-size:,-size:])
    else:
        raise ValueError('Map size lower that requested - change the requested/input size keyword')

    dx=np.array(dx)
    dy=np.array(dy)

    data = torch.cat((torch.unsqueeze(torch.Tensor(dx),1),torch.unsqueeze(torch.Tensor(dy),1)), 1)
    
    model_unet = model.UNet(n_channels=1, n_classes=5, bilinear=False).to(device)
    model_unet.load_state_dict(m)
    model_unet.eval()

    x=torch.unsqueeze(data[:,0,:,:],1)
    y=torch.unsqueeze(data[:,1,:,:],1).to(torch.int32)
    
    with torch.no_grad():    
         pred_mask = model_unet(x.to(device))

    pred_mask_np = pred_mask.cpu().detach().numpy()  
    pred_mask_class = torch.argmax(pred_mask, axis=1)
    pred_mask_class_np=pred_mask_class.cpu().detach().numpy()

    val_overall_pa, val_per_class_pa, val_jaccard_index, val_dice_index, val_pc_opa, val_pc_j, val_pc_d = eval_metrics_sem(y.to(device), pred_mask_class.to(device), 5, device)
    a = acc(y,pred_mask).numpy()

    losses = [a, val_overall_pa, val_per_class_pa, val_jaccard_index, val_dice_index, val_pc_opa, val_pc_j, val_pc_d]
    
    return dx, dy, pred_mask_np, pred_mask_class_np, losses

def metrics_plots(l, save=False, Title='Model 1'):
  plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
  plt.rcParams.update({'font.size': 28})

  plot_losses = np.array(l)
  plt.figure(figsize=(12,8))
  plt.plot(plot_losses[:,0], plot_losses[:,1], '--b')
  plt.plot(plot_losses[:,0], plot_losses[:,2], '.-b')
  plt.plot(plot_losses[:,0], plot_losses[:,3], '--r')
  plt.plot(plot_losses[:,0], plot_losses[:,5], '.-r')
  plt.plot(plot_losses[:,0], plot_losses[:,6], color='tab:green')
  plt.plot(plot_losses[:,0], plot_losses[:,7], color='tab:orange')
  plt.xlabel('Epochs')
  plt.ylabel('Loss/accuracy')
  plt.ylim(0.0, 1.0)
  plt.xlim(0.0, 100.0)
  plt.grid()
  plt.legend(['Training Loss', 
              'Training Overall pixel accuracy', 
              'Validation Loss',
              'Validation Overall pixel accuracy', 
              'Validation PerClass accuracy',
              'Validation Jaccard index'], fontsize=12, loc='lower right') # using a named size
  #plt.title(Title)
  plt.tight_layout()
  plt.show()
  if save == True:
    plt.savefig('Plot.pdf')

def comparative_maps(raw_map, gt_mask, p_mask, bin_classes, save=False):
    plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    plt.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,7), sharex=True, sharey=True)
    im1=ax[0].imshow(raw_map, origin='lower', cmap='gray')
    im2=ax[1].imshow(gt_mask, origin='lower', cmap = plt.get_cmap('PiYG', 5))
    im3=ax[2].imshow(p_mask, origin='lower', cmap = plt.get_cmap('PiYG', 5))
    values = np.unique(gt_mask.ravel())
    colors = [im2.cmap(im2.norm(value)) for value in values]

    patches = [mpatches.Patch(color=colors[i], 
    label="{l}".format(l=bin_classes[i])) for i in range(len(bin_classes))]
    lgd = plt.legend(handles=patches, bbox_to_anchor=(2.1, 0.75), loc=1, borderaxespad=0. , ncol=1)
    corres=np.round(np.count_nonzero(gt_mask == p_mask)*100/(768*768), 2)
    ax[0].set_title('Original map')
    ax[1].set_title('Ground-truth map')
    ax[2].set_title('Predicted map')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    if save == True:
        fig.savefig('Cplot.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

def comparative_maps_wz(raw_map, gt_mask, p_mask, bin_classes, save=False):
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    plt.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,7), sharex=True, sharey=True)

    im1=ax[0].imshow(raw_map, origin='lower', cmap='gray')
    axins1 = zoomed_inset_axes(ax[0], 2, loc=2)
    axins1.imshow(raw_map, origin='lower', cmap='gray')
    axins1.set_xlim(549, 699)
    axins1.set_ylim(157, 307)
    axins1.set_xticks([])
    axins1.set_yticks([])
    mark_inset(ax[0], axins1, loc1=1, loc2=3, fc="none", ec="0.5", color='blue')

    im2=ax[1].imshow(gt_mask, origin='lower', cmap = plt.get_cmap('PiYG', 5))
    axins2 = zoomed_inset_axes(ax[1], 2, loc=2)
    axins2.imshow(gt_mask, origin='lower', cmap = plt.get_cmap('PiYG', 5))
    axins2.set_xlim(549, 699)
    axins2.set_ylim(157, 307)
    axins2.set_xticks([])
    axins2.set_yticks([])
    mark_inset(ax[1], axins2, loc1=1, loc2=3, fc="none", ec="0.5")

    im3=ax[2].imshow(p_mask, origin='lower', cmap = plt.get_cmap('PiYG', 5))
    axins3 = zoomed_inset_axes(ax[2], 2, loc=2)
    axins3.imshow(p_mask, origin='lower', cmap = plt.get_cmap('PiYG', 5))
    axins3.set_xlim(549, 699)
    axins3.set_ylim(157, 307)
    axins3.set_xticks([])
    axins3.set_yticks([])
    mark_inset(ax[2], axins3, loc1=1, loc2=3, fc="none", ec="0.5")

    values = np.unique(gt_mask.ravel())
    colors = [im2.cmap(im2.norm(value)) for value in values]

    patches = [mpatches.Patch(color=colors[i], 
    label="{l}".format(l=bin_classes[i])) for i in range(len(bin_classes))]
    lgd = plt.legend(handles=patches, bbox_to_anchor=(5.3, 0.50), loc=1, borderaxespad=0. , ncol=1)
    corres=np.round(np.count_nonzero(gt_mask == p_mask)*100/(768*768), 2)
    ax[0].set_title('Original map')
    ax[1].set_title('Ground-truth map')
    ax[2].set_title('Predicted map')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    if save == True:
        fig.savefig('Cplot_zoom.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

def probability_maps(raw_map, prob_maps, bin_classes, save=True):
    plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    plt.rcParams.update({'font.size': 18})
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30,10), sharex=True, sharey=True)

    im2=ax[0].imshow(expit(prob_maps[0]), origin='lower', vmin=0.0, vmax=1.0)
    ax[0].set_title(bin_classes[0])
    divider = make_axes_locatable(ax[0])
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, orientation='vertical')

    im3=ax[1].imshow(expit(prob_maps[1]), origin='lower', vmin=0.0, vmax=1.0)
    ax[1].set_title(bin_classes[1])
    divider = make_axes_locatable(ax[1])
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax3, orientation='vertical')
    
    im4=ax[2].imshow(expit(prob_maps[2]), origin='lower', vmin=0.0, vmax=1.0)
    ax[2].set_title(bin_classes[2])
    divider = make_axes_locatable(ax[2])
    cax4 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im4, cax=cax4, orientation='vertical')
    
    im5=ax[3].imshow(expit(prob_maps[3]), origin='lower', vmin=0.0, vmax=1.0)
    ax[3].set_title(bin_classes[3])
    divider = make_axes_locatable(ax[3])
    cax5 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im5, cax=cax5, orientation='vertical')
    
    im6=ax[4].imshow(expit(prob_maps[4]), origin='lower', vmin=0.0, vmax=1.0)
    ax[4].set_title(bin_classes[4])
    divider = make_axes_locatable(ax[4])
    cax6 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im6, cax=cax6, orientation='vertical')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    if save == True:
        fig.savefig('Pmaps.pdf', bbox_inches='tight')

def test_centers(mask, cx, cy):
    bin_classes = ['Intergranular lane', 'Granules with dots', 'Granules with lanes',
                   'Complex-shape granules','Normal-shape granules']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), sharex=True, sharey=True)
    im=ax.imshow(mask, origin='lower', cmap = plt.get_cmap('PiYG', 5))
    ax.scatter(cy, cx, color = 'blue')
    values = np.unique(mask.ravel())
    colors = [im.cmap(im.norm(value)) for value in values]
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    patches = [mpatches.Patch(color=colors[i], 
    label="Class: {l}".format(l=bin_classes[i])) for i in range(len(bin_classes))]
    lgd = plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.2), loc=8, borderaxespad=0.)
    fig.savefig('Test_centers.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

def class_prop(root):
    file_list = sorted(glob(root+'*.npz'))
    mask_smap = []
    for f in file_list:
        file = np.load(f)
        mask_smap.append(file['cmask_map'].astype(np.float32))

    mask_smap=np.array(mask_smap)
    values, counts = np.unique(mask_smap, return_counts=True)
    for v in values:
        print('Class {} - proportion {}'.format(v, counts[int(v)]/mask_smap.size))

#def test_Imax(save_file, m, bin_classes, s=512, frame_num=4, save=True):
#    fs = Granules_labelling.IMaX_maps(save_file)
#    size = fs.shape[0]
#    #random_ind = [random.randint(20, size-20) for x in range(frame_num)]
#    random_ind =[24, 69, 75, 95]
#
#    model_unet = model.UNet(n_channels=1, n_classes=5, bilinear=False).to('cpu')
#    model_unet.load_state_dict(m)
#    model_unet.eval()
#    data=[]
#
#    for ind in random_ind:
#        fs.select_map(ind)
#        map_f = fs.smap
#        map_f = map_f/map_f.max()
#        map_f_s = len(map_f) 
#
#        dx=[]
#        if map_f_s > size:
#            x1 = int(abs(map_f_s/2) - (s/2))
#            x2 = int(abs(map_f_s/2) + (s/2))
#            dx.append(map_f[x1:x2,x1:x2])
#        else:
#            raise ValueError('Map size lower that requested - change the requested/input size keyword')
#
#        dx=np.array(dx)
#        partial = torch.unsqueeze(torch.Tensor(dx),1)
#        x=torch.unsqueeze(partial[:,0,:,:],1)
#  
#        with torch.no_grad():    
#            pred_mask = model_unet(x.to('cpu'))          
#        pred = torch.argmax(pred_mask, axis=1)
#        pred_np=pred.cpu().detach().numpy()
#
#        data.append([dx[0], pred_np[0]])
#
#    plt.rcParams.update({
#    "font.family": "sans-serif",
#    "font.sans-serif": ["Helvetica"]})
#    plt.rcParams.update({'font.size': 15})
#    data= np.array(data)
#    values = np.unique(data[:,1,:,:].ravel())
#
#    fig, ax = plt.subplots(nrows=2, ncols=frame_num, figsize=(20,9), sharex=True, sharey=True)
#      
#    for d in range(len(data)):
#        im0=ax[0][d].imshow(data[d,0,:,:], origin='lower', cmap='gray')
#        im1=ax[1][d].imshow(data[d,1,:,:], origin='lower', cmap = plt.get_cmap('PiYG', 5))
#        colors = [im1.cmap(im1.norm(value)) for value in values]
#
#        ax[0][d].set_title('Original map')
#        ax[1][d].set_title('Model Predicted map')
#        ax[0][d].set_xticks([])
#        ax[1][d].set_yticks([])
#
#    patches = [mpatches.Patch(color=colors[i], 
#    label="Class: {l}".format(l=bin_classes[i])) for i in range(len(bin_classes))]
#    lgd = plt.legend(handles=patches, loc=8, bbox_to_anchor=(-1.68, -0.3, 0.5, 0.5), borderaxespad=0. , ncol=3)
#    
#    if save == True:
#        fig.savefig('ImaXplot.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
#