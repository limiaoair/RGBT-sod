def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=80):
    aa = epoch // decay_epoch
    if aa == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr
            lr=param_group['lr']
    elif aa == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr*0.1
            lr=param_group['lr']   
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
            lr=param_group['lr']
    return lr
#print(optimizer.state_dict()['param_groups'][0]['lr'])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
