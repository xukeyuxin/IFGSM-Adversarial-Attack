import numpy as np
import tensorflow as tf


# def gkern(kernlen=21, nsig=3):
#     """Returns a 2D Gaussian kernel array."""
#     import scipy.stats as st

#     interval = (2*nsig+1.)/(kernlen)
#     x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#     kernel = kernel_raw/kernel_raw.sum()
#     return kernel

# class GaussianBlur(nn.Module):
#     def __init__(self, kernel):
#         super(GaussianBlur, self).__init__()
#         self.kernel_size = len(kernel)
#         print('kernel size is {0}.'.format(self.kernel_size))
#         assert self.kernel_size % 2 == 1, 'kernel size must be odd.'
        
#         self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
#         self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
 
#     def forward(self, x):
#         x1 = x[:,0,:,:].unsqueeze_(1)
#         x2 = x[:,1,:,:].unsqueeze_(1)
#         x3 = x[:,2,:,:].unsqueeze_(1)
#         padding = self.kernel_size // 2
#         x1 = F.conv2d(x1, self.weight, padding=padding)
#         x2 = F.conv2d(x2, self.weight, padding=padding)
#         x3 = F.conv2d(x3, self.weight, padding=padding)
#         x = torch.cat([x1, x2, x3], dim=1)
#         return x
    
    
# def get_gaussian_blur(kernel_size, device):
#     kernel = gkern(kernel_size, 2).astype(np.float32)
#     gaussian_blur = GaussianBlur(kernel)
#     return gaussian_blur.to(device)


class GaussianBlur(object):
    def __init__(self,kernel_size = 3,sigma = 0.01):
        self.kernel_size = kernel_size
        self.gauss_filter = self.gauss_2d_kernel(3, sigma).astype(dtype=np.float32)
        self.kernel = self.gauss_filter.reshape([kernel_size, kernel_size, 1, 1])

    def gaussian_blur(self,image,cdim=3):
        # kernel as placeholder variable, so it can change
        kernel = self.gauss_filter
        kernel_size = self.kernel_size
        outputs = []
        pad_w = (kernel_size - 1) // 2
        padded = tf.pad(image, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
        for channel_idx in range(cdim):
            data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
            data_c = tf.nn.conv2d(data_c, self.kernel, [1, 1, 1, 1], 'VALID')
            outputs.append(data_c)
        return tf.concat(outputs, axis=3)

    def gauss_2d_kernel(self,kernel_size = 3,sigma = 0):
        kernel = np.zeros([kernel_size,kernel_size])
        center = (kernel_size - 1) /2
        if sigma == 0:
            sigma = ((kernel_size-1)*0.5 - 1)*0.3 + 0.8
    
        s = 2*(sigma**2)
        sum_val = 0
        for i in range(0,kernel_size):
            for j in range(0,kernel_size):
                x = i-center
                y = j-center
                kernel[i,j] = np.exp(-(x  **2+y**2) / s)
                sum_val += kernel[i,j]
        sum_val = 1/sum_val
        return kernel*sum_val
    
    def __call__(self,input):
        output = self.gaussian_blur(input)
        return output


# if __name__ == '__main__':
#     print(gkern().shape)