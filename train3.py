# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:47:05 2019

@author: marry
@target: training
"""

from VGG import vgg19, VGG2
from Generator import CartoonGenerator
from Discriminator import Discriminator
from Loss import ContentLoss, AdversialLoss, BinaryCrossEntropyLoss
from random import shuffle
from PIL import Image
import torchvision.models as models
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=int, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--batch_size', type=int, default=4, help='batch size during training, default=8')

opt = parser.parse_args()
print(opt)

IMG_SIZE = 256

#### some metric functions
def binary_cross_entropy_metric(y_pred, y_true):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0-eps)
    return -(y_true * np.log(y_pred+eps) + (1-y_true) * np.log(1-y_pred+eps)).mean()

def binary_accuracy_metric(y_pred, y_true):
    y_pred = np.where(y_pred > 0.5, 1, 0)
    y_pred = y_pred.astype(np.int32)
    return (y_true == y_pred).mean()


if __name__ == '__main__':
    '''Doc prepare'''
    check_pth = os.path.join(os.getcwd(), 'checkpoints')
    if os.path.exists(check_pth) is not True:
        os.mkdir(check_pth)
    
    txt_pth = os.path.join(check_pth, 'records.txt')                     
    txt_handle = open(txt_pth, 'w')
    
    '''Global Variables'''
    EPOCH = 101
    gpu = 0
    omega = 10  # loss = lossAdv + omega * lossCon
    real_label = 1.0
    fake_label = 0.0

    datasets_pth = {
        'trainA': os.path.join(os.getcwd(), 'datasets', 'real'),
        'trainB': os.path.join(os.getcwd(), 'datasets', 'comic'),
        'trainC': os.path.join(os.getcwd(), 'datasets', 'comic_blurred')
    }

    '''Data'''
    trainA_files = os.listdir(datasets_pth['trainA'])
    trainB_files = os.listdir(datasets_pth['trainB'])
    trainC_files = os.listdir(datasets_pth['trainC'])
    l_A = len(trainA_files)
    l_B = len(trainB_files)
    l_C = len(trainC_files)
    l_L = max([l_A, l_B, l_C])
    for i in range(l_A):
        trainA_files[i] = os.path.join(datasets_pth['trainA'], trainA_files[i])
    for i in range(l_B):
        trainB_files[i] = os.path.join(datasets_pth['trainB'], trainB_files[i])
    for i in range(l_C):
        trainC_files[i] = os.path.join(datasets_pth['trainC'], trainC_files[i])
    print("Finish loading the datasets, and there are: <<{}>> human-faces, and <<{}>> manga-faces!".format(l_A, l_B))

    transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                    transforms.CenterCrop(IMG_SIZE)])
    '''Models'''
    G = CartoonGenerator()
    D = Discriminator()
    
    #### loading in pretrained vgg16
    
    vgg = VGG2()
    vgg.load_state_dict(torch.load('vgg16.pth'))
    for param in vgg.parameters():
        param.requires_grad = False
    """
    vgg = vgg19()
    vgg19 = models.vgg19(pretrained=True)
    pretrained_dict = vgg19.state_dict()
    vgg_dict = vgg.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in vgg_dict}
    vgg_dict.update(pretrained_dict)
    vgg.load_state_dict(vgg_dict)
    
    for param in vgg.parameters():
        param.requires_grad = False
    """
    print("Finish initializing the models, and they are: Cartoon-Generator, Cartoon-Discriminator, and VGG19")


    '''Loss functions'''
    criterionCon = ContentLoss()
    criterionAdv = AdversialLoss()# nn.MSELoss()# nn.BCELoss()# 

    '''Optimizers'''
    optim_G = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    '''Use GPU'''
    if gpu >= 0:
        G = G.to(gpu)
        D = D.to(gpu)
        vgg = vgg.to(gpu)

        criterionCon = criterionCon.to(gpu)
        criterionAdv = criterionAdv.to(gpu)


    #### pretrain the generator
    for epoch in range(50):# 100
        if os.path.exists("./checkpoints/G_pretrained.pth") is True:
            if gpu >= 0:
                G = G.cpu()
            G.load_state_dict(torch.load("./checkpoints/G_pretrained.pth"))
            if gpu >= 0:
                G = G.cuda()
            break
        '''initialization phase'''
        shuffle(trainA_files)
        shuffle(trainB_files)
        shuffle(trainC_files)
        
        for i_l in tqdm(range(0, l_L, opt.batch_size)):
            X = np.zeros((opt.batch_size, 3, IMG_SIZE, IMG_SIZE))
            for c in range(opt.batch_size):
                img = Image.open(trainA_files[(i_l * opt.batch_size+c) % l_A])
                # img.thumbnail((IMG_SIZE, IMG_SIZE))
                img = transform(img)
                X[c, :, :, :] = np.array(img).transpose(2,0,1)
            
            X = torch.from_numpy(X.astype(np.float32))
            X = 2*(X/255)-1
            # X = Variable(X)
                
            if gpu >= 0:
                X = X.to(gpu)
                
            Out = G(X)
            # print(out.data.size())
                
            F1 = vgg(Out)
            F2 = vgg(X)
            loss = criterionCon(F1, F2)
                
            optim_G.zero_grad()
            loss.backward()
            optim_G.step()
            
            log = "This is the end of the {}-th epoch, the content loss for initialization is: {}.".format(epoch, loss)
            # print(log)
            if i_l % 10 == 0:
                txt_handle.write(log+'\n')
                
        
        # save image
        print(log)
        if gpu >= 0:
            X = X.cpu()
            Out = Out.cpu()
        x = np.array(X.data[0,:,:,:]).transpose(1,2,0)
        o = np.array(Out.data[0,:,:,:]).transpose(1,2,0)
            
        img_x = Image.fromarray(((x*0.5+0.5)*255).astype(np.uint8))
        img_o = Image.fromarray(((o*0.5+0.5)*255).astype(np.uint8))
            
        img_x.save(os.path.join(os.getcwd(), check_pth, 'x-{}.jpg'.format(epoch)))
        img_o.save(os.path.join(os.getcwd(), check_pth, 'o-{}.jpg'.format(epoch)))
    ## save the models
    if gpu >= 0:
        G = G.cpu()
    torch.save(G.state_dict(), "./checkpoints/G_pretrained.pth")
    if gpu >= 0:
        G = G.cuda()


    #### pretrain the discriminator
    criterionAdv = nn.MSELoss()# nn.BCELoss()
    PRETRAIN_DISCRIMINATOR_BATCH_SIZE = 16
    opt.batch_size = PRETRAIN_DISCRIMINATOR_BATCH_SIZE
    n_A = int(PRETRAIN_DISCRIMINATOR_BATCH_SIZE / 4)
    n_B = int(PRETRAIN_DISCRIMINATOR_BATCH_SIZE / 2)
    n_C = int(PRETRAIN_DISCRIMINATOR_BATCH_SIZE / 4)
    max_iterations = max(l_A // n_A, l_B // n_B, l_C // n_C)
    
    for epoch in range(50):# 50
        if os.path.exists("./checkpoints/D_pretrained.pth") is True:
            if gpu >= 0:
                D = D.cpu()
            D.load_state_dict(torch.load("./checkpoints/D_pretrained.pth"))
            if gpu >= 0:
                D = D.cuda()
            break
        shuffle(trainA_files)
        shuffle(trainB_files)
        shuffle(trainC_files)
        
        i_A, i_B, i_C = 0, 0, 0
        ## get input images as well as corresponding labels
        for i in tqdm(range(max_iterations)):
            inputs = []
            labels = []
            n_a, n_b, n_c = 0, 0, 0
            # get real
            while n_a < n_A:
                img = Image.open(trainA_files[i_A % l_A])
                i_A += 1
                H, W = img.size[0], img.size[1]
                if H<IMG_SIZE or W<IMG_SIZE:
                    continue
                # img.thumbnail((IMG_SIZE, IMG_SIZE))
                img = transform(img)
                #print(img.size)
                arr = np.array(img).transpose(2,0,1)
                arr = 2*(arr/255) - 1.0
                arr = torch.from_numpy(arr.astype(np.float32))
                
                label = np.zeros((1, IMG_SIZE//4, IMG_SIZE//4))
                label = torch.from_numpy(label.astype(np.float32))
                
                inputs.append(arr.unsqueeze(0))
                labels.append((label).unsqueeze(0))
                n_a += 1
                
            # get comic
            while n_b < n_B:
                img = Image.open(trainB_files[i_B % l_B])
                i_B += 1
                H, W = img.size[0], img.size[1]
                if H<IMG_SIZE or W<IMG_SIZE:
                    continue
                # img.thumbnail((IMG_SIZE, IMG_SIZE))
                img = transform(img)
                #print(img.size)
                arr = np.array(img).transpose(2,0,1)
                arr = 2*(arr/255) - 1.0
                arr = torch.from_numpy(arr.astype(np.float32))
                
                label = np.ones((1, IMG_SIZE//4, IMG_SIZE//4))
                label = torch.from_numpy(label.astype(np.float32))
                
                inputs.append(arr.unsqueeze(0))
                labels.append((label).unsqueeze(0))
                n_b += 1
            
            # get comic_blurred
            while n_c < n_C:
                img = Image.open(trainC_files[i_C % l_C])
                i_C += 1
                H, W = img.size[0], img.size[1]
                if H<IMG_SIZE or W<IMG_SIZE:
                    continue
                # img.thumbnail((IMG_SIZE, IMG_SIZE))
                img = transform(img)
                #print(img.size)
                arr = np.array(img).transpose(2,0,1)
                arr = 2*(arr/255) - 1.0
                arr = torch.from_numpy(arr.astype(np.float32))
                
                label = np.zeros((1, IMG_SIZE//4, IMG_SIZE//4))
                label = torch.from_numpy(label.astype(np.float32))
                
                inputs.append(arr.unsqueeze(0))
                labels.append((label).unsqueeze(0))
                n_c += 1
            
            
            ## concatenate
            inputs = torch.cat(inputs, 0)
            labels = torch.cat(labels, 0)
            
            ## randomize the order
            randomize = np.arange(PRETRAIN_DISCRIMINATOR_BATCH_SIZE)
            np.random.shuffle(randomize)
            inputs = inputs[randomize]
            labels = labels[randomize]

            if gpu >= 0:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            optim_D.zero_grad()
            preds = D(inputs)

            preds[preds < 0.0] = 0.0
            preds[preds > 1.0] = 1.0

            # print(preds.size(), labels.size())
            loss = criterionAdv(preds, labels)#.mean()
            # print(loss)
            loss.backward()
            optim_D.step()
            
            log = ("The loss is: {}".format(loss))
            if i % 10 == 0:
                txt_handle.write(log+'\n')
            
        ## validation
        print(log)
        if gpu >= 0:
            preds = preds.cpu()
            labels = labels.cpu()
        
        acc = binary_accuracy_metric(np.array(preds.data), np.array(labels.data))
        log = ("The loss is: {}, The accuracy is: {}".format(loss, acc))
        print(log)
    
    ## save the models
    if gpu >= 0:
        D = D.cpu()
    torch.save(D.state_dict(), "./checkpoints/D_pretrained.pth")
    if gpu >= 0:
        D = D.cuda()    
    
    
    #### deversial training the entire framework
    print("Ready for the adversarial training")
    opt.batch_size = 4
    max_iterations = max(l_A // opt.batch_size, l_B // opt.batch_size, l_C // opt.batch_size)
    criterionAdv = AdversialLoss()
    
    loss_D = loss_adv = loss_con = 0.0
    for epoch in range(EPOCH):
        shuffle(trainA_files)
        shuffle(trainB_files)
        shuffle(trainC_files)
        '''training'''
        i_A = i_B = i_C = 0
        ii = 0
        for i_L in tqdm(range(0, max_iterations, opt.batch_size)):
            
            ii += 1
            
            ## inputs of G
            X = np.zeros((opt.batch_size, 3, IMG_SIZE, IMG_SIZE))
            c = 0
            
            while c < opt.batch_size:
                img = Image.open(trainA_files[i_A % l_A])
                i_A += 1
                H, W = img.size[0], img.size[1]
                if H < IMG_SIZE or W < IMG_SIZE:
                    continue
                # img.thumbnail((IMG_SIZE, IMG_SIZE))
                img = transform(img)
                X[c, :, :, :] = np.array(img).transpose(2,0,1)
                c += 1
            X = 2*(X / 255) - 1.0
            X = torch.from_numpy(X.astype(np.float32))
            if gpu >= 0:
                X = X.to(gpu)
                
            if True:# ii % 3 == 0:
                ## train D
                loss_D = 0
                optim_D.zero_grad()
                # 1A: train D on real
                # >> prepare the inouts
                real_X = np.zeros((opt.batch_size, 3, IMG_SIZE, IMG_SIZE))
                c = 0
                while c < opt.batch_size:
                    img = Image.open(trainB_files[i_B % l_B])
                    i_B += 1
                    H, W = img.size[0], img.size[1]
                    if H < IMG_SIZE or W < IMG_SIZE:
                        continue
                    # img.thumbnail((IMG_SIZE, IMG_SIZE))
                    img = transform(img)
                    real_X[c, :, :, :] = np.array(img).transpose(2,0,1)
                    c += 1
                real_X = 2*(real_X / 255) - 1.0
                real_X = torch.from_numpy(real_X.astype(np.float32))
                if gpu >= 0:
                    real_X = real_X.to(gpu)
                # >> forward
                real_decision = D(real_X)
                d_real_error = criterionAdv(real_decision, True)   # torch.ones([opt.batch_size, 3, 256, 256]))
                # d_real_error.backward()
                loss_D = d_real_error
                
                # 1B: train D on blur   
                # >> prepare the inouts 
                blur_X = np.zeros((opt.batch_size, 3, IMG_SIZE, IMG_SIZE))
                c = 0
                # for c in range(opt.batch_size):
                while c < opt.batch_size:
                    img = Image.open(trainC_files[i_C % l_C])
                    i_C += 1
                    H, W = img.size[0], img.size[1]
                    if H < IMG_SIZE or W < IMG_SIZE:
                        continue
                    # img.thumbnail((IMG_SIZE, IMG_SIZE))
                    img = transform(img)
                    blur_X[c, :, :, :] = np.array(img).transpose(2,0,1)
                    c += 1
                blur_X = 2*(blur_X / 255) - 1.0
                blur_X = torch.from_numpy(blur_X.astype(np.float32))
                if gpu >= 0:
                    blur_X = blur_X.to(gpu)
                # >> forward
                blur_decision = D(blur_X)
                d_blur_error = criterionAdv(blur_decision, False)  # torch.zeros([opt.batch_size, 3, 256, 256]))
                # d_blur_error.backward()
                loss_D += d_blur_error
                
                # 1C: train D on fake
                fake_X = G(X)
                d_fake_decision = D(fake_X)
                d_fake_error = criterionAdv(d_fake_decision, False)# torch.zeros([opt.batch_size, 3, 256, 256]))
                # d_fake_error.backward()
                loss_D += d_fake_error
                
                # 1D: update
                loss_D = loss_D / 3
                loss_D.backward(retain_graph=True)
                optim_D.step()
            
            
            ## 2: train G
            loss_G = 0
            optim_G.zero_grad()
            Out = G(X)
            F1 = vgg(Out)
            F2 = vgg(X)
            loss_con = criterionCon(F1, F2)# x10
            
            
            g_fake_decision= D(Out)
            loss_adv = criterionAdv(g_fake_decision, True)
            
            loss_G = loss_adv + loss_con * 0.5
            loss_G.backward()
            optim_G.step()
            
            if i_L % 100 == 0:
                log = "Epoch-{}, iteration-{} >> Content loss of G: {}, Adversarial loss of G: {}, Adversarial loss of D: {}".format(epoch, i_L, loss_con, loss_adv, loss_D)
                txt_handle.write(log+'\n')
                print(log)
                
        ## save image
        if gpu >= 0:
        
            X = X.cpu()
            Out = Out.cpu()

        x = (np.array(X.data[0,:,:,:])*255).astype(np.uint8).transpose(1,2,0)
        # x_r = np.array(X.data[0,:,:,:]).astype(np.uint8).transpose(1,2,0)
        # x_b = np.array(X.data[0,:,:,:]).astype(np.uint8).transpose(1,2,0)
        o = (np.array(Out.data[0,:,:,:])*255).astype(np.uint8).transpose(1,2,0)
        
        plt.subplot(121)
        plt.title("Real image")
        plt.imshow(x)
        plt.subplot(122)
        plt.title("Cartoonalized image")
        plt.imshow(o)
        plt.savefig(os.path.join(os.getcwd(), check_pth, '%04d-%06d.jpg'%(epoch, i_L)))
            
        # img_x = Image.fromarray(x)
        # img_o = Image.fromarray(o)
        # img_r = Image.fromarray(x_r)
        # img_b = Image.fromarray(x_b)
            
        # img_x.save(os.path.join(os.getcwd(), check_pth, '{}-x.jpg'.format(epoch)))
        # img_o.save(os.path.join(os.getcwd(), check_pth, '{}-o.jpg'.format(epoch)))
        # img_r.save(os.path.join(os.getcwd(), check_pth, '{}-r.jpg'.format(epoch)))
        # img_b.save(os.path.join(os.getcwd(), check_pth, '{}-b.jpg'.format(epoch)))
        
            
    txt_handle.close()

    if gpu >= 0:
        G = G.cpu()
        D = D.cpu()
    torch.save(G.state_dict(), "./checkpoints/G_adv.pth")
    torch.save(D.state_dict(), "./checkpoints/D_adv.pth")
    if gpu >= 0:
        G = G.cuda()
        D = D.cuda()
