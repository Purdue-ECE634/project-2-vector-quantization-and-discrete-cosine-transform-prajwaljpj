#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import cv2
import argparse
from math import log10, sqrt
import time

# import os

# PSNR helper
def PSNR(orig, comp):
    mse = np.mean((orig - comp) ** 2)
    psnr = 20 * log10(255.0 / sqrt(mse))
    return psnr


class VectorQuantization:
    def __init__(self, input_img, level_val, train_path=""):
        self.input_img = input_img
        self.level_val = level_val
        self.train_path = train_path

    # compute the MSE
    def MSE(self, a, b):
        return np.mean((a - b) ** 2)

    # read the training data and reshape it for training
    def setup_training(self):
        imgs = []  # image list
        if self.train_path:
            imgs = Path(self.train_path).glob("*")
        else:
            imgs.append(Path(self.input_img))

        data = []
        for img in imgs:
            image = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            H, W = image.shape
            num_blocks = H // 4 * W // 4
            # accumulate all 4x4 blocks
            data_arr = np.zeros((num_blocks, 4, 4))
            idx = 0
            for i in range(H // 4):
                for j in range(W // 4):
                    data_arr[idx] = image[4 * i : 4 * (i + 1), 4 * j : 4 * (j + 1)]
                    idx += 1
            data.append(data_arr)

        data_out = data[0]
        for i in range(1, len(data)):
            data_out = np.vstack((data_out, data[i]))

        # print(len(data), data[0].shape)
        # print(data_out.shape)

        return data_out

    # train the codebook using training data
    def train(self):
        # print("Data Prep...")
        data = self.setup_training()
        # print("Done!")

        # print("Started training...")
        L = int(self.level_val)
        block_n = data.shape[0]
        # print(data.shape)
        # initialize codebook
        codebook = np.arange(0, 256, 256 // L).reshape((L, 1))
        # print(codebook.shape)
        codebook = np.tile(codebook, 16).reshape((L, 4, 4))
        # print(codebook.shape)

        idx = 0
        # train max 100 iterations if change in distortion is too low
        while idx < 100:
            code_vec = np.zeros((block_n))
            err_vec = np.zeros((block_n))
            for i in range(block_n):
                mse_best = np.inf
                # compare data with codebook
                for j in range(L):
                    mse = self.MSE(data[i], codebook[j])
                    if mse < mse_best:
                        mse_best = mse
                        code = j
                # update code vector and error
                code_vec[i] = code
                err_vec[i] = mse_best
            dist = np.mean(err_vec)

            # create codebook
            if idx == 0 or np.abs(dist - dist_) / dist_ >= 0.01:
                print("Iteration: {} \t Distortion: {} ".format(idx, dist))
                for m in range(L):
                    if np.sum(code_vec == m) == 0:
                        codebook[m] = np.zeros((4, 4))
                    else:
                        codebook[m] = np.mean(data[code_vec == m], axis=0)
                dist_ = dist
                idx += 1
            else:
                break

        # print("Done!")
        return codebook

    # compress the image codebook
    def vec_quant(self, codebook, img):
        # print("Compressing...")
        H, W = img.shape
        out_img = np.zeros_like(img)

        for i in range(H // 4):
            for j in range(W // 4):
                block = img[4 * i : 4 * (i + 1), 4 * j : 4 * (j + 1)]
                mse_max = np.inf
                for k in range(codebook.shape[0]):
                    mse = self.MSE(block, codebook[k])
                    if mse < mse_max:
                        mse_max = mse
                        L = k
                out_img[4 * i : 4 * (i + 1), 4 * j : 4 * (j + 1)] = codebook[L]
        # print("Done!")
        return out_img


if __name__ == "__main__":
    # input from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        "-t",
        type=str,
        help="training path if training is required",
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="input image path"
    )
    parser.add_argument(
        "--level", "-l", type=int, required=True, help="Quantization (128 or 256)"
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="Output file")
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

    start = time.time()
    if args.train:
        train = args.train
    else:
        train = ""
    vq = VectorQuantization(args.input, args.level, train)

    codebook = vq.train()
    # print(codebook.shape)
    quant_img = vq.vec_quant(codebook, img)

    latency = time.time() - start
    cv2.imwrite(args.output, quant_img)

    psnr = PSNR(img, quant_img)
    print("PSNR: {}dB \t Latency: {} ms".format(psnr, latency * 1000))
