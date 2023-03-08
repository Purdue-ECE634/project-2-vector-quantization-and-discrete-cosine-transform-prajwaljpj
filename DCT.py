#!/usr/bin/env python3

import numpy as np
import cv2
import argparse
from scipy.fftpack import dct, idct
from math import log10, sqrt
import time


#  PSNR helper function
def PSNR(orig, comp):
    mse = np.mean((orig - comp) ** 2)
    psnr = 20 * log10(255.0 / sqrt(mse))
    return psnr


class DiscreteCosineTransform:
    def __init__(self, input_img, coef):
        self.input_img = input_img
        self.coef = coef

        # dct and idct from scipy
        self.DCT = lambda block: dct(
            dct(block, axis=0, norm="ortho"), axis=1, norm="ortho"
        )
        self.IDCT = lambda block: idct(
            idct(block, axis=0, norm="ortho"), axis=1, norm="ortho"
        )

    def calc_dct(self):
        """Calculate the DCT over the given image and coefficients"""
        out_img = np.zeros_like(self.input_img)
        H, W = img.shape

        for i in range(H // 8):
            for j in range(W // 8):
                block = img[8 * i : 8 * (i + 1), 8 * j : 8 * (j + 1)]
                block_dct = self.DCT(block)
                block_truncated = self.zigzag(block_dct)
                out_img[8 * i : 8 * (i + 1), 8 * j : 8 * (j + 1)] = self.IDCT(
                    block_truncated
                )

        return out_img

    def zigzag(self, block):
        """reconstruct block in zigzag order"""
        ret_block = np.zeros_like(block)
        H, W = block.shape
        # directions: right(r), down (d), up right(ur), down left(dl). Let us start with right
        direction = "r"
        i = 0
        j = 0
        for k in range(self.coef):
            ret_block[i, j] = block[i, j]
            if direction == "r":
                if k == 0:
                    j += 1
                    direction = "dl"
                else:
                    i += 1
                    j -= 1
                    direction = "dl"
            elif direction == "dl":
                if j == 0:
                    direction = "d"
                    i += 1
                else:
                    direction = "dl"
                    i += 1
                    j -= 1
            elif direction == "ur":
                if i == 0:
                    direction = "r"
                    j += 1
                elif j == W - 1:
                    direction = "d"
                    i += 1
                else:
                    direction = "ur"
                    j += 1
                    i -= 1
            elif direction == "d":
                if i == H - 1:
                    direction = "r"
                    j += 1
                elif j == W - 1:
                    direction = "dl"
                    j -= 1
                    i += 1
                else:
                    direction = "ur"
                    j += 1
                    i -= 1

        return ret_block


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="image path")
    parser.add_argument(
        "--coef", "-k", required=True, type=int, help="number of coefficients to use"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="output file name"
    )
    args = parser.parse_args()

    start = time.time()
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    dct_class = DiscreteCosineTransform(img, args.coef)
    out = dct_class.calc_dct()

    latency = time.time() - start
    cv2.imwrite(args.output, out)
    psnr = PSNR(img, out)
    print("PSNR: {} dB \t Latency: {} ms".format(psnr, latency * 1000))
