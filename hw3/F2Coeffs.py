#
# *VERY IMPORTANT* Only use for problem3 of MLSP hw3
#
# This script is provided for converting the sparse vector F (used in Matlab)
# to the coefficient list (used in Python library), so that you can use pywt.waverec2
# to recover the image
#
# Written by the TAs of MLSP 2019.
# Date: Oct. 24, 2019

import pywt
import numpy as np
from PIL import Image

def F2Coeffs(F):
    # Coeffs is the Coefficients list [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
    # For this problem, n is 7
    # cHn, cVn, and cDn are 2D matrices with the size of 2^(7-n) x 2^(7-n)
    # In Matlab, the Coeffs is flatten as a vector.
    # Now we reform it to a coefficient list

    # Input: F, a vector with size of 16384,
    # Output: Coeffs, see https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html#d-multilevel-decomposition-using-wavedec2 for explanation

    Coeffs = []
    # cAn
    Coeffs.append(np.reshape(F[0], (1, 1)))
    # (cHn, cVn, cDn)
    pt = 1
    for n in range(7, 0, -1):
        dim = 2 ** (7 - n)
        coeff = np.reshape(F[pt:pt + 3 * dim * dim], (3, dim, dim))
        coeff = np.transpose(coeff, (0, 2, 1))
        cHn, cVn, cDn = list(coeff)
        Coeffs.append((cHn, cVn, cDn))
        pt += 3 * dim * dim
    return Coeffs 

# example
if __name__ == "__main__":
   with open('F.csv', 'r') as f:
       lines = f.readlines()
   lines = [float(line.rstrip()) for line in lines]
   F = np.array(lines)
   Coeffs = F2Coeffs(F)
   
   image = pywt.waverec2(Coeffs, 'db1')
   im = Image.fromarray(image.astype('uint8'))
   im.save('example.png')

