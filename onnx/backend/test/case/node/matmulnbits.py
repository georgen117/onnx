# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import math

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect

def DequantizeLinearNBits(
    X: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray | None = None,
    **kwargs
) -> np.ndarray:
    if 'N' in kwargs:
        N = kwargs['N']
    else:
        print('Error: "N" attribute is required')
        return
    if 'K' in kwargs:
        K = kwargs['K']
    else:
        print('Error: "K" attribute is required')
        return
    bits = kwargs.get('bits', 4)
    block_size = kwargs.get('block_size', 128)

    zero_points = zero_points if zero_points is not None else np.full(scales.shape, (2 ** (bits - 1))).astype(A.dtype)

    # TODO(george) check if B and zero_points has the required number of bytes based on the bits and block_size
    # Input B is stored as uint8_t with shape: `[N][n_blocks_per_col][blob_size]`
    # in which:
    #    - `n_blocks_per_col` = `(K + block_size - 1) / block_size`
    #    - `blob_size` = `CeilDiv(block_size * bits, bitsof(uint8_t)<8>)`
    # Input zero_points is stored as uint8_t or same as type(A). It has the same packing method as input B.
    #    - [CeilDiv((N * n_blocks_per_col + 1) *bits, 8)]
    Y = np.empty((N,K), dtype=scales.dtype) # create empty array
    n_blocks_per_col = (K + block_size - 1) // block_size
    blob_size = math.ceil((block_size * bits)/8)
    mask = (1 << bits) - 1 # create mask of bit_size (e.g. 0b1111)
    if (zero_points.dtype != scales.dtype):
        zp_size = N * n_blocks_per_col
        unpacked_zp = []
        current_bit_pos = 0
        total_zp_bits = len(zero_points) * 8
        while current_bit_pos < total_zp_bits and len(unpacked_zp) < zp_size:
            byte_pos = (current_bit_pos // 8)
            bit_offset = current_bit_pos % 8

            bits_available = 8 - bit_offset
            if (bits_available >= bits):
                # all bit are in current byte
                value = (zero_points[byte_pos] >> (bits_available - bits)) & mask
            else:
                #Bits are split accoss two bytes
                upper_bits = zero_points[byte_pos] << bits - bits_available & mask
                lower_bits = zero_points[byte_pos + 1] >> 8 - (bits - bits_available) & mask
                value = upper_bits | lower_bits
            unpacked_zp.append(value)
            current_bit_pos += bits
        # replace zero_points with unpacked zero_points
        zero_points = np.array(unpacked_zp).astype(scales.dtype)

    # total bits is actually the number of bits per row
    total_bits = n_blocks_per_col * blob_size * 8
    for n in range(N):
        unpacked_row_buf = []
        current_bit_pos = 0
        for n_bpc in range(n_blocks_per_col):
            unpacked_col_buf = []
            # Unpack the packed bits blob at a time or until the all K elements have been unpacked
            # this will result in block_size chunks of data the zero_points and scales can be applied to
            while len(unpacked_col_buf) < block_size and (len(unpacked_row_buf) + len(unpacked_col_buf)) < K and current_bit_pos < total_bits:
                byte_pos = (current_bit_pos // 8)
                bit_offset = current_bit_pos % 8

                bits_available = 8 - bit_offset
                if (bits_available >= bits):
                    # all bit are in current byte
                    value = (X[n][byte_pos] >> (bits_available - bits)) & mask
                else:
                    #Bits are split accoss two bytes
                    upper_bits = X[n][byte_pos] << bits - bits_available & mask
                    lower_bits = X[n][byte_pos + 1] >> 8 - (bits - bits_available) & mask
                    value = upper_bits | lower_bits
                unpacked_col_buf.append(np.uint8(value).astype(scales.dtype))
                current_bit_pos += bits
            # Apply the dequantize linear algorithm to the block_size chunk of data
            unpacked_row_buf.extend((unpacked_col_buf - zero_points[n * n_blocks_per_col + n_bpc]) * scales[n * n_blocks_per_col + n_bpc])
        Y[n] = unpacked_row_buf
    return Y

def matmulnbits_reference_implementation(
    #inputs
    A: np.ndarray,
    B: np.ndarray,
    scales: np.ndarray,
    # optional inputs
    zero_points: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    **kwargs
) -> np.ndarray:
    # read in attributes
    if 'N' in kwargs:
        N = kwargs['N']
    else:
        print('Error: "N" attribute is required')
        return
    if 'K' in kwargs:
        K = kwargs['K']
    else:
        print('Error: "K" attribute is required')
        return
    accuracy_level = kwargs.get('accuracy_level', 0)
    bits = kwargs.get('bits', 4)
    block_size = kwargs.get('block_size', 128)
    # set defaults for optional inputs
    zero_points = zero_points if zero_points is not None else np.full(scales.shape, (2 ** (bits - 1))).astype(A.dtype)
    bias = bias if bias is not None else np.array(0).astype(A.dtype)
    # TODO(george) do we need to check if `B` and `zero_points`` has the required number of bytes based on
    # the bits and block_size.
    # Input B is stored as uint8_t with shape: `[N][n_blocks_per_col * blob_size]`
    # in which:
    #    - `n_blocks_per_col` = `(K + block_size - 1) / block_size`
    #    - `blob_size` = `CeilDiv(block_size * bits, bitsof(uint8_t)<8>)`
    # Input zero_points is stored as uint8_t or same as type(A). It has the same packing method as input B.
    #    - [`CeilDiv((N * n_blocks_per_col + 1) *bits, 8)``]
    dq_B = DequantizeLinearNBits(B, scales, zero_points, K = K, N = N, bits = bits, block_size = block_size)
    # NOTE: the output from DequantizeLinearNBits is {N,K} so it will need to be transposed for MatMul bellow
    # accuracy_level defaults to 0 which means the type used for matmul type matches A.dtype
    matmul_type = A.dtype
    # accuracy_level 1 == float32
    if( accuracy_level == 1):
        matmul_type = np.float32
    # accuracy_level 2 == float16
    if( accuracy_level == 2):
        matmul_type = np.float16
    # accuracy_level 3 == bfloat16
    # TODO numpy dose not support bfloat16 using float32 for now
    if( accuracy_level == 3):
        matmul_type = np.float32
    # accuracy_level 4 == int8
    if( accuracy_level == 4):
        matmul_type = np.int8
    # base case is assumed to be accuracy_level == 0
    c = np.matmul(A.astype(matmul_type), np.transpose(dq_B.astype(matmul_type)))
    Y = c.astype(A.dtype) + bias
    return Y

# TODO(george) is a pack function need to pack or quantize the numbers into NBits for testing?

class MatMulNBits(Base):
  @staticmethod
  def export_matmulnbits_required_inputs_only() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales'],
                                 outputs = ['y'],
                                 K = 4,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16)
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8).reshape((3,8))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, K=4, N=3, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales], outputs=[y], name="test_matmulnbits_required_inputs_only")

  @staticmethod
  def export_matmulnbits_with_zero_points_f32() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', 'zero_points'],
                                 outputs = ['y'],
                                 K = 4,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16)
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8).reshape((3,8))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    zero_points = np.array([7.0, 7.0, 7.0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points, K=4, N=3, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales, zero_points], outputs=[y], name="test_matmulnbits_with_zero_points_f32")

  @staticmethod
  def export_matmulnbits_with_zero_points_u8() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', 'zero_points'],
                                 outputs = ['y'],
                                 K = 4,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16)
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8).reshape((3,8))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    zero_points = np.array([0xff, 0xf0], dtype=np.uint8)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points, K=4, N=3, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales, zero_points], outputs=[y], name="test_matmulnbits_with_zero_points_u8")
# TODO(george) add a test for at least each input configuration and adjusted attribute
