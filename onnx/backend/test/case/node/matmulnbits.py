# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect

def matmulnbits_reference_implementation(

) -> np.ndarray:
  #TODO create the matmulnbits reference implementation
  return np.array(0)

class MatMulNBits(Base):
  @staticmethod
  def export_only_required() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits", 
                                 inputs = ['a', 'b', 'scale'],
                                 outputs = ['y'],
                                 K = 5,
                                 N = 4)
    a = np.random.ranf([3,5]).astype(np.float32)
    b = np.random.ranf([5,4]).astype(np.int4) #this should actually be int8 data type
    scale = np.zeros([1,4]).astype(np.float)
    y = matmulnbits_reference_implementation(a, b, scale)
    expect(node, inputs=[a, b, scale], outputs[y])

# TODO add a test for at least each input configuration and adjusted attribute

