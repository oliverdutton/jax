module @broadcast_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_multiply_fusion(%arg0: tensor<4096xf32> {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<f32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<4096xf32> {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, xla.slice_index = 2 : index}) -> tensor<4096xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %extracted = tensor.extract %arg1[] : tensor<f32>
    %0 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %arg2) -> (tensor<4096xf32>) {
      %1 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4096xf32>) {
        %2 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 64 + d1), domain: d0 in [0, 63], d1 in [0, 63]">(%arg3, %arg5)
        %extracted_0 = tensor.extract %arg0[%2] : tensor<4096xf32>
        %3 = arith.mulf %extracted_0, %extracted : f32
        %inserted = tensor.insert %3 into %arg6[%2] : tensor<4096xf32>
        scf.yield %inserted : tensor<4096xf32>
      }
      scf.yield %1 : tensor<4096xf32>
    } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    return %0 : tensor<4096xf32>
  }
}