module @bitcast_concatenate_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__concatenate_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @bitcast_concatenate_fusion(%arg0: tensor<3xi32> {llvm.align = 64 : index, llvm.dereferenceable = 12 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<3xi32> {llvm.align = 64 : index, llvm.dereferenceable = 12 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<6xi32> {llvm.align = 64 : index, llvm.dereferenceable = 24 : index, xla.slice_index = 2 : index}) -> tensor<6xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %arg2) -> (tensor<6xi32>) {
      %extracted = tensor.extract %arg1[%arg3] : tensor<3xi32>
      %2 = xla.apply_indexing #xla.indexing_map<"(d0) -> (d0 * 2), domain: d0 in [0, 2]">(%arg3)
      %inserted = tensor.insert %extracted into %arg4[%2] : tensor<6xi32>
      scf.yield %inserted : tensor<6xi32>
    }
    %1 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %0) -> (tensor<6xi32>) {
      %extracted = tensor.extract %arg0[%arg3] : tensor<3xi32>
      %2 = xla.apply_indexing #xla.indexing_map<"(d0) -> (d0 * 2 + 1), domain: d0 in [0, 2]">(%arg3)
      %inserted = tensor.insert %extracted into %arg4[%2] : tensor<6xi32>
      scf.yield %inserted : tensor<6xi32>
    }
    return %1 : tensor<6xi32>
  }
}