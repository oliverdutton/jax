module @broadcast_add_fusion.2_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_add_fusion.2(%arg0: tensor<2xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<3xi32> {llvm.align = 64 : index, llvm.dereferenceable = 12 : index, xla.slice_index = 1 : index}) -> tensor<3xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %extracted = tensor.extract %arg0[%c1] : tensor<2xi32>
    %0 = scf.for %arg2 = %c0 to %c3 step %c1 iter_args(%arg3 = %arg1) -> (tensor<3xi32>) {
      %1 = arith.index_castui %arg2 : index to i64
      %2 = arith.trunci %1 : i64 to i32
      %3 = arith.addi %2, %extracted {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
      %inserted = tensor.insert %3 into %arg3[%arg2] : tensor<3xi32>
      scf.yield %inserted : tensor<3xi32>
    }
    return %0 : tensor<3xi32>
  }
}