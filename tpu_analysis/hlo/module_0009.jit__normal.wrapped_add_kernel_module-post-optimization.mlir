module @wrapped_add_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @wrapped_add(%arg0: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.slice_index = 2 : index}) -> tensor<i32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %extracted = tensor.extract %arg0[] : tensor<i32>
    %extracted_0 = tensor.extract %arg1[] : tensor<i32>
    %0 = arith.addi %extracted, %extracted_0 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %inserted = tensor.insert %0 into %arg2[] : tensor<i32>
    return %inserted : tensor<i32>
  }
}