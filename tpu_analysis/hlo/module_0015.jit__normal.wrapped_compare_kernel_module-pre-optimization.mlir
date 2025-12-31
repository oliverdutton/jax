module @wrapped_compare_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @wrapped_compare(%arg0: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<i8> {llvm.align = 64 : index, llvm.dereferenceable = 1 : index, xla.slice_index = 2 : index}) -> tensor<i8> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg3, %arg4, %arg5) in (1, 1, 1) shared_outs(%arg6 = %arg2) -> (tensor<i8>) {
      %xla_loop = xla.loop (%arg3, %arg4, %arg5, %0, %1, %2)[] -> () in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z) -> (), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0]"> iter_args(%iter = %arg6) -> (tensor<i8>) {
        %pure_call = xla.pure_call @wrapped_compare_computation_lt_3(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> i8
        %inserted = tensor.insert %pure_call into %iter[] : tensor<i8>
        xla.yield %inserted : tensor<i8>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg6[] [] [] : tensor<i8> into tensor<i8>
      }
    }
    return %3 : tensor<i8>
  }
  func.func private @wrapped_compare_computation_lt_3(%arg0: tensor<i32>, %arg1: tensor<i32>) -> i8 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[] : tensor<i32>
    %extracted_0 = tensor.extract %arg1[] : tensor<i32>
    %0 = arith.cmpi slt, %extracted, %extracted_0 : i32
    %1 = arith.extui %0 : i1 to i8
    return %1 : i8
  }
}