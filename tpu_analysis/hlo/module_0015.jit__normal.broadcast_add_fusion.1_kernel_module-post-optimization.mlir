module @broadcast_add_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_add_fusion.1(%arg0: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<4xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<2048xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, xla.invariant, xla.slice_index = 2 : index}, %arg3: tensor<2048xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, xla.invariant, xla.slice_index = 3 : index}, %arg4: tensor<2048xi32> {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, xla.slice_index = 4 : index}) -> tensor<2048xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 1 : index]}
    %1 = arith.cmpi sge, %0, %c0 : index
    %2 = arith.cmpi sle, %0, %c1 : index
    %3 = arith.andi %1, %2 : i1
    %4 = scf.if %3 -> (tensor<2048xi32>) {
      %extracted = tensor.extract %arg1[%c0] : tensor<4xi32>
      %5 = arith.subi %c32_i32, %extracted : i32
      %6 = arith.cmpi ult, %extracted, %c32_i32 : i32
      %7 = arith.cmpi ult, %5, %c32_i32 : i32
      %extracted_0 = tensor.extract %arg1[%c1] : tensor<4xi32>
      %8 = arith.subi %c32_i32, %extracted_0 : i32
      %9 = arith.cmpi ult, %extracted_0, %c32_i32 : i32
      %10 = arith.cmpi ult, %8, %c32_i32 : i32
      %extracted_1 = tensor.extract %arg1[%c2] : tensor<4xi32>
      %11 = arith.subi %c32_i32, %extracted_1 : i32
      %12 = arith.cmpi ult, %extracted_1, %c32_i32 : i32
      %13 = arith.cmpi ult, %11, %c32_i32 : i32
      %extracted_2 = tensor.extract %arg0[] : tensor<i32>
      %14 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %arg4) -> (tensor<2048xi32>) {
        %15 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %arg6) -> (tensor<2048xi32>) {
          %16 = xla.apply_indexing #xla.indexing_map<"(d0, bl_x, d2) -> (bl_x * 1024 + d2 * 32 + d0), domain: d0 in [0, 31], bl_x in [0, 1], d2 in [0, 31]">(%arg7, %0, %arg5)
          %extracted_3 = tensor.extract %arg2[%16] : tensor<2048xi32>
          %17 = arith.shli %extracted_3, %extracted : i32
          %18 = arith.select %6, %17, %c0_i32 : i32
          %19 = arith.shrui %extracted_3, %5 : i32
          %20 = arith.select %7, %19, %c0_i32 : i32
          %extracted_4 = tensor.extract %arg3[%16] : tensor<2048xi32>
          %21 = arith.ori %18, %20 : i32
          %22 = arith.addi %extracted_4, %extracted_3 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
          %23 = arith.xori %22, %21 : i32
          %24 = arith.shli %23, %extracted_0 : i32
          %25 = arith.select %9, %24, %c0_i32 : i32
          %26 = arith.shrui %23, %8 : i32
          %27 = arith.select %10, %26, %c0_i32 : i32
          %28 = arith.ori %25, %27 : i32
          %29 = arith.addi %22, %23 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
          %30 = arith.xori %29, %28 : i32
          %31 = arith.shli %30, %extracted_1 : i32
          %32 = arith.select %12, %31, %c0_i32 : i32
          %33 = arith.shrui %30, %11 : i32
          %34 = arith.select %13, %33, %c0_i32 : i32
          %35 = arith.ori %32, %34 : i32
          %36 = arith.addi %29, %30 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
          %37 = arith.xori %36, %35 : i32
          %38 = arith.addi %36, %37 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
          %39 = arith.addi %38, %extracted_2 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
          %inserted = tensor.insert %39 into %arg8[%16] : tensor<2048xi32>
          scf.yield %inserted : tensor<2048xi32>
        }
        scf.yield %15 : tensor<2048xi32>
      } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
      scf.yield %14 : tensor<2048xi32>
    } else {
      scf.yield %arg4 : tensor<2048xi32>
    }
    return %4 : tensor<2048xi32>
  }
}