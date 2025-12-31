module @broadcast_add_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_add_fusion(%arg0: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<4xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<4096xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, xla.invariant, xla.slice_index = 2 : index}, %arg3: tensor<4096xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, xla.invariant, xla.slice_index = 3 : index}, %arg4: tensor<i32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 4 : index}, %arg5: tensor<4096xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, xla.slice_index = 5 : index}) -> tensor<4096xi32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %c13 = arith.constant 13 : index
    %c64 = arith.constant 64 : index
    %c12 = arith.constant 12 : index
    %c4 = arith.constant 4 : index
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 4 : index]}
    %1 = arith.cmpi sle, %0, %c3 : index
    %2 = scf.if %1 -> (tensor<4096xi32>) {
      %3 = arith.cmpi sge, %0, %c0 : index
      %4 = arith.andi %3, %1 : i1
      %5 = scf.if %4 -> (tensor<4096xi32>) {
        %extracted = tensor.extract %arg1[%c0] : tensor<4xi32>
        %6 = arith.subi %c32_i32, %extracted : i32
        %7 = arith.cmpi ult, %extracted, %c32_i32 : i32
        %8 = arith.cmpi ult, %6, %c32_i32 : i32
        %extracted_0 = tensor.extract %arg1[%c1] : tensor<4xi32>
        %9 = arith.subi %c32_i32, %extracted_0 : i32
        %10 = arith.cmpi ult, %extracted_0, %c32_i32 : i32
        %11 = arith.cmpi ult, %9, %c32_i32 : i32
        %extracted_1 = tensor.extract %arg1[%c2] : tensor<4xi32>
        %12 = arith.subi %c32_i32, %extracted_1 : i32
        %13 = arith.cmpi ult, %extracted_1, %c32_i32 : i32
        %14 = arith.cmpi ult, %12, %c32_i32 : i32
        %extracted_2 = tensor.extract %arg1[%c3] : tensor<4xi32>
        %15 = arith.subi %c32_i32, %extracted_2 : i32
        %16 = arith.cmpi ult, %extracted_2, %c32_i32 : i32
        %17 = arith.cmpi ult, %15, %c32_i32 : i32
        %extracted_3 = tensor.extract %arg0[] : tensor<i32>
        %extracted_4 = tensor.extract %arg4[] : tensor<i32>
        %18 = arith.addi %extracted_4, %c1_i32 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %19 = scf.for %arg6 = %c0 to %c13 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4096xi32>) {
          %20 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %arg7) -> (tensor<4096xi32>) {
            %21 = xla.apply_indexing #xla.indexing_map<"(d0, bl_x, d2) -> (bl_x * 832 + d2 * 64 + d0), domain: d0 in [0, 63], bl_x in [0, 3], d2 in [0, 12]">(%arg8, %0, %arg6)
            %extracted_5 = tensor.extract %arg2[%21] : tensor<4096xi32>
            %22 = arith.shli %extracted_5, %extracted : i32
            %23 = arith.select %7, %22, %c0_i32 : i32
            %24 = arith.shrui %extracted_5, %6 : i32
            %25 = arith.select %8, %24, %c0_i32 : i32
            %extracted_6 = tensor.extract %arg3[%21] : tensor<4096xi32>
            %26 = arith.ori %23, %25 : i32
            %27 = arith.addi %extracted_6, %extracted_5 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %28 = arith.xori %27, %26 : i32
            %29 = arith.shli %28, %extracted_0 : i32
            %30 = arith.select %10, %29, %c0_i32 : i32
            %31 = arith.shrui %28, %9 : i32
            %32 = arith.select %11, %31, %c0_i32 : i32
            %33 = arith.ori %30, %32 : i32
            %34 = arith.addi %27, %28 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %35 = arith.xori %34, %33 : i32
            %36 = arith.shli %35, %extracted_1 : i32
            %37 = arith.select %13, %36, %c0_i32 : i32
            %38 = arith.shrui %35, %12 : i32
            %39 = arith.select %14, %38, %c0_i32 : i32
            %40 = arith.ori %37, %39 : i32
            %41 = arith.addi %34, %35 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %42 = arith.xori %41, %40 : i32
            %43 = arith.shli %42, %extracted_2 : i32
            %44 = arith.select %16, %43, %c0_i32 : i32
            %45 = arith.shrui %42, %15 : i32
            %46 = arith.select %17, %45, %c0_i32 : i32
            %47 = arith.addi %41, %42 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %48 = arith.ori %44, %46 : i32
            %49 = arith.xori %47, %48 : i32
            %50 = arith.addi %49, %extracted_3 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %51 = arith.addi %50, %18 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %inserted = tensor.insert %51 into %arg9[%21] : tensor<4096xi32>
            scf.yield %inserted : tensor<4096xi32>
          }
          scf.yield %20 : tensor<4096xi32>
        } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
        scf.yield %19 : tensor<4096xi32>
      } else {
        scf.yield %arg5 : tensor<4096xi32>
      }
      scf.yield %5 : tensor<4096xi32>
    } else {
      %3 = arith.cmpi eq, %0, %c4 : index
      %4 = scf.if %3 -> (tensor<4096xi32>) {
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
        %extracted_2 = tensor.extract %arg1[%c3] : tensor<4xi32>
        %14 = arith.subi %c32_i32, %extracted_2 : i32
        %15 = arith.cmpi ult, %extracted_2, %c32_i32 : i32
        %16 = arith.cmpi ult, %14, %c32_i32 : i32
        %extracted_3 = tensor.extract %arg0[] : tensor<i32>
        %extracted_4 = tensor.extract %arg4[] : tensor<i32>
        %17 = arith.addi %extracted_4, %c1_i32 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
        %18 = scf.for %arg6 = %c0 to %c12 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4096xi32>) {
          %19 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %arg7) -> (tensor<4096xi32>) {
            %20 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d1 * 64 + d0 + 3328), domain: d0 in [0, 63], d1 in [0, 11]">(%arg8, %arg6)
            %extracted_5 = tensor.extract %arg2[%20] : tensor<4096xi32>
            %21 = arith.shli %extracted_5, %extracted : i32
            %22 = arith.select %6, %21, %c0_i32 : i32
            %23 = arith.shrui %extracted_5, %5 : i32
            %24 = arith.select %7, %23, %c0_i32 : i32
            %extracted_6 = tensor.extract %arg3[%20] : tensor<4096xi32>
            %25 = arith.ori %22, %24 : i32
            %26 = arith.addi %extracted_6, %extracted_5 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %27 = arith.xori %26, %25 : i32
            %28 = arith.shli %27, %extracted_0 : i32
            %29 = arith.select %9, %28, %c0_i32 : i32
            %30 = arith.shrui %27, %8 : i32
            %31 = arith.select %10, %30, %c0_i32 : i32
            %32 = arith.ori %29, %31 : i32
            %33 = arith.addi %26, %27 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %34 = arith.xori %33, %32 : i32
            %35 = arith.shli %34, %extracted_1 : i32
            %36 = arith.select %12, %35, %c0_i32 : i32
            %37 = arith.shrui %34, %11 : i32
            %38 = arith.select %13, %37, %c0_i32 : i32
            %39 = arith.ori %36, %38 : i32
            %40 = arith.addi %33, %34 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %41 = arith.xori %40, %39 : i32
            %42 = arith.shli %41, %extracted_2 : i32
            %43 = arith.select %15, %42, %c0_i32 : i32
            %44 = arith.shrui %41, %14 : i32
            %45 = arith.select %16, %44, %c0_i32 : i32
            %46 = arith.addi %40, %41 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %47 = arith.ori %43, %45 : i32
            %48 = arith.xori %46, %47 : i32
            %49 = arith.addi %48, %extracted_3 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %50 = arith.addi %49, %17 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
            %inserted = tensor.insert %50 into %arg9[%20] : tensor<4096xi32>
            scf.yield %inserted : tensor<4096xi32>
          }
          scf.yield %19 : tensor<4096xi32>
        } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
        scf.yield %18 : tensor<4096xi32>
      } else {
        scf.yield %arg5 : tensor<4096xi32>
      }
      scf.yield %4 : tensor<4096xi32>
    }
    return %2 : tensor<4096xi32>
  }
}