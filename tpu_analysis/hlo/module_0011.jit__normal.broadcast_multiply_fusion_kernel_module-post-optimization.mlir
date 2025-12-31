module @broadcast_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @broadcast_multiply_fusion(%arg0: tensor<4096xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<4096xi32> {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<4096xf32> {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, xla.slice_index = 2 : index}) -> tensor<4096xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %cst = arith.constant 1.41421354 : f32
    %cst_0 = arith.constant 0x7F800000 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 2.83297682 : f32
    %cst_3 = arith.constant 1.50140941 : f32
    %cst_4 = arith.constant 1.00167406 : f32
    %cst_5 = arith.constant 0.246640727 : f32
    %cst_6 = arith.constant 0.00943887047 : f32
    %cst_7 = arith.constant -0.00417768164 : f32
    %cst_8 = arith.constant -0.0076224613 : f32
    %cst_9 = arith.constant -0.00125372503 : f32
    %cst_10 = arith.constant 0.00573950773 : f32
    %cst_11 = arith.constant 2.1858087E-4 : f32
    %cst_12 = arith.constant -0.00367342844 : f32
    %cst_13 = arith.constant -4.39150654E-6 : f32
    %cst_14 = arith.constant 0.00134934322 : f32
    %cst_15 = arith.constant -3.5233877E-6 : f32
    %cst_16 = arith.constant -3.000000e+00 : f32
    %cst_17 = arith.constant -2.500000e+00 : f32
    %cst_18 = arith.constant 5.000000e+00 : f32
    %cst_19 = arith.constant -0.99999994 : f32
    %cst_20 = arith.constant 2.000000e+00 : f32
    %cst_21 = arith.constant -1.000000e+00 : f32
    %c1065353216_i32 = arith.constant 1065353216 : i32
    %c9_i32 = arith.constant 9 : i32
    %cst_22 = arith.constant 2.81022636E-8 : f32
    %cst_23 = arith.constant -2.00214257E-4 : f32
    %cst_24 = arith.constant 3.43273939E-7 : f32
    %cst_25 = arith.constant 1.00950558E-4 : f32
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c11 = arith.constant 11 : index
    %c64 = arith.constant 64 : index
    %c9 = arith.constant 9 : index
    %c5 = arith.constant 5 : index
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 5 : index]}
    %1 = arith.cmpi sle, %0, %c4 : index
    %2 = scf.if %1 -> (tensor<4096xf32>) {
      %3 = arith.cmpi sge, %0, %c0 : index
      %4 = arith.andi %3, %1 : i1
      %5 = scf.if %4 -> (tensor<4096xf32>) {
        %6 = scf.for %arg3 = %c0 to %c11 step %c1 iter_args(%arg4 = %arg2) -> (tensor<4096xf32>) {
          %7 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4096xf32>) {
            %8 = xla.apply_indexing #xla.indexing_map<"(d0, bl_x, d2) -> (bl_x * 704 + d2 * 64 + d0), domain: d0 in [0, 63], bl_x in [0, 4], d2 in [0, 10]">(%arg5, %0, %arg3)
            %extracted = tensor.extract %arg0[%8] : tensor<4096xi32>
            %extracted_26 = tensor.extract %arg1[%8] : tensor<4096xi32>
            %9 = arith.xori %extracted, %extracted_26 : i32
            %10 = arith.shrui %9, %c9_i32 : i32
            %11 = arith.ori %10, %c1065353216_i32 : i32
            %12 = arith.bitcast %11 : i32 to f32
            %13 = arith.addf %12, %cst_21 : f32
            %14 = arith.mulf %13, %cst_20 : f32
            %15 = arith.addf %14, %cst_19 : f32
            %16 = arith.maximumf %15, %cst_19 : f32
            %17 = arith.negf %16 : f32
            %18 = arith.mulf %16, %17 : f32
            %19 = math.log1p %18 : f32
            %20 = arith.negf %19 : f32
            %21 = arith.cmpf olt, %20, %cst_18 : f32
            %22 = arith.select %21, %cst_22, %cst_23 : f32
            %23 = arith.select %21, %cst_24, %cst_25 : f32
            %24 = math.sqrt %20 : f32
            %25 = arith.addf %20, %cst_17 : f32
            %26 = arith.addf %24, %cst_16 : f32
            %27 = arith.select %21, %25, %26 : f32
            %28 = arith.mulf %22, %27 : f32
            %29 = arith.addf %23, %28 : f32
            %30 = arith.select %21, %cst_15, %cst_14 : f32
            %31 = arith.mulf %29, %27 : f32
            %32 = arith.addf %30, %31 : f32
            %33 = arith.select %21, %cst_13, %cst_12 : f32
            %34 = arith.mulf %32, %27 : f32
            %35 = arith.addf %33, %34 : f32
            %36 = arith.select %21, %cst_11, %cst_10 : f32
            %37 = arith.mulf %35, %27 : f32
            %38 = arith.addf %36, %37 : f32
            %39 = arith.select %21, %cst_9, %cst_8 : f32
            %40 = arith.mulf %38, %27 : f32
            %41 = arith.addf %39, %40 : f32
            %42 = arith.select %21, %cst_7, %cst_6 : f32
            %43 = arith.mulf %41, %27 : f32
            %44 = arith.addf %42, %43 : f32
            %45 = arith.select %21, %cst_5, %cst_4 : f32
            %46 = arith.mulf %44, %27 : f32
            %47 = arith.addf %45, %46 : f32
            %48 = arith.select %21, %cst_3, %cst_2 : f32
            %49 = arith.mulf %47, %27 : f32
            %50 = math.absf %16 : f32
            %51 = arith.addf %48, %49 : f32
            %52 = arith.cmpf oeq, %50, %cst_1 : f32
            %53 = arith.mulf %16, %cst_0 : f32
            %54 = arith.mulf %51, %16 : f32
            %55 = arith.select %52, %53, %54 : f32
            %56 = arith.mulf %55, %cst : f32
            %inserted = tensor.insert %56 into %arg6[%8] : tensor<4096xf32>
            scf.yield %inserted : tensor<4096xf32>
          }
          scf.yield %7 : tensor<4096xf32>
        } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
        scf.yield %6 : tensor<4096xf32>
      } else {
        scf.yield %arg2 : tensor<4096xf32>
      }
      scf.yield %5 : tensor<4096xf32>
    } else {
      %3 = arith.cmpi eq, %0, %c5 : index
      %4 = scf.if %3 -> (tensor<4096xf32>) {
        %5 = scf.for %arg3 = %c0 to %c9 step %c1 iter_args(%arg4 = %arg2) -> (tensor<4096xf32>) {
          %6 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4096xf32>) {
            %7 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d1 * 64 + d0 + 3520), domain: d0 in [0, 63], d1 in [0, 8]">(%arg5, %arg3)
            %extracted = tensor.extract %arg0[%7] : tensor<4096xi32>
            %extracted_26 = tensor.extract %arg1[%7] : tensor<4096xi32>
            %8 = arith.xori %extracted, %extracted_26 : i32
            %9 = arith.shrui %8, %c9_i32 : i32
            %10 = arith.ori %9, %c1065353216_i32 : i32
            %11 = arith.bitcast %10 : i32 to f32
            %12 = arith.addf %11, %cst_21 : f32
            %13 = arith.mulf %12, %cst_20 : f32
            %14 = arith.addf %13, %cst_19 : f32
            %15 = arith.maximumf %14, %cst_19 : f32
            %16 = arith.negf %15 : f32
            %17 = arith.mulf %15, %16 : f32
            %18 = math.log1p %17 : f32
            %19 = arith.negf %18 : f32
            %20 = arith.cmpf olt, %19, %cst_18 : f32
            %21 = arith.select %20, %cst_22, %cst_23 : f32
            %22 = arith.select %20, %cst_24, %cst_25 : f32
            %23 = math.sqrt %19 : f32
            %24 = arith.addf %19, %cst_17 : f32
            %25 = arith.addf %23, %cst_16 : f32
            %26 = arith.select %20, %24, %25 : f32
            %27 = arith.mulf %21, %26 : f32
            %28 = arith.addf %22, %27 : f32
            %29 = arith.select %20, %cst_15, %cst_14 : f32
            %30 = arith.mulf %28, %26 : f32
            %31 = arith.addf %29, %30 : f32
            %32 = arith.select %20, %cst_13, %cst_12 : f32
            %33 = arith.mulf %31, %26 : f32
            %34 = arith.addf %32, %33 : f32
            %35 = arith.select %20, %cst_11, %cst_10 : f32
            %36 = arith.mulf %34, %26 : f32
            %37 = arith.addf %35, %36 : f32
            %38 = arith.select %20, %cst_9, %cst_8 : f32
            %39 = arith.mulf %37, %26 : f32
            %40 = arith.addf %38, %39 : f32
            %41 = arith.select %20, %cst_7, %cst_6 : f32
            %42 = arith.mulf %40, %26 : f32
            %43 = arith.addf %41, %42 : f32
            %44 = arith.select %20, %cst_5, %cst_4 : f32
            %45 = arith.mulf %43, %26 : f32
            %46 = arith.addf %44, %45 : f32
            %47 = arith.select %20, %cst_3, %cst_2 : f32
            %48 = arith.mulf %46, %26 : f32
            %49 = math.absf %15 : f32
            %50 = arith.addf %47, %48 : f32
            %51 = arith.cmpf oeq, %49, %cst_1 : f32
            %52 = arith.mulf %15, %cst_0 : f32
            %53 = arith.mulf %50, %15 : f32
            %54 = arith.select %51, %52, %53 : f32
            %55 = arith.mulf %54, %cst : f32
            %inserted = tensor.insert %55 into %arg6[%7] : tensor<4096xf32>
            scf.yield %inserted : tensor<4096xf32>
          }
          scf.yield %6 : tensor<4096xf32>
        } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
        scf.yield %5 : tensor<4096xf32>
      } else {
        scf.yield %arg2 : tensor<4096xf32>
      }
      scf.yield %4 : tensor<4096xf32>
    }
    return %2 : tensor<4096xf32>
  }
}