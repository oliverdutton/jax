module @broadcast_add_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @broadcast_add_fusion.1(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 16> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 16384> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %2[3, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %10 = llvm.load %9 invariant dereferenceable<bytes = 16384> : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %2[4, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %12 = llvm.load %11 invariant dereferenceable<bytes = 16384> : !llvm.ptr -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %14 = llvm.load %13 : !llvm.ptr -> !llvm.ptr
    %15 = llvm.getelementptr inbounds %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %16 = llvm.load %15 invariant : !llvm.ptr -> i64
    %17 = llvm.getelementptr inbounds %14[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %18 = llvm.load %17 invariant : !llvm.ptr -> i64
    %19 = llvm.getelementptr inbounds %14[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %20 = llvm.load %19 invariant : !llvm.ptr -> i64
    llvm.call @broadcast_add_fusion.1_wrapped(%4, %6, %8, %10, %12, %16, %18, %20) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @broadcast_add_fusion.1_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, llvm.noalias, xla.invariant}, %arg3: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, llvm.noalias, xla.invariant}, %arg4: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, llvm.noalias}, %arg5: i64, %arg6: i64, %arg7: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(3328 : index) : i64
    %1 = llvm.mlir.constant(832 : index) : i64
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = llvm.mlir.constant(12 : index) : i64
    %4 = llvm.mlir.constant(64 : index) : i64
    %5 = llvm.mlir.constant(13 : index) : i64
    %6 = llvm.mlir.constant(3 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.icmp "sle" %arg5, %6 : i64
    llvm.cond_br %11, ^bb1, ^bb10
  ^bb1:  // pred: ^bb0
    %12 = llvm.icmp "sge" %arg5, %7 : i64
    llvm.cond_br %12, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %13 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i32
    %15 = llvm.sub %8, %14 : i32
    %16 = llvm.icmp "ult" %14, %8 : i32
    %17 = llvm.icmp "ult" %15, %8 : i32
    %18 = llvm.getelementptr inbounds %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %19 = llvm.load %18 invariant : !llvm.ptr -> i32
    %20 = llvm.sub %8, %19 : i32
    %21 = llvm.icmp "ult" %19, %8 : i32
    %22 = llvm.icmp "ult" %20, %8 : i32
    %23 = llvm.getelementptr inbounds %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %24 = llvm.load %23 invariant : !llvm.ptr -> i32
    %25 = llvm.sub %8, %24 : i32
    %26 = llvm.icmp "ult" %24, %8 : i32
    %27 = llvm.icmp "ult" %25, %8 : i32
    %28 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %29 = llvm.load %28 invariant : !llvm.ptr -> i32
    %30 = llvm.mul %arg5, %1 overflow<nsw> : i64
    llvm.br ^bb3(%7 : i64)
  ^bb3(%31: i64):  // 2 preds: ^bb2, ^bb7
    %32 = llvm.icmp "slt" %31, %5 : i64
    llvm.cond_br %32, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %33 = llvm.mul %31, %4 overflow<nsw> : i64
    %34 = llvm.add %30, %33 overflow<nsw> : i64
    llvm.br ^bb5(%7 : i64)
  ^bb5(%35: i64):  // 2 preds: ^bb4, ^bb6
    %36 = llvm.icmp "slt" %35, %4 : i64
    llvm.cond_br %36, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %37 = llvm.add %34, %35 overflow<nsw> : i64
    %38 = llvm.getelementptr inbounds %arg2[0, %37] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %39 = llvm.load %38 invariant : !llvm.ptr -> i32
    %40 = llvm.shl %39, %14 : i32
    %41 = llvm.select %16, %40, %9 : i1, i32
    %42 = llvm.lshr %39, %15 : i32
    %43 = llvm.select %17, %42, %9 : i1, i32
    %44 = llvm.getelementptr inbounds %arg3[0, %37] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %45 = llvm.load %44 invariant : !llvm.ptr -> i32
    %46 = llvm.or %41, %43 : i32
    %47 = llvm.add %45, %39 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %48 = llvm.xor %47, %46 : i32
    %49 = llvm.shl %48, %19 : i32
    %50 = llvm.select %21, %49, %9 : i1, i32
    %51 = llvm.lshr %48, %20 : i32
    %52 = llvm.select %22, %51, %9 : i1, i32
    %53 = llvm.or %50, %52 : i32
    %54 = llvm.add %47, %48 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %55 = llvm.xor %54, %53 : i32
    %56 = llvm.shl %55, %24 : i32
    %57 = llvm.select %26, %56, %9 : i1, i32
    %58 = llvm.lshr %55, %25 : i32
    %59 = llvm.select %27, %58, %9 : i1, i32
    %60 = llvm.or %57, %59 : i32
    %61 = llvm.add %54, %55 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %62 = llvm.xor %61, %60 : i32
    %63 = llvm.add %61, %62 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %64 = llvm.add %63, %29 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %65 = llvm.getelementptr inbounds %arg4[0, %37] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    llvm.store %64, %65 : i32, !llvm.ptr
    %66 = llvm.add %35, %10 : i64
    llvm.br ^bb5(%66 : i64)
  ^bb7:  // pred: ^bb5
    %67 = llvm.add %31, %10 : i64
    llvm.br ^bb3(%67 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb8:  // 2 preds: ^bb3, ^bb12
    llvm.br ^bb9
  ^bb9:  // 3 preds: ^bb1, ^bb8, ^bb10
    llvm.br ^bb17
  ^bb10:  // pred: ^bb0
    %68 = llvm.icmp "eq" %arg5, %2 : i64
    llvm.cond_br %68, ^bb11, ^bb9
  ^bb11:  // pred: ^bb10
    %69 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %70 = llvm.load %69 invariant : !llvm.ptr -> i32
    %71 = llvm.sub %8, %70 : i32
    %72 = llvm.icmp "ult" %70, %8 : i32
    %73 = llvm.icmp "ult" %71, %8 : i32
    %74 = llvm.getelementptr inbounds %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %75 = llvm.load %74 invariant : !llvm.ptr -> i32
    %76 = llvm.sub %8, %75 : i32
    %77 = llvm.icmp "ult" %75, %8 : i32
    %78 = llvm.icmp "ult" %76, %8 : i32
    %79 = llvm.getelementptr inbounds %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %80 = llvm.load %79 invariant : !llvm.ptr -> i32
    %81 = llvm.sub %8, %80 : i32
    %82 = llvm.icmp "ult" %80, %8 : i32
    %83 = llvm.icmp "ult" %81, %8 : i32
    %84 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %85 = llvm.load %84 invariant : !llvm.ptr -> i32
    llvm.br ^bb12(%7 : i64)
  ^bb12(%86: i64):  // 2 preds: ^bb11, ^bb16
    %87 = llvm.icmp "slt" %86, %3 : i64
    llvm.cond_br %87, ^bb13, ^bb8
  ^bb13:  // pred: ^bb12
    %88 = llvm.mul %86, %4 overflow<nsw> : i64
    llvm.br ^bb14(%7 : i64)
  ^bb14(%89: i64):  // 2 preds: ^bb13, ^bb15
    %90 = llvm.icmp "slt" %89, %4 : i64
    llvm.cond_br %90, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %91 = llvm.add %88, %89 overflow<nsw> : i64
    %92 = llvm.add %91, %0 overflow<nsw> : i64
    %93 = llvm.getelementptr inbounds %arg2[0, %92] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %94 = llvm.load %93 invariant : !llvm.ptr -> i32
    %95 = llvm.shl %94, %70 : i32
    %96 = llvm.select %72, %95, %9 : i1, i32
    %97 = llvm.lshr %94, %71 : i32
    %98 = llvm.select %73, %97, %9 : i1, i32
    %99 = llvm.getelementptr inbounds %arg3[0, %92] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %100 = llvm.load %99 invariant : !llvm.ptr -> i32
    %101 = llvm.or %96, %98 : i32
    %102 = llvm.add %100, %94 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %103 = llvm.xor %102, %101 : i32
    %104 = llvm.shl %103, %75 : i32
    %105 = llvm.select %77, %104, %9 : i1, i32
    %106 = llvm.lshr %103, %76 : i32
    %107 = llvm.select %78, %106, %9 : i1, i32
    %108 = llvm.or %105, %107 : i32
    %109 = llvm.add %102, %103 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %110 = llvm.xor %109, %108 : i32
    %111 = llvm.shl %110, %80 : i32
    %112 = llvm.select %82, %111, %9 : i1, i32
    %113 = llvm.lshr %110, %81 : i32
    %114 = llvm.select %83, %113, %9 : i1, i32
    %115 = llvm.or %112, %114 : i32
    %116 = llvm.add %109, %110 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %117 = llvm.xor %116, %115 : i32
    %118 = llvm.add %116, %117 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %119 = llvm.add %118, %85 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %120 = llvm.getelementptr inbounds %arg4[0, %92] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    llvm.store %119, %120 : i32, !llvm.ptr
    %121 = llvm.add %89, %10 : i64
    llvm.br ^bb14(%121 : i64)
  ^bb16:  // pred: ^bb14
    %122 = llvm.add %86, %10 : i64
    llvm.br ^bb12(%122 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb17:  // pred: ^bb9
    llvm.return
  }
}