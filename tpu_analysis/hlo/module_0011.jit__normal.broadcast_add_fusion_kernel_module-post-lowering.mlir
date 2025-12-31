module @broadcast_add_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @broadcast_add_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
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
    %12 = llvm.load %11 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %2[5, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %14 = llvm.load %13 invariant dereferenceable<bytes = 16384> : !llvm.ptr -> !llvm.ptr
    %15 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %16 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %17 = llvm.getelementptr inbounds %16[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %18 = llvm.load %17 invariant : !llvm.ptr -> i64
    %19 = llvm.getelementptr inbounds %16[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %20 = llvm.load %19 invariant : !llvm.ptr -> i64
    %21 = llvm.getelementptr inbounds %16[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %22 = llvm.load %21 invariant : !llvm.ptr -> i64
    llvm.call @broadcast_add_fusion_wrapped(%4, %6, %8, %10, %12, %14, %18, %20, %22) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @broadcast_add_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, llvm.noalias, xla.invariant}, %arg3: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, llvm.noalias, xla.invariant}, %arg4: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg5: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, llvm.noalias}, %arg6: i64, %arg7: i64, %arg8: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(3328 : index) : i64
    %1 = llvm.mlir.constant(832 : index) : i64
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = llvm.mlir.constant(12 : index) : i64
    %4 = llvm.mlir.constant(64 : index) : i64
    %5 = llvm.mlir.constant(13 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(32 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(3 : index) : i64
    %12 = llvm.icmp "sle" %arg6, %11 : i64
    llvm.cond_br %12, ^bb1, ^bb10
  ^bb1:  // pred: ^bb0
    %13 = llvm.icmp "sge" %arg6, %6 : i64
    llvm.cond_br %13, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %14 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %15 = llvm.load %14 invariant : !llvm.ptr -> i32
    %16 = llvm.sub %7, %15 : i32
    %17 = llvm.icmp "ult" %15, %7 : i32
    %18 = llvm.icmp "ult" %16, %7 : i32
    %19 = llvm.getelementptr inbounds %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %20 = llvm.load %19 invariant : !llvm.ptr -> i32
    %21 = llvm.sub %7, %20 : i32
    %22 = llvm.icmp "ult" %20, %7 : i32
    %23 = llvm.icmp "ult" %21, %7 : i32
    %24 = llvm.getelementptr inbounds %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %25 = llvm.load %24 invariant : !llvm.ptr -> i32
    %26 = llvm.sub %7, %25 : i32
    %27 = llvm.icmp "ult" %25, %7 : i32
    %28 = llvm.icmp "ult" %26, %7 : i32
    %29 = llvm.getelementptr inbounds %arg1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %30 = llvm.load %29 invariant : !llvm.ptr -> i32
    %31 = llvm.sub %7, %30 : i32
    %32 = llvm.icmp "ult" %30, %7 : i32
    %33 = llvm.icmp "ult" %31, %7 : i32
    %34 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %35 = llvm.load %34 invariant : !llvm.ptr -> i32
    %36 = llvm.getelementptr inbounds %arg4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %37 = llvm.load %36 invariant : !llvm.ptr -> i32
    %38 = llvm.add %37, %9 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %39 = llvm.mul %arg6, %1 overflow<nsw> : i64
    llvm.br ^bb3(%6 : i64)
  ^bb3(%40: i64):  // 2 preds: ^bb2, ^bb7
    %41 = llvm.icmp "slt" %40, %5 : i64
    llvm.cond_br %41, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %42 = llvm.mul %40, %4 overflow<nsw> : i64
    %43 = llvm.add %39, %42 overflow<nsw> : i64
    llvm.br ^bb5(%6 : i64)
  ^bb5(%44: i64):  // 2 preds: ^bb4, ^bb6
    %45 = llvm.icmp "slt" %44, %4 : i64
    llvm.cond_br %45, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %46 = llvm.add %43, %44 overflow<nsw> : i64
    %47 = llvm.getelementptr inbounds %arg2[0, %46] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %48 = llvm.load %47 invariant : !llvm.ptr -> i32
    %49 = llvm.shl %48, %15 : i32
    %50 = llvm.select %17, %49, %8 : i1, i32
    %51 = llvm.lshr %48, %16 : i32
    %52 = llvm.select %18, %51, %8 : i1, i32
    %53 = llvm.getelementptr inbounds %arg3[0, %46] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %54 = llvm.load %53 invariant : !llvm.ptr -> i32
    %55 = llvm.or %50, %52 : i32
    %56 = llvm.add %54, %48 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %57 = llvm.xor %56, %55 : i32
    %58 = llvm.shl %57, %20 : i32
    %59 = llvm.select %22, %58, %8 : i1, i32
    %60 = llvm.lshr %57, %21 : i32
    %61 = llvm.select %23, %60, %8 : i1, i32
    %62 = llvm.or %59, %61 : i32
    %63 = llvm.add %56, %57 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %64 = llvm.xor %63, %62 : i32
    %65 = llvm.shl %64, %25 : i32
    %66 = llvm.select %27, %65, %8 : i1, i32
    %67 = llvm.lshr %64, %26 : i32
    %68 = llvm.select %28, %67, %8 : i1, i32
    %69 = llvm.or %66, %68 : i32
    %70 = llvm.add %63, %64 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %71 = llvm.xor %70, %69 : i32
    %72 = llvm.shl %71, %30 : i32
    %73 = llvm.select %32, %72, %8 : i1, i32
    %74 = llvm.lshr %71, %31 : i32
    %75 = llvm.select %33, %74, %8 : i1, i32
    %76 = llvm.add %70, %71 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %77 = llvm.or %73, %75 : i32
    %78 = llvm.xor %76, %77 : i32
    %79 = llvm.add %78, %35 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %80 = llvm.add %79, %38 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %81 = llvm.getelementptr inbounds %arg5[0, %46] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    llvm.store %80, %81 : i32, !llvm.ptr
    %82 = llvm.add %44, %10 : i64
    llvm.br ^bb5(%82 : i64)
  ^bb7:  // pred: ^bb5
    %83 = llvm.add %40, %10 : i64
    llvm.br ^bb3(%83 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb8:  // 2 preds: ^bb3, ^bb12
    llvm.br ^bb9
  ^bb9:  // 3 preds: ^bb1, ^bb8, ^bb10
    llvm.br ^bb17
  ^bb10:  // pred: ^bb0
    %84 = llvm.icmp "eq" %arg6, %2 : i64
    llvm.cond_br %84, ^bb11, ^bb9
  ^bb11:  // pred: ^bb10
    %85 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %86 = llvm.load %85 invariant : !llvm.ptr -> i32
    %87 = llvm.sub %7, %86 : i32
    %88 = llvm.icmp "ult" %86, %7 : i32
    %89 = llvm.icmp "ult" %87, %7 : i32
    %90 = llvm.getelementptr inbounds %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %91 = llvm.load %90 invariant : !llvm.ptr -> i32
    %92 = llvm.sub %7, %91 : i32
    %93 = llvm.icmp "ult" %91, %7 : i32
    %94 = llvm.icmp "ult" %92, %7 : i32
    %95 = llvm.getelementptr inbounds %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %96 = llvm.load %95 invariant : !llvm.ptr -> i32
    %97 = llvm.sub %7, %96 : i32
    %98 = llvm.icmp "ult" %96, %7 : i32
    %99 = llvm.icmp "ult" %97, %7 : i32
    %100 = llvm.getelementptr inbounds %arg1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %101 = llvm.load %100 invariant : !llvm.ptr -> i32
    %102 = llvm.sub %7, %101 : i32
    %103 = llvm.icmp "ult" %101, %7 : i32
    %104 = llvm.icmp "ult" %102, %7 : i32
    %105 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %106 = llvm.load %105 invariant : !llvm.ptr -> i32
    %107 = llvm.getelementptr inbounds %arg4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %108 = llvm.load %107 invariant : !llvm.ptr -> i32
    %109 = llvm.add %108, %9 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    llvm.br ^bb12(%6 : i64)
  ^bb12(%110: i64):  // 2 preds: ^bb11, ^bb16
    %111 = llvm.icmp "slt" %110, %3 : i64
    llvm.cond_br %111, ^bb13, ^bb8
  ^bb13:  // pred: ^bb12
    %112 = llvm.mul %110, %4 overflow<nsw> : i64
    llvm.br ^bb14(%6 : i64)
  ^bb14(%113: i64):  // 2 preds: ^bb13, ^bb15
    %114 = llvm.icmp "slt" %113, %4 : i64
    llvm.cond_br %114, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %115 = llvm.add %112, %113 overflow<nsw> : i64
    %116 = llvm.add %115, %0 overflow<nsw> : i64
    %117 = llvm.getelementptr inbounds %arg2[0, %116] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %118 = llvm.load %117 invariant : !llvm.ptr -> i32
    %119 = llvm.shl %118, %86 : i32
    %120 = llvm.select %88, %119, %8 : i1, i32
    %121 = llvm.lshr %118, %87 : i32
    %122 = llvm.select %89, %121, %8 : i1, i32
    %123 = llvm.getelementptr inbounds %arg3[0, %116] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %124 = llvm.load %123 invariant : !llvm.ptr -> i32
    %125 = llvm.or %120, %122 : i32
    %126 = llvm.add %124, %118 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %127 = llvm.xor %126, %125 : i32
    %128 = llvm.shl %127, %91 : i32
    %129 = llvm.select %93, %128, %8 : i1, i32
    %130 = llvm.lshr %127, %92 : i32
    %131 = llvm.select %94, %130, %8 : i1, i32
    %132 = llvm.or %129, %131 : i32
    %133 = llvm.add %126, %127 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %134 = llvm.xor %133, %132 : i32
    %135 = llvm.shl %134, %96 : i32
    %136 = llvm.select %98, %135, %8 : i1, i32
    %137 = llvm.lshr %134, %97 : i32
    %138 = llvm.select %99, %137, %8 : i1, i32
    %139 = llvm.or %136, %138 : i32
    %140 = llvm.add %133, %134 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %141 = llvm.xor %140, %139 : i32
    %142 = llvm.shl %141, %101 : i32
    %143 = llvm.select %103, %142, %8 : i1, i32
    %144 = llvm.lshr %141, %102 : i32
    %145 = llvm.select %104, %144, %8 : i1, i32
    %146 = llvm.add %140, %141 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %147 = llvm.or %143, %145 : i32
    %148 = llvm.xor %146, %147 : i32
    %149 = llvm.add %148, %106 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %150 = llvm.add %149, %109 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %151 = llvm.getelementptr inbounds %arg5[0, %116] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    llvm.store %150, %151 : i32, !llvm.ptr
    %152 = llvm.add %113, %10 : i64
    llvm.br ^bb14(%152 : i64)
  ^bb16:  // pred: ^bb14
    %153 = llvm.add %110, %10 : i64
    llvm.br ^bb12(%153 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb17:  // pred: ^bb9
    llvm.return
  }
}