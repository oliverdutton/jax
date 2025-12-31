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
    %8 = llvm.load %7 invariant dereferenceable<bytes = 8192> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %2[3, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %10 = llvm.load %9 invariant dereferenceable<bytes = 8192> : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %2[4, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %12 = llvm.load %11 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %2[5, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %14 = llvm.load %13 invariant dereferenceable<bytes = 8192> : !llvm.ptr -> !llvm.ptr
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
  llvm.func internal @broadcast_add_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, llvm.noalias, xla.invariant}, %arg3: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, llvm.noalias, xla.invariant}, %arg4: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg5: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, llvm.noalias}, %arg6: i64, %arg7: i64, %arg8: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(1024 : index) : i64
    %1 = llvm.mlir.constant(32 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(32 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.icmp "sge" %arg6, %2 : i64
    %8 = llvm.icmp "sle" %arg6, %6 : i64
    %9 = llvm.and %7, %8 : i1
    llvm.cond_br %9, ^bb1, ^bb8
  ^bb1:  // pred: ^bb0
    %10 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %11 = llvm.load %10 invariant : !llvm.ptr -> i32
    %12 = llvm.sub %3, %11 : i32
    %13 = llvm.icmp "ult" %11, %3 : i32
    %14 = llvm.icmp "ult" %12, %3 : i32
    %15 = llvm.getelementptr inbounds %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %16 = llvm.load %15 invariant : !llvm.ptr -> i32
    %17 = llvm.sub %3, %16 : i32
    %18 = llvm.icmp "ult" %16, %3 : i32
    %19 = llvm.icmp "ult" %17, %3 : i32
    %20 = llvm.getelementptr inbounds %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %21 = llvm.load %20 invariant : !llvm.ptr -> i32
    %22 = llvm.sub %3, %21 : i32
    %23 = llvm.icmp "ult" %21, %3 : i32
    %24 = llvm.icmp "ult" %22, %3 : i32
    %25 = llvm.getelementptr inbounds %arg1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %26 = llvm.load %25 invariant : !llvm.ptr -> i32
    %27 = llvm.sub %3, %26 : i32
    %28 = llvm.icmp "ult" %26, %3 : i32
    %29 = llvm.icmp "ult" %27, %3 : i32
    %30 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %31 = llvm.load %30 invariant : !llvm.ptr -> i32
    %32 = llvm.getelementptr inbounds %arg4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %33 = llvm.load %32 invariant : !llvm.ptr -> i32
    %34 = llvm.add %33, %5 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %35 = llvm.mul %arg6, %0 overflow<nsw> : i64
    llvm.br ^bb2(%2 : i64)
  ^bb2(%36: i64):  // 2 preds: ^bb1, ^bb6
    %37 = llvm.icmp "slt" %36, %1 : i64
    llvm.cond_br %37, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    %38 = llvm.mul %36, %1 overflow<nsw> : i64
    %39 = llvm.add %35, %38 overflow<nsw> : i64
    llvm.br ^bb4(%2 : i64)
  ^bb4(%40: i64):  // 2 preds: ^bb3, ^bb5
    %41 = llvm.icmp "slt" %40, %1 : i64
    llvm.cond_br %41, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %42 = llvm.add %39, %40 overflow<nsw> : i64
    %43 = llvm.getelementptr inbounds %arg2[0, %42] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x i32>
    %44 = llvm.load %43 invariant : !llvm.ptr -> i32
    %45 = llvm.shl %44, %11 : i32
    %46 = llvm.select %13, %45, %4 : i1, i32
    %47 = llvm.lshr %44, %12 : i32
    %48 = llvm.select %14, %47, %4 : i1, i32
    %49 = llvm.getelementptr inbounds %arg3[0, %42] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x i32>
    %50 = llvm.load %49 invariant : !llvm.ptr -> i32
    %51 = llvm.or %46, %48 : i32
    %52 = llvm.add %50, %44 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %53 = llvm.xor %52, %51 : i32
    %54 = llvm.shl %53, %16 : i32
    %55 = llvm.select %18, %54, %4 : i1, i32
    %56 = llvm.lshr %53, %17 : i32
    %57 = llvm.select %19, %56, %4 : i1, i32
    %58 = llvm.or %55, %57 : i32
    %59 = llvm.add %52, %53 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %60 = llvm.xor %59, %58 : i32
    %61 = llvm.shl %60, %21 : i32
    %62 = llvm.select %23, %61, %4 : i1, i32
    %63 = llvm.lshr %60, %22 : i32
    %64 = llvm.select %24, %63, %4 : i1, i32
    %65 = llvm.or %62, %64 : i32
    %66 = llvm.add %59, %60 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %67 = llvm.xor %66, %65 : i32
    %68 = llvm.shl %67, %26 : i32
    %69 = llvm.select %28, %68, %4 : i1, i32
    %70 = llvm.lshr %67, %27 : i32
    %71 = llvm.select %29, %70, %4 : i1, i32
    %72 = llvm.add %66, %67 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %73 = llvm.or %69, %71 : i32
    %74 = llvm.xor %72, %73 : i32
    %75 = llvm.add %74, %31 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %76 = llvm.add %75, %34 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %77 = llvm.getelementptr inbounds %arg5[0, %42] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x i32>
    llvm.store %76, %77 : i32, !llvm.ptr
    %78 = llvm.add %40, %6 : i64
    llvm.br ^bb4(%78 : i64)
  ^bb6:  // pred: ^bb4
    %79 = llvm.add %36, %6 : i64
    llvm.br ^bb2(%79 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb7:  // pred: ^bb2
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb0, ^bb7
    llvm.return
  }
}