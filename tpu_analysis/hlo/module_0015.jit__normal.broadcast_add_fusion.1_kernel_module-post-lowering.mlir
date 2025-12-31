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
    %8 = llvm.load %7 invariant dereferenceable<bytes = 8192> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %2[3, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %10 = llvm.load %9 invariant dereferenceable<bytes = 8192> : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %2[4, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %12 = llvm.load %11 invariant dereferenceable<bytes = 8192> : !llvm.ptr -> !llvm.ptr
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
  llvm.func internal @broadcast_add_fusion.1_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, llvm.noalias, xla.invariant}, %arg3: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, llvm.noalias, xla.invariant}, %arg4: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, llvm.noalias}, %arg5: i64, %arg6: i64, %arg7: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(1024 : index) : i64
    %1 = llvm.mlir.constant(32 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(32 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.icmp "sge" %arg5, %2 : i64
    %7 = llvm.icmp "sle" %arg5, %5 : i64
    %8 = llvm.and %6, %7 : i1
    llvm.cond_br %8, ^bb1, ^bb8
  ^bb1:  // pred: ^bb0
    %9 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %10 = llvm.load %9 invariant : !llvm.ptr -> i32
    %11 = llvm.sub %3, %10 : i32
    %12 = llvm.icmp "ult" %10, %3 : i32
    %13 = llvm.icmp "ult" %11, %3 : i32
    %14 = llvm.getelementptr inbounds %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %15 = llvm.load %14 invariant : !llvm.ptr -> i32
    %16 = llvm.sub %3, %15 : i32
    %17 = llvm.icmp "ult" %15, %3 : i32
    %18 = llvm.icmp "ult" %16, %3 : i32
    %19 = llvm.getelementptr inbounds %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %20 = llvm.load %19 invariant : !llvm.ptr -> i32
    %21 = llvm.sub %3, %20 : i32
    %22 = llvm.icmp "ult" %20, %3 : i32
    %23 = llvm.icmp "ult" %21, %3 : i32
    %24 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %25 = llvm.load %24 invariant : !llvm.ptr -> i32
    %26 = llvm.mul %arg5, %0 overflow<nsw> : i64
    llvm.br ^bb2(%2 : i64)
  ^bb2(%27: i64):  // 2 preds: ^bb1, ^bb6
    %28 = llvm.icmp "slt" %27, %1 : i64
    llvm.cond_br %28, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    %29 = llvm.mul %27, %1 overflow<nsw> : i64
    %30 = llvm.add %26, %29 overflow<nsw> : i64
    llvm.br ^bb4(%2 : i64)
  ^bb4(%31: i64):  // 2 preds: ^bb3, ^bb5
    %32 = llvm.icmp "slt" %31, %1 : i64
    llvm.cond_br %32, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %33 = llvm.add %30, %31 overflow<nsw> : i64
    %34 = llvm.getelementptr inbounds %arg2[0, %33] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x i32>
    %35 = llvm.load %34 invariant : !llvm.ptr -> i32
    %36 = llvm.shl %35, %10 : i32
    %37 = llvm.select %12, %36, %4 : i1, i32
    %38 = llvm.lshr %35, %11 : i32
    %39 = llvm.select %13, %38, %4 : i1, i32
    %40 = llvm.getelementptr inbounds %arg3[0, %33] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x i32>
    %41 = llvm.load %40 invariant : !llvm.ptr -> i32
    %42 = llvm.or %37, %39 : i32
    %43 = llvm.add %41, %35 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %44 = llvm.xor %43, %42 : i32
    %45 = llvm.shl %44, %15 : i32
    %46 = llvm.select %17, %45, %4 : i1, i32
    %47 = llvm.lshr %44, %16 : i32
    %48 = llvm.select %18, %47, %4 : i1, i32
    %49 = llvm.or %46, %48 : i32
    %50 = llvm.add %43, %44 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %51 = llvm.xor %50, %49 : i32
    %52 = llvm.shl %51, %20 : i32
    %53 = llvm.select %22, %52, %4 : i1, i32
    %54 = llvm.lshr %51, %21 : i32
    %55 = llvm.select %23, %54, %4 : i1, i32
    %56 = llvm.or %53, %55 : i32
    %57 = llvm.add %50, %51 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %58 = llvm.xor %57, %56 : i32
    %59 = llvm.add %57, %58 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %60 = llvm.add %59, %25 {xla.range = [-9223372036854775808 : index, 9223372036854775807 : index]} : i32
    %61 = llvm.getelementptr inbounds %arg4[0, %33] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x i32>
    llvm.store %60, %61 : i32, !llvm.ptr
    %62 = llvm.add %31, %5 : i64
    llvm.br ^bb4(%62 : i64)
  ^bb6:  // pred: ^bb4
    %63 = llvm.add %27, %5 : i64
    llvm.br ^bb2(%63 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb7:  // pred: ^bb2
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb0, ^bb7
    llvm.return
  }
}