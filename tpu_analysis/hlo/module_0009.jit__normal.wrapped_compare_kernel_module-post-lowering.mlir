module @wrapped_compare_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @wrapped_compare(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 1> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %10 = llvm.load %9 : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %10[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    %15 = llvm.getelementptr inbounds %10[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %16 = llvm.load %15 invariant : !llvm.ptr -> i64
    llvm.call @wrapped_compare_wrapped(%4, %6, %8, %12, %14, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @wrapped_compare_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 1 : index, llvm.noalias}, %arg3: i64, %arg4: i64, %arg5: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %1 = llvm.load %0 invariant : !llvm.ptr -> i32
    %2 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %3 = llvm.load %2 invariant : !llvm.ptr -> i32
    %4 = llvm.icmp "slt" %1, %3 : i32
    %5 = llvm.zext %4 : i1 to i8
    %6 = llvm.getelementptr inbounds %arg2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i8>
    llvm.store %5, %6 : i8, !llvm.ptr
    llvm.return
  }
}